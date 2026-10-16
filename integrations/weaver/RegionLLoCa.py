"""Weaver adapter; reuse the repository's JetClass-II evaluation implementation."""
from __future__ import annotations

import torch
from weaver.utils.logger import _logger
from weaver.utils.nn.tools import train_classification
from integrations.weaver.ParT import evaluate_classification_lean
from aio.models.jets.region_transformer import RegionTransformer
from aio.models.jets.representation_experiments import experiment_config, experiment_family


class RegionTransformerWrapper(torch.nn.Module):
    def __init__(self, export_embed=False, **kwargs):
        super().__init__()
        self.export_embed = export_embed
        self.mod = RegionTransformer(**kwargs)

    def forward(self, points, features, lorentz_vectors, mask):
        del points  # Local geometry is computed from four-momenta, not eta/phi arrays.
        if not self.export_embed:
            return self.mod(features, lorentz_vectors, mask)
        embedding, _ = self.mod.encode(features, lorentz_vectors, mask)
        output = self.mod.head(embedding)
        if self.mod.for_inference:
            output = output.softmax(-1)
        return torch.cat((output, embedding), -1)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"mod.cls_token"} if self.mod.cls_token is not None else set()


def get_model(data_config, **kwargs):
    # Weaver injects these flags; autocast/compilation are managed by its outer loop.
    kwargs.pop("use_amp", None)
    kwargs.pop("compile_model", None)
    preset = kwargs.pop("preset", "full_sv")
    if experiment_family(preset) != "region":
        raise ValueError("RegionLLoCa requires a region-family preset")
    geometry_checkpoint = kwargs.pop("geometry_checkpoint", None)
    freeze_geometry = kwargs.pop("freeze_geometry", bool(geometry_checkpoint))
    export_embed = kwargs.pop("export_embed", False)
    config = experiment_config(preset)
    config.update(kwargs)
    config["input_dim"] = len(data_config.input_dicts["pf_features"])
    if config.get("num_classes") is None:
        raise ValueError("Provide -o num_classes 188 for JetClass-II")
    model = RegionTransformerWrapper(export_embed=export_embed, **config)
    if geometry_checkpoint:
        model.mod.load_geometry(geometry_checkpoint, freeze=freeze_geometry)
    elif freeze_geometry:
        raise ValueError("freeze_geometry requires a trained geometry_checkpoint")
    _logger.info("RegionLLoCa %s: %s", preset, config)
    _logger.info("Grouping=%s: K=%d, geometry frozen=%s", model.mod.region_source,
                 model.mod.num_regions, model.mod.geometry_frozen)
    info = dict(input_names=list(data_config.input_names),
                input_shapes={k: (1,) + s[1:] for k, s in data_config.input_shapes.items()},
                output_names=["softmax"],
                dynamic_axes={**{k: {0: "N", 2: "n_" + k.split("_")[0]}
                                 for k in data_config.input_names}, "softmax": {0: "N"}})
    return model, info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()


def train_with_region_regularization(model, loss_func, *args, **kwargs):
    base = model
    while hasattr(base, "module"):
        base = base.module
    if isinstance(model, torch.nn.DataParallel):
        raise ValueError("Use one GPU or DDP; DataParallel cannot safely collect region auxiliary losses")

    def objective(logits, labels):
        loss = loss_func(logits, labels)
        auxiliary = base.mod.last_aux_loss
        if auxiliary is None:
            raise RuntimeError("Region forward must precede the loss")
        return loss + auxiliary

    result = train_classification(model, objective, *args, **kwargs)
    _logger.info("Region diagnostics (last training batch): %s",
                 {k: float(v) for k, v in base.mod.last_diagnostics.items()})
    return result


def get_train_fn(data_config, **kwargs):
    return train_with_region_regularization


def get_evaluate_fn(data_config, **kwargs):
    # Validation reports CE only; grouping penalties are optimization regularizers.
    return evaluate_classification_lean
