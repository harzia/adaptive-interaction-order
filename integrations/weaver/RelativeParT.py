"""Weaver adapter for relative particle attention and its matched controls."""
import torch
from weaver.utils.logger import _logger

from integrations.weaver.part_training import ParticleTransformerWrapper, get_loss, get_train_fn, get_evaluate_fn
from aio.models.jets.relative_experiments import DEFAULT_PRESET, canonical_preset, relative_config
from aio.models.jets.relative_particle_transformer import RelativeParticleTransformer


class RelativeParTWrapper(ParticleTransformerWrapper):
    def __init__(self, export_embed=False, **cfg):
        torch.nn.Module.__init__(self)
        self.export_embed = export_embed
        self.mod = RelativeParticleTransformer(**cfg)

    def forward(self, points, features, lorentz_vectors, mask):
        return self.mod(features, lorentz_vectors, mask, points=points, export_embed=self.export_embed)


def get_model(data_config, preset=DEFAULT_PRESET, **kwargs):
    preset = canonical_preset(preset)
    cfg = relative_config(preset, input_dim=len(data_config.input_dicts["pf_features"]), **kwargs)
    if cfg.get("num_classes") is None:
        raise ValueError("Supply -o num_classes 188 for JetClass-II")
    _logger.info("Relative attention preset %s: %s", preset, cfg)
    wrapper = RelativeParTWrapper if "score" in cfg else ParticleTransformerWrapper
    names = list(data_config.input_names)
    info = dict(input_names=names, output_names=["softmax"],
                input_shapes={k: (1,) + s[1:] for k, s in data_config.input_shapes.items()},
                dynamic_axes={**{k: {0: "N", 2: "n_" + k.split("_")[0]} for k in names},
                              "softmax": {0: "N"}})
    return wrapper(**cfg), info
