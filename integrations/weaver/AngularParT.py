"""Weaver adapter for actual ordinary-ParT baselines and angular-value extensions."""
from __future__ import annotations

import torch
from weaver.utils.logger import _logger
from integrations.weaver.ParT import (
    ParticleTransformerWrapper, get_loss, get_train_fn, get_evaluate_fn,
)
from aio.models.jets.angular_experiments import build_experiment, experiment_config


class AngularParTWrapper(ParticleTransformerWrapper):
    def __init__(self, preset, input_dim, num_classes, export_embed=False, **kwargs):
        torch.nn.Module.__init__(self)
        self.export_embed = export_embed
        self.mod = build_experiment(preset, input_dim=input_dim, num_classes=num_classes, **kwargs)


def get_model(data_config, **kwargs):
    preset = kwargs.pop("preset", "part_d2")
    num_classes = kwargs.pop("num_classes", None)
    if num_classes is None:
        raise ValueError("Provide -o num_classes 188 for JetClass-II")
    family, config = experiment_config(preset)
    _logger.info("Angular experiment %s (%s): %s", preset, family, {**config, **kwargs})
    model = AngularParTWrapper(preset, len(data_config.input_dicts["pf_features"]), num_classes, **kwargs)
    info = dict(input_names=list(data_config.input_names),
                input_shapes={k: (1,) + s[1:] for k, s in data_config.input_shapes.items()},
                output_names=["softmax"], dynamic_axes={
                    **{k: {0: "N", 2: "n_" + k.split("_")[0]} for k in data_config.input_names},
                    "softmax": {0: "N"}})
    return model, info
