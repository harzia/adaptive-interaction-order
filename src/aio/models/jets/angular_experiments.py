"""Matched ordinary-ParT and angular-state experiments, separate from old pilots."""
from __future__ import annotations

import json
from pathlib import Path

CONFIG_ROOT = Path(__file__).resolve().parents[4] / "configs/experiments/angular_moments"


def experiment_config(name):
    if not name or Path(name).name != name or name == "common":
        raise ValueError("Use a simple angular experiment preset name")
    path = CONFIG_ROOT / f"{name}.json"
    if not path.is_file():
        raise ValueError(f"Unknown angular experiment {name!r}")
    document = json.loads(path.read_text())
    if document["family"] not in ("part", "angular_part"):
        raise ValueError("Unknown angular experiment family")
    return document["family"], {**json.loads((CONFIG_ROOT / "common.json").read_text()), **document["model"]}


def build_experiment(name, input_dim=7, num_classes=188, **overrides):
    family, config = experiment_config(name)
    if family == "part":
        from .particle_transformer import ParticleTransformer
        cls = ParticleTransformer
    else:
        from .angular_particle_transformer import AngularParticleTransformer
        cls = AngularParticleTransformer
    return cls(input_dim=input_dim, num_classes=num_classes, **{**config, **overrides})
