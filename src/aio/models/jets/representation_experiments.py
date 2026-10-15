"""Single source of truth for Weaver, profiling, and pilot launches."""
from __future__ import annotations

import json
from pathlib import Path

CONFIG_ROOT = Path(__file__).resolve().parents[4] / "configs" / "experiments" / "representation_order"


def _document(name):
    if not name or Path(name).name != name:
        raise ValueError("Experiment name must be a simple preset name")
    path = CONFIG_ROOT / f"{name}.json"
    if not path.is_file():
        raise ValueError(f"Unknown representation experiment {name!r}")
    return json.loads(path.read_text())


def experiment_family(name):
    return _document(name).get("family", "region")


def experiment_config(name):
    document = _document(name)
    base = (json.loads((CONFIG_ROOT / "common.json").read_text())
            if document.get("family", "region") == "region" else {})
    return {**base, **document["model"]}


def build_experiment(name, input_dim=7, num_classes=188, **overrides):
    """Unwrapped backbone for profiling and offline diagnostics."""
    config = {**experiment_config(name), **overrides}
    if experiment_family(name) == "part":
        from .particle_transformer import ParticleTransformer
        return ParticleTransformer(input_dim, num_classes, **config)
    from .region_transformer import RegionTransformer
    return RegionTransformer(input_dim, num_classes, **config)
