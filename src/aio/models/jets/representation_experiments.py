"""Single source of truth for Weaver, profiling, and pilot launches."""
from __future__ import annotations

import json
from pathlib import Path

CONFIG_ROOT = Path(__file__).resolve().parents[4] / "configs" / "experiments" / "representation_order"


def experiment_config(name):
    if not name or Path(name).name != name:
        raise ValueError("Experiment name must be a simple preset name")
    path = CONFIG_ROOT / f"{name}.json"
    if not path.is_file():
        raise ValueError(f"Unknown representation experiment {name!r}")
    document = json.loads(path.read_text())
    base = json.loads((CONFIG_ROOT / "common.json").read_text())
    return {**base, **document["model"]}
