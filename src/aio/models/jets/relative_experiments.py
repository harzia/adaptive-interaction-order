"""Relative-attention experiment arms. All inherit the actual part_d4 recipe."""
from .angular_experiments import experiment_config

PRESETS = {
    "A": {},  # ordinary ParT
    "B": {"pair_embed_dims": None},
    "C8": {"pair_embed_dims": [8, 8, 8]},
    "C": {"score": "linear", "rotate_values": False},
    "H": {"score": "linear"},
    "H_small": {"score": "linear", "value_bank": "small"},
    "E": {"score": "rotary"},
    "E_small": {"score": "rotary", "bank": "small"},
    "D": {"score": "rotary", "rotate_values": False},
    "F": {"score": "harmonic", "rotate_values": False},
    "G": {"score": "low_rank", "rotate_values": False},
}


def relative_config(preset, **overrides):
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}. Choose one of {', '.join(PRESETS)}")
    _, common = experiment_config("part_d4")
    if "score" in PRESETS[preset]:
        del common["pair_embed_dims"]
    return {**common, **PRESETS[preset], **overrides}
