"""Load the matched JetClass-II model presets by their filenames."""
import json
from pathlib import Path

PRESET_DIR = Path(__file__).with_name("presets")
PRESETS = tuple(sorted(p.stem for p in PRESET_DIR.glob("*.json") if p.stem != "common"))
DEFAULT_PRESET = "linear_pair_rotary_values15_d4"
PILOT = ("linear_pair_plain_values_d4", DEFAULT_PRESET, "plain_transformer_d4",
         "rotary_qk30_rotary_values15_d4")

# The proposal labels continue to work in saved Weaver commands.
LEGACY_NAMES = {
    "A": "part_pair_mlp64_d4",
    "B": "plain_transformer_d4",
    "C8": "part_pair_mlp8_d4",
    "C": "linear_pair_plain_values_d4",
    "H": "linear_pair_rotary_values15_d4",
    "H_small": "linear_pair_rotary_values6_d4",
    "E": "rotary_qk30_rotary_values15_d4",
    "E_small": "rotary_qk12_rotary_values6_d4",
    "D": "rotary_qk30_plain_values_d4",
    "F": "rest_harmonics_l2_plain_values_d4",
    "G": "low_rank8_plain_values_d4"
}


def canonical_preset(name):
    return LEGACY_NAMES.get(name, name)


def relative_config(preset, **overrides):
    name = canonical_preset(preset)
    if name not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}. Choose one of {', '.join(PRESETS)}")
    common = json.loads((PRESET_DIR / "common.json").read_text())
    config = json.loads((PRESET_DIR / f"{name}.json").read_text())
    return {**common, **config, **overrides}
