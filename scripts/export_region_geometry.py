#!/usr/bin/env python3
"""Export grouping AND frame prediction, so an order ablation uses identical geometry."""
from __future__ import annotations

import argparse
from pathlib import Path
import torch
from aio.models.jets.region_transformer import RegionTransformer
from aio.models.jets.representation_experiments import experiment_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-dim", type=int, default=7)
    parser.add_argument("--num-classes", type=int, default=188)
    args = parser.parse_args()
    config = experiment_config(args.preset)
    if config["num_regions"] == 0:
        parser.error("A full-attention model has no learned regions to export")
    model = RegionTransformer(args.input_dim, args.num_classes, **config)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload)
    cleaned = {}
    for key, value in state.items():
        while key.startswith(("module.", "_orig_mod.", "mod.")):
            key = key.split(".", 1)[1]
        cleaned[key] = value
    # Strict load also checks the selected preset, width and classification head.
    model.load_state_dict(cleaned, strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save_geometry(args.output)
    print(f"Saved learned geometry for K={model.num_regions}: {args.output}")


if __name__ == "__main__":
    main()
