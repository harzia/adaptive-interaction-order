#!/usr/bin/env python3
"""Apply a fitted hard gate before expert evaluation, using selected sub-batches."""
from __future__ import annotations

import argparse
from pathlib import Path
import torch
from aio.core.routing.frozen_experts import FEATURE_NAMES, InputGate, hard_dispatch, load_expert
from cache_representation_experts import preprocessing_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("router", "inputs", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.batch_size, args.threads) < 1 or args.output.exists():
        parser.error("Positive batch size/threads and a new output path required")
    torch.set_num_threads(args.threads)
    saved = torch.load(args.router, map_location="cpu", weights_only=True)
    if saved.get("format") != "aio-frozen-router-v1" or saved["metadata"]["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("Unknown router/feature schema")
    metadata = saved["metadata"]
    data = torch.load(args.inputs, map_location="cpu", weights_only=True)
    if preprocessing_digest(data) != metadata["preprocessing_sha256"] or data["features"].shape[1] != metadata["input_dim"]:
        raise ValueError("Router/input preprocessing mismatch")
    state = saved["states"]["hard"]
    gate = InputGate(state["mean"], state["scale"], len(metadata["experts"]), saved["hidden"])
    gate.load_state_dict(state, strict=True)
    gate = gate.to(args.device).eval()
    experts = [load_expert(spec, metadata["input_dim"], len(metadata["class_names"]), args.device) for spec in metadata["experts"]]
    probabilities, choices = [], []
    with torch.no_grad():
        for start in range(0, len(data["features"]), args.batch_size):
            x, v, mask = (data[k][start:start+args.batch_size].to(args.device) for k in ("features", "vectors", "mask"))
            p, choice = hard_dispatch(experts, gate, x.float(), v.float(), mask)
            probabilities.append(p.cpu())
            choices.append(choice.cpu())
    if not probabilities:
        raise ValueError("Empty inputs")
    result = dict(probabilities=torch.cat(probabilities), expert_choice=torch.cat(choices),
                  class_names=metadata["class_names"], expert_names=[e["name"] for e in metadata["experts"]],
                  input_metadata=data.get("metadata", {}))
    for key in ("labels", "source_file_index", "source_selected_index"):
        if key in data:
            result[key] = data[key]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(f"Saved {len(result['expert_choice'])} predictions to {args.output}; only selected experts evaluated")


if __name__ == "__main__":
    main()
