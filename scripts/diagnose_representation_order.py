#!/usr/bin/env python3
"""Inspect a trained regional model using already-preprocessed Weaver input tensors.

No training or raw-file preprocessing. The .pt input must contain features
[B,F,N], vectors [B,4,N] in px,py,pz,E order, and mask [B,1,N]. Optional labels
[B] enable --tensor-ablations. Use a held-out validation sample, not training data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch
from aio.core.geometry.diagnostics import assignment_statistics, tensor_interventions
from aio.models.jets.representation_experiments import build_experiment, experiment_family


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSON summary; per-jet tensors saved to matching .pt")
    parser.add_argument("--num-classes", type=int, default=188)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-jets", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--tensor-ablations", action="store_true")
    args = parser.parse_args()
    if min(args.max_jets, args.batch_size, args.threads) < 1:
        parser.error("Positive max-jets, batch-size, threads required")
    if args.output.suffix != ".json":
        parser.error("--output must end in .json")
    if args.output.resolve() in (args.inputs.resolve(), args.checkpoint.resolve()) or args.output.with_suffix(".pt").resolve() in (args.inputs.resolve(), args.checkpoint.resolve()):
        parser.error("Output paths must not overwrite inputs or checkpoint")
    torch.set_num_threads(args.threads)
    data = torch.load(args.inputs, map_location="cpu", weights_only=True)
    for key in ("features", "vectors", "mask"):
        if key not in data or not isinstance(data[key], torch.Tensor):
            parser.error(f"Missing tensor {key}")
    count = min(len(data["features"]), args.max_jets)
    if not count or any(len(data[k]) != len(data["features"]) for k in ("vectors", "mask")):
        parser.error("Input tensors must have matching nonzero batch sizes")
    if args.tensor_ablations and ("labels" not in data or len(data["labels"]) != len(data["features"])):
        parser.error("--tensor-ablations requires labels [B]")
    if experiment_family(args.preset) != "region":
        parser.error("Assignment diagnostics require a regional model")
    model = build_experiment(args.preset, input_dim=data["features"].shape[1], num_classes=args.num_classes)
    if not model.num_regions:
        parser.error("Assignment diagnostics require num_regions > 0")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("state_dict", payload)
    cleaned = {}
    for key, value in state.items():
        while key.startswith(("module.", "_orig_mod.", "mod.")):
            key = key.split(".", 1)[1]
        cleaned[key] = value
    model.load_state_dict(cleaned, strict=True)
    model = model.to(args.device).eval().requires_grad_(False)
    result = {}
    for start in range(0, count, args.batch_size):
        end = min(start + args.batch_size, count)
        x, v, mask = (data[k][start:end].to(args.device) for k in ("features", "vectors", "mask"))
        x, v = x.float(), v.float()
        with torch.no_grad():
            logits, g = model(x, v, mask, return_aux=True)
            stats = assignment_statistics(g["assignments"], mask[:, 0].bool(), v[:, [3, 0, 1, 2]].transpose(1, 2))
            stats.update(assignments=g["assignments"], frame_valid=g["frame_valid"], mask=mask[:, 0], logits=logits)
        if "labels" in data:
            labels = data["labels"][start:end].to(args.device).long()
            stats["labels"] = labels
        if "source_file_index" in data:
            stats["source_file_index"] = data["source_file_index"][start:end]
        if args.tensor_ablations:
            stats.update(tensor_interventions(model, x, v, mask, labels))
        for key, value in stats.items():
            result.setdefault(key, []).append(value.detach().cpu())
    result = {k: torch.cat(v) for k, v in result.items()}
    scalars = ["effective_occupied_regions", "hard_occupied_regions", "assignment_entropy",
               "normalized_assignment_entropy", "mean_max_assignment", "mean_region_overlap",
               "base_ce", "all_tensors_off_delta_ce", "all_tensors_off_changed_prediction"]
    summary = {k: float(result[k].float().mean()) for k in scalars if k in result}
    summary["invalid_frame_fraction"] = float((result["mask"].bool() & ~result["frame_valid"]).sum() / result["mask"].sum())
    if args.tensor_ablations:
        summary["fraction_positive_region_off_delta_ce"] = float((result["region_off_delta_ce"] > 0).float().mean())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    detail_path = args.output.with_suffix(".pt")
    torch.save(result, detail_path)
    report = dict(preset=args.preset, checkpoint=str(args.checkpoint), inputs=str(args.inputs),
                  num_jets=count, num_regions=model.num_regions, summary=summary, details=str(detail_path),
                  input_metadata=data.get("metadata", {}),
                  note="Offline diagnostics. Region IDs are exchangeable across models. Tensor ablations use labels and do not save compute.")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
