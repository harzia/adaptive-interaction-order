#!/usr/bin/env python3
"""Inspect a trained regional model using already-preprocessed Weaver input tensors.

No training or raw-file preprocessing. The .pt input must contain features
[B,F,N], vectors [B,4,N] in px,py,pz,E order, and mask [B,1,N]. Optional labels
[B] enable --tensor-ablations. Use a held-out validation sample, not training data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import torch
from aio.core.geometry.diagnostics import assignment_statistics, tensor_interventions, communication_interventions
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
    parser.add_argument("--communication-ablations", action="store_true",
                        help="Remove global messages and, where present, local/global pair biases")
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
    if "labels" in data and (not isinstance(data["labels"], torch.Tensor) or data["labels"].ndim != 1 or len(data["labels"]) != len(data["features"])):
        parser.error("labels must be a tensor [B] matching the input count")
    if (args.tensor_ablations or args.communication_ablations) and "labels" not in data:
        parser.error("Interventions require labels [B]")
    if count < len(data["features"]):
        print(f"Inspecting first {count}/{len(data['features'])} input jets; raise --max-jets to include all")
    if experiment_family(args.preset) != "region":
        parser.error("Assignment diagnostics require a regional model")
    model = build_experiment(args.preset, input_dim=data["features"].shape[1], num_classes=args.num_classes)
    if args.tensor_ablations and (not model.num_regions or not all(b.global_attention.tensors for b in model.blocks)):
        parser.error("--tensor-ablations requires a regional rank-2 preset")
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
            stats = (assignment_statistics(g["assignments"], mask[:, 0].bool(), v[:, [3, 0, 1, 2]].transpose(1, 2))
                     if model.num_regions else {"constituent_count": mask[:, 0].bool().sum(1)})
            if model.num_regions:
                stats["assignments"] = g["assignments"]
            if model.region_source == "ca":
                from aio.core.geometry.lund import ca_tree_regions
                tree = ca_tree_regions(v[:, [3, 0, 1, 2]].transpose(1, 2), mask[:, 0].bool(),
                                       model.num_regions, model.geometry.radius)
                # Align slots by constituent overlap; equal-pt nodes may have
                # different order in FastJet's batched and object interfaces.
                overlap = torch.einsum("bnk,bnl->bkl", g["assignments"], tree["assignments"])
                order = overlap.argmax(-1)
                aligned = tree["assignments"].gather(2, order[:, None].expand(-1, mask.shape[-1], -1))
                active = g["region_mask"]
                if not torch.equal(aligned * active[:, None], g["assignments"]):
                    raise RuntimeError("Batched and diagnostic C/A partitions disagree")
                stats.update({key: value.gather(1, order).masked_fill(~active, 0)
                              for key, value in tree.items() if key.startswith("lund_")})
            stats.update(frame_valid=g["frame_valid"], mask=mask[:, 0], logits=logits)
        if "labels" in data:
            labels = data["labels"][start:end].to(args.device).long()
            stats["labels"] = labels
            stats["base_ce"] = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
        for key in data:
            if key.startswith("observer__") or key in ("source_file_index", "source_selected_index"):
                stats[key] = data[key][start:end]
        if args.tensor_ablations:
            stats.update(tensor_interventions(model, x, v, mask, labels))
        if args.communication_ablations:
            stats.update(communication_interventions(model, x, v, mask, labels))
        for key, value in stats.items():
            result.setdefault(key, []).append(value.detach().cpu())
    result = {k: torch.cat(v) for k, v in result.items()}
    scalars = ["effective_occupied_regions", "hard_occupied_regions", "assignment_entropy",
               "normalized_assignment_entropy", "mean_max_assignment", "mean_region_overlap",
               "base_ce", "all_tensors_off_delta_ce", "all_tensors_off_changed_prediction"]
    summary = {k: float(result[k].float().mean()) for k in scalars if k in result}
    summary.update({k: float(v.float().mean()) for k, v in result.items()
                    if v.ndim == 1 and (k.endswith("_delta_ce") or k.endswith("_changed_prediction"))})
    summary["invalid_frame_fraction"] = float((result["mask"].bool() & ~result["frame_valid"]).sum() / result["mask"].sum())
    if args.tensor_ablations:
        active = result["region_mask"]
        summary["fraction_positive_region_off_delta_ce"] = float((result["region_off_delta_ce"][active] > 0).float().mean())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    detail_path = args.output.with_suffix(".pt")
    torch.save(result, detail_path)
    groups = {}
    if "labels" in result:
        # Reuse the evaluator's canonical groups rather than a second label map.
        from integrations.weaver.eval.jetclass2_eval import SIGNAL_GROUPS, QCD_INDICES
        for name, indices in {**SIGNAL_GROUPS, "label_QCD": QCD_INDICES.tolist()}.items():
            selected = torch.isin(result["labels"], torch.tensor(indices))
            if selected.any():
                groups[name.removeprefix("label_")] = dict(
                    num_jets=int(selected.sum()),
                    **{k: float(v[selected].float().mean()) for k, v in result.items()
                       if v.ndim == 1 and (k == "base_ce" or k.endswith("_delta_ce"))})
    with args.inputs.open("rb") as handle:
        input_digest = hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(handle.read()).hexdigest()
    report = dict(preset=args.preset, checkpoint=str(args.checkpoint), inputs=str(args.inputs),
                  input_sha256=input_digest,
                  num_jets=count, num_regions=model.num_regions, summary=summary, details=str(detail_path),
                  region_source=model.region_source, group_summaries=groups,
                  input_metadata=data.get("metadata", {}),
                  note="Offline diagnostics. Region IDs are exchangeable across models. Tensor ablations use labels and do not save compute.")
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
