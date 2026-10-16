#!/usr/bin/env python3
"""Capture a small fixed diagnostic sample using Weaver's actual preprocessing.

Reads the first selected jets from each explicit input file, without training
reweighting or shuffling. This is an inspection sample, not an unbiased estimate
of full validation performance. Select held-out files and reuse the output across
checkpoints. Supply the training run's resolved YAML when preprocessing is auto.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from weaver.utils.dataset import SimpleIterDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=Path, nargs="+", required=True)
    parser.add_argument("--data-config", type=Path, default=Path("configs/datasets/jetclass2/JetClassII_kin.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jets-per-file", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--label", default="truth_label")
    parser.add_argument("--label-ids", type=int, nargs="+", help="Only export these labels")
    parser.add_argument("--jets-per-label", type=int, help="Optional per-label cap across all files")
    parser.add_argument("--max-scanned-per-file", type=int, default=100000,
                        help="Maximum selected events inspected per file before moving on")
    args = parser.parse_args()
    if min(args.jets_per_file, args.batch_size, args.max_scanned_per_file) < 1 or (args.jets_per_label is not None and args.jets_per_label < 1):
        parser.error("Positive jets-per-file and batch-size required")
    if not args.data_config.is_file() or any(not p.is_file() for p in args.files):
        parser.error("Config and all input files must exist")
    if args.output.suffix != ".pt" or args.output.resolve() in [p.resolve() for p in [args.data_config, *args.files]]:
        parser.error("Output must be a distinct .pt file")
    result = {k: [] for k in ("features", "vectors", "mask", "labels", "source_file_index", "source_selected_index")}
    counts, scanned_counts, label_counts = [], [], Counter()
    wanted = set(args.label_ids) if args.label_ids is not None else None
    for index, path in enumerate(args.files):
        dataset = SimpleIterDataset({"diagnostic": [str(path)]}, str(args.data_config),
                                    for_training=False, fetch_by_files=False, fetch_step=0.01,
                                    async_load=False, name="val_diagnostic")
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
        count, scanned = 0, 0
        for inputs, labels, observers in loader:
            batch_labels = labels[args.label].long().cpu()
            accepted = []
            available = min(len(batch_labels), args.max_scanned_per_file - scanned)
            for row in range(available):
                label = int(batch_labels[row])
                if wanted is not None and label not in wanted:
                    continue
                if args.jets_per_label is not None and label_counts[label] >= args.jets_per_label:
                    continue
                if count + len(accepted) >= args.jets_per_file:
                    break
                accepted.append(row)
                label_counts[label] += 1
            keep = torch.tensor(accepted, dtype=torch.long)
            for output, key in (("features", "pf_features"), ("vectors", "pf_vectors"), ("mask", "pf_mask")):
                result[output].append(inputs[key][keep].cpu())
            result["labels"].append(batch_labels[keep])
            result["source_file_index"].append(torch.full((len(keep),), index, dtype=torch.long))
            result["source_selected_index"].append(keep + scanned)
            for key, value in observers.items():
                value = torch.as_tensor(value)
                if value.ndim == 1 and len(value) == len(batch_labels):
                    result.setdefault("observer__" + key, []).append(value[keep].cpu())
            count += len(keep)
            scanned += available
            complete = (wanted is not None and args.jets_per_label is not None
                        and all(label_counts[label] >= args.jets_per_label for label in wanted))
            if count == args.jets_per_file or scanned >= args.max_scanned_per_file or complete:
                break
        counts.append(count)
        scanned_counts.append(scanned)
        print(f"{path}: exported {count} of {scanned} inspected selected jets", flush=True)
        if wanted is not None and args.jets_per_label is not None and all(label_counts[label] >= args.jets_per_label for label in wanted):
            break
    if not sum(counts):
        parser.error("No jets passed the configured selection")
    output = {k: torch.cat(v) for k, v in result.items()}
    output["metadata"] = dict(files=[str(p) for p in args.files], counts=counts,
                              data_config=str(args.data_config),
                              scanned_counts=scanned_counts, label_counts=dict(label_counts),
                              requested_labels=args.label_ids, jets_per_label=args.jets_per_label,
                              sampling="First qualifying selected jets per file, optional label caps; no shuffle/reweighting",
                              source_index_note="source_selected_index is the ordinal after YAML selection, not a raw file row")
    if wanted is not None and args.jets_per_label is not None:
        missing = {label: args.jets_per_label - label_counts[label] for label in sorted(wanted)
                   if label_counts[label] < args.jets_per_label}
        output["metadata"]["unfilled_label_counts"] = missing
        if missing:
            print(f"Unfilled label quotas (supply more held-out files): {missing}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(f"Saved {sum(counts)} jets to {args.output}")


if __name__ == "__main__":
    main()
