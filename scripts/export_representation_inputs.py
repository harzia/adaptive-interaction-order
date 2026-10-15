#!/usr/bin/env python3
"""Capture a small fixed diagnostic sample using Weaver's actual preprocessing.

Reads the first selected jets from each explicit input file, without training
reweighting or shuffling. This is an inspection sample, not an unbiased estimate
of full validation performance. Select held-out files and reuse the output across
checkpoints. Supply the training run's resolved YAML when preprocessing is auto.
"""
from __future__ import annotations

import argparse
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
    args = parser.parse_args()
    if min(args.jets_per_file, args.batch_size) < 1:
        parser.error("Positive jets-per-file and batch-size required")
    if not args.data_config.is_file() or any(not p.is_file() for p in args.files):
        parser.error("Config and all input files must exist")
    if args.output.suffix != ".pt" or args.output.resolve() in [p.resolve() for p in [args.data_config, *args.files]]:
        parser.error("Output must be a distinct .pt file")
    result = {k: [] for k in ("features", "vectors", "mask", "labels", "source_file_index")}
    counts = []
    for index, path in enumerate(args.files):
        dataset = SimpleIterDataset({"diagnostic": [str(path)]}, str(args.data_config),
                                    for_training=False, fetch_by_files=False, fetch_step=0.01,
                                    async_load=False, name="val_diagnostic")
        loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
        count = 0
        for inputs, labels, _ in loader:
            take = min(args.jets_per_file - count, len(inputs["pf_features"]))
            for output, key in (("features", "pf_features"), ("vectors", "pf_vectors"), ("mask", "pf_mask")):
                result[output].append(inputs[key][:take].cpu())
            result["labels"].append(labels[args.label][:take].long().cpu())
            result["source_file_index"].append(torch.full((take,), index, dtype=torch.long))
            count += take
            if count == args.jets_per_file:
                break
        counts.append(count)
        print(f"{path}: exported {count} selected jets", flush=True)
    if not sum(counts):
        parser.error("No jets passed the configured selection")
    output = {k: torch.cat(v) for k, v in result.items()}
    output["metadata"] = dict(files=[str(p) for p in args.files], counts=counts,
                              data_config=str(args.data_config),
                              sampling="First selected jets per file, no shuffle/reweighting")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(f"Saved {sum(counts)} jets to {args.output}")


if __name__ == "__main__":
    main()
