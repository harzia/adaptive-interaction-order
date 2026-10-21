#!/usr/bin/env python3
"""Evaluate frozen checkpoints on the SAME preprocessed input shards.

Writes memory-mapped probability/feature arrays. Re-evaluation deliberately
avoids guessing event alignment from old ROOT predictions without stable IDs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch

from aio.core.routing.frozen_experts import (
    FEATURE_NAMES, ExpertCache, file_sha256, input_identities, jet_gate_features,
    load_expert, resolve_experts,
)


def preprocessing_digest(data):
    metadata = data.get("metadata", {})
    digest = metadata.get("data_config_sha256")
    if digest:
        return digest
    path = Path(metadata.get("data_config", ""))
    if not path.is_file():
        raise ValueError("Missing preprocessing identity: re-export inputs with the resolved training YAML")
    return file_sha256(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", type=Path, required=True, help="JSON manifest; checkpoint paths relative to manifest")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True, help="Shards from export_representation_inputs.py")
    parser.add_argument("--output", type=Path, required=True, help="New cache directory; never overwritten")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.batch_size, args.threads) < 1:
        parser.error("Positive batch size and threads required")
    torch.set_num_threads(args.threads)
    specs = resolve_experts(args.experts)
    shards, count, preprocessing, input_dim = [], 0, None, None
    for path in args.inputs:
        data = torch.load(path, map_location="cpu", weights_only=True)
        n = len(data["features"])
        if n < 1 or any(len(data[k]) != n for k in ("vectors", "mask", "labels")):
            raise ValueError("Mismatched or empty input shard")
        if data["labels"].ndim != 1 or data["labels"].dtype not in (torch.int32, torch.int64):
            raise ValueError("labels must be a 1D integer tensor")
        digest = preprocessing_digest(data)
        if preprocessing is not None and (preprocessing != digest or input_dim != data["features"].shape[1]):
            raise ValueError("Input shards use different preprocessing/features")
        preprocessing, input_dim = digest, data["features"].shape[1]
        shards.append(dict(path=str(path.resolve()), sha256=file_sha256(path), count=n,
                           metadata=data.get("metadata", {})))
        count += n
    del data
    experts = [load_expert(e, input_dim, len(specs["class_names"]), args.device) for e in specs["experts"]]
    args.output.mkdir(parents=True, exist_ok=False)
    shapes = dict(probabilities=(count, len(experts), len(specs["class_names"])),
                  features=(count, len(FEATURE_NAMES)), labels=(count,), event_ids=(count,), content_ids=(count,))
    arrays = {key: np.lib.format.open_memmap(args.output / f"{key}.npy", mode="w+", shape=shape,
              dtype="S64" if key.endswith("_ids") else ("int64" if key == "labels" else "float32"))
              for key, shape in shapes.items()}
    offset = 0
    with torch.no_grad():
        for shard in shards:
            if file_sha256(shard["path"]) != shard["sha256"]:
                raise ValueError("Input changed during export")
            data = torch.load(shard["path"], map_location="cpu", weights_only=True)
            n = shard["count"]
            ids, hashes = input_identities(data)
            arrays["event_ids"][offset:offset+n], arrays["content_ids"][offset:offset+n] = ids, hashes
            arrays["labels"][offset:offset+n] = data["labels"].numpy()
            for start in range(0, n, args.batch_size):
                end = min(start + args.batch_size, n)
                x, v, mask = (data[k][start:end].to(args.device) for k in ("features", "vectors", "mask"))
                arrays["features"][offset+start:offset+end] = jet_gate_features(v, mask).cpu().numpy()
                for index, expert in enumerate(experts):
                    logits = expert(x.float(), v.float(), mask)
                    arrays["probabilities"][offset+start:offset+end, index] = logits.float().softmax(-1).cpu().numpy()
            offset += n
            print(f"Cached {offset}/{count} jets from {shard['path']}", flush=True)
    for array in arrays.values():
        array.flush()
    metadata = dict(format="aio-frozen-experts-v1", **specs, feature_names=list(FEATURE_NAMES),
                    input_dim=input_dim, preprocessing_sha256=preprocessing, num_events=count, inputs=shards,
                    note="All experts saw identical rows. No gate fitted. Sampling follows input exports.")
    manifest = args.output / "manifest.json"
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    try:
        ExpertCache(args.output)  # catches duplicated shards, malformed probabilities, bad labels
    except Exception:
        manifest.unlink()  # leave an unmistakably incomplete cache, never a usable bad manifest
        raise
    print(f"Saved validated cache: {args.output}")


if __name__ == "__main__":
    main()
