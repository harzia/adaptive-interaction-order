#!/usr/bin/env python3
"""Export class-token embeddings, logits and constituents from a trained tagger.

Writes one shard per input file with Weaver's own preprocessing and no
training reweighting. Each shard holds, per jet: the truth label, the frozen
class-token embedding, the original logits, the scalar observers, and the valid
constituent four-momenta exactly as the model received them. Files whose shard
already exists are skipped, so an interrupted export can be resumed.

Use held-out files for every split: files the checkpoint never trained on for
``train`` and ``val``, and test files for ``test``.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from weaver.utils.dataset import SimpleIterDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from integrations.weaver.RelativeParT import get_model  # noqa: E402

# The Sophon-aligned ParT that integrations/weaver/ParT.py trained on main.
FULL_SIZE_PART = dict(embed_dims=[128, 512, 128], num_heads=8, num_layers=8)


def parse_option(text):
    key, separator, value = text.partition("=")
    if not key or not separator:
        raise argparse.ArgumentTypeError(f"Expected KEY=VALUE, got {text!r}")
    try:
        return key, ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return key, value


def load_state(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = {}
    for key, value in payload.get("state_dict", payload).items():
        while key.startswith(("module.", "_orig_mod.")):
            key = key.split(".", 1)[1]
        state[key] = value
    if not any(key.startswith("mod.") for key in state):   # saved from the bare model
        state = {"mod." + key: value for key, value in state.items()}
    return state


def open_file(path, group, args):
    return SimpleIterDataset({group: [str(path)]}, str(args.data_config), for_training=False,
                             fetch_by_files=True, fetch_step=1, async_load=False, name=f"probe_{group}")


def split_group(text):
    group, sep, path = text.partition(":")
    return (group, Path(path)) if sep else (Path(text).stem.split("_")[0], Path(text))


@torch.no_grad()
def export_file(path, group, dataset, model, device, args, rng):
    names = list(dataset.config.input_names)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    out = {k: [] for k in ("label", "embedding", "logits", "part_vectors", "counts")}
    observers_out, seen, kept = {}, 0, 0
    for inputs, labels, observers in loader:
        size = len(labels[args.label])
        seen += size
        keep = torch.from_numpy(rng.random(size) < args.keep_fraction)
        if args.max_jets_per_file is not None:
            allowed = max(args.max_jets_per_file - kept, 0)
            keep &= keep.cumsum(0) <= allowed
        if not keep.any():
            if args.max_jets_per_file is not None and kept >= args.max_jets_per_file:
                break
            continue
        batch = [inputs[k][keep].to(device, non_blocking=True) for k in names]
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp):
            output = model(*batch)
        output = output.float().cpu()
        if output.shape[1] <= args.num_classes:
            raise RuntimeError("The model returned no embedding; export_embed must be enabled")
        mask = inputs["pf_mask"][keep][:, 0] > 0
        out["label"].append(labels[args.label][keep].numpy().astype(np.int16))
        out["logits"].append(output[:, :args.num_classes].numpy())
        out["embedding"].append(output[:, args.num_classes:].numpy())
        out["part_vectors"].append(inputs["pf_vectors"][keep].transpose(1, 2)[mask].numpy().astype(np.float32))
        out["counts"].append(mask.sum(1).numpy())
        for key, value in observers.items():
            if torch.is_tensor(value) and value.ndim == 1 and len(value) == size:
                observers_out.setdefault(key, []).append(value[keep].numpy().astype(np.float32))
        kept += int(keep.sum())
    if not kept:
        return None
    arrays = {k: np.concatenate(v) for k, v in out.items()}
    arrays["part_offsets"] = np.concatenate(([0], np.cumsum(arrays.pop("counts")))).astype(np.int64)
    arrays.update({k: np.concatenate(v) for k, v in observers_out.items()})
    for required in ("jet_pt", "jet_sdmass"):
        if required not in arrays:
            raise RuntimeError(f"Observer {required!r} is missing; the data config must list it under observers")
    arrays["meta"] = np.array(json.dumps(dict(
        source=str(path), group=group, split=args.split, seen=seen, kept=kept,
        keep_fraction=args.keep_fraction, checkpoint=str(args.checkpoint), preset=args.preset,
        model_options=args.resolved_options, data_config=str(args.data_config),
        data_config_sha256=hashlib.sha256(args.data_config.read_bytes()).hexdigest())))
    return arrays


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--files", nargs="+", required=True,
                   help="Input files, optionally GROUP:path as in Weaver (group defaults to the filename prefix)")
    p.add_argument("--output-dir", type=Path, required=True, help="Root of the shard tree")
    p.add_argument("--split", required=True, choices=("train", "val", "test"))
    p.add_argument("--preset", default="part_pair_mlp64_d4")
    p.add_argument("--full-size-part", action="store_true",
                   help="Width 128, 8 heads, 8 blocks: the default ParT configuration")
    p.add_argument("--model-option", type=parse_option, action="append", default=[], metavar="KEY=VALUE",
                   help="Override a model argument, as Weaver's -o does")
    p.add_argument("--data-config", type=Path, default=ROOT / "configs/datasets/jetclass2/JetClassII_kin.yaml")
    p.add_argument("--num-classes", type=int, default=188)
    p.add_argument("--label", default="truth_label")
    p.add_argument("--keep-fraction", type=float, default=1.0, help="Random fraction of each file to export")
    p.add_argument("--max-jets-per-file", type=int)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true", help="bfloat16 autocast, as in training; off keeps float32")
    p.add_argument("--seed", type=int, default=20261121)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if not 0 < args.keep_fraction <= 1 or args.batch_size < 1:
        p.error("Expected 0 < keep-fraction <= 1 and a positive batch size")
    files = [split_group(f) for f in args.files]
    missing = [str(path) for _, path in files if not path.is_file()]
    if missing or not args.checkpoint.is_file() or not args.data_config.is_file():
        p.error("Missing checkpoint, data config or input files: " + ", ".join(missing[:5]))
    options = dict(FULL_SIZE_PART if args.full_size_part else {}, **dict(args.model_option))
    args.resolved_options = {k: repr(v) for k, v in options.items()}
    out_dir = args.output_dir / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    model = None
    total, correct = 0, 0
    for index, (group, path) in enumerate(files):
        target = out_dir / f"{path.stem}.npz"
        if target.is_file() and not args.overwrite:
            print(f"[{index + 1}/{len(files)}] {target.name} exists, skipping", flush=True)
            continue
        dataset = open_file(path, group, args)
        if model is None:
            model, _ = get_model(dataset.config, preset=args.preset, num_classes=args.num_classes,
                                 export_embed=True, **options)
            try:
                model.load_state_dict(load_state(args.checkpoint), strict=True)
            except RuntimeError as error:
                raise SystemExit(
                    f"The checkpoint does not fit preset {args.preset!r} with options {options}.\n"
                    "Use --full-size-part for the default ParT (width 128, 8 heads, 8 blocks), the preset name\n"
                    f"the checkpoint was trained with, and --model-option for anything passed with -o.\n\n{error}") from None
            model = model.to(device).eval()
            print(f"Loaded {args.checkpoint} into {type(model).__name__} "
                  f"({sum(p.numel() for p in model.parameters()):,} parameters)", flush=True)
        # One generator per file keeps the selection independent of which files were skipped.
        digest = hashlib.sha256(f"{args.seed}:{path.name}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        arrays = export_file(path, group, dataset, model, device, args, rng)
        if arrays is None:
            print(f"[{index + 1}/{len(files)}] {path.name}: no jets selected", flush=True)
            continue
        temporary = target.with_suffix(".tmp.npz")
        np.savez(temporary, **arrays)
        os.replace(temporary, target)
        hits = int((arrays["logits"].argmax(1) == arrays["label"]).sum())
        total, correct = total + len(arrays["label"]), correct + hits
        print(f"[{index + 1}/{len(files)}] {path.name}: {len(arrays['label'])} jets, "
              f"top-1 {hits / len(arrays['label']):.4f}, embedding width {arrays['embedding'].shape[1]}", flush=True)
    if total:
        print(f"Exported {total} jets to {out_dir}; top-1 of the original head {correct / total:.4f}")


if __name__ == "__main__":
    main()
