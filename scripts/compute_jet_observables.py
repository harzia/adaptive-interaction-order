#!/usr/bin/env python3
"""Compute hand-made jet observables for exported shards (CPU, NumPy only).

Reads the constituent four-momenta stored by extract_embeddings.py and writes a
companion ``<shard>.obs.npz`` next to each shard, plus ``observables.json`` with
the column names. Existing companions are skipped unless --overwrite is given.
"""
from __future__ import annotations

import os

# One BLAS thread per worker; the pool provides the parallelism.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse  # noqa: E402
import json  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from aio.analysis import jet_observables  # noqa: E402
from aio.analysis.head_probe import observable_path, shard_paths  # noqa: E402


def process(job):
    shard, chunk = job
    start = time.time()
    with np.load(shard) as data:
        vectors, offsets = data["part_vectors"].astype(np.float64), data["part_offsets"]
    values, names = jet_observables.compute(vectors, offsets, chunk=chunk)
    target = observable_path(shard)
    temporary = target.with_suffix(".tmp.npz")
    np.savez(temporary, values=values)
    os.replace(temporary, target)
    return str(shard), len(values), time.time() - start, [list(n) for n in names]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directories", type=Path, nargs="+", help="Split directories written by extract_embeddings.py")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--chunk", type=int, default=128, help="Jets per padded batch")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.workers < 1 or args.chunk < 1:
        p.error("Expected positive workers and chunk")
    jobs = []
    for directory in args.directories:
        shards = shard_paths(directory)
        if not shards:
            p.error(f"No shards in {directory}")
        jobs += [(s, args.chunk) for s in shards if args.overwrite or not observable_path(s).is_file()]
    # Column names do not depend on the data, so they can be written even when nothing is pending.
    _, names = jet_observables.compute_batch(np.zeros((1, 1, 4)), np.zeros((1, 1), dtype=bool))
    for directory in args.directories:
        (directory / "observables.json").write_text(json.dumps([list(n) for n in names], indent=1))
    print(f"{len(jobs)} shards to process with {args.workers} workers; {len(names)} observables", flush=True)
    done, jets, start = 0, 0, time.time()
    with Pool(args.workers) as pool:
        for shard, count, seconds, shard_names in pool.imap_unordered(process, jobs):
            if shard_names != [list(n) for n in names]:
                raise RuntimeError(f"Observable columns changed while processing {shard}")
            done, jets = done + 1, jets + count
            print(f"[{done}/{len(jobs)}] {Path(shard).name}: {count} jets in {seconds:.0f}s "
                  f"({jets / max(time.time() - start, 1e-9):.0f} jets/s overall)", flush=True)


if __name__ == "__main__":
    main()
