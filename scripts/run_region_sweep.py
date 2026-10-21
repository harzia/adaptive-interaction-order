#!/usr/bin/env python3
"""Print a matched region-source/order/granularity matrix; --execute runs it serially."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regions", type=int, nargs="+", choices=[4, 8, 16, 32], default=[4, 8, 16, 32])
    parser.add_argument("--sources", nargs="+", choices=["ca", "learned"], default=["ca", "learned"])
    parser.add_argument("--representations", nargs="+", choices=["sv", "rank2", "wide"], default=["sv", "rank2", "wide"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[123, 124])
    parser.add_argument("--pair-bias", choices=["none", "local"], default="none")
    parser.add_argument("--comment-prefix", default="aro_sweep")
    parser.add_argument("--execute", action="store_true", help="Run sequentially; default only prints commands")
    parser.add_argument("weaver_args", nargs=argparse.REMAINDER, help="After --, pass training/Weaver options unchanged")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.comment_prefix) or any(s < 0 for s in args.seeds):
        parser.error("Use a path-safe comment prefix and nonnegative seeds")
    for key in ("regions", "sources", "representations", "seeds"):
        values = getattr(args, key)
        if len(set(values)) != len(values):
            parser.error(f"Duplicate --{key} entries would repeat a run")
    extra = args.weaver_args[1:] if args.weaver_args[:1] == ["--"] else args.weaver_args
    if any(value == "--seed" or value.startswith("--seed=") for value in extra):
        parser.error("Set seeds via --seeds, not Weaver passthrough")
    root = Path(__file__).resolve().parents[1]
    commands = []
    for k in args.regions:
        for source in args.sources:
            for representation in args.representations:
                preset = f"regions{k}_{'ca_' if source == 'ca' else ''}{representation}"
                preset += "_pair_local" if args.pair_bias == "local" else ""
                if not (root / "configs/experiments/representation_order" / f"{preset}.json").is_file():
                    parser.error(f"Missing preset {preset}")
                for seed in args.seeds:
                    comment = f"{args.comment_prefix}_{preset}_seed{seed}"
                    command = ["bash", str(root / "scripts/run_representation_experiment.sh"), preset, "--seed", str(seed), *extra]
                    commands.append((comment, command))
    print(f"{'Executing' if args.execute else 'Plan only:'} {len(commands)} runs", file=sys.stderr)
    for comment, command in commands:
        print("COMMENT=" + shlex.quote(comment) + " " + shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, cwd=root, env={**os.environ, "COMMENT": comment}, check=True)


if __name__ == "__main__":
    main()
