#!/usr/bin/env python3
"""Print the primary angular/ParT comparison. --execute runs it sequentially."""
import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets", nargs="+", default=["part_d2", "part_d4", "moments_l1_d2", "moments_l2_d2"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[123, 124])
    parser.add_argument("--comment-prefix", default="angular")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("weaver_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.comment_prefix) or any(s < 0 for s in args.seeds):
        parser.error("Invalid comment prefix or seed")
    if len(set(args.presets)) != len(args.presets) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Duplicate presets/seeds would repeat a run")
    for preset in args.presets:
        if not re.fullmatch(r"[a-zA-Z0-9_]+", preset) or preset == "common" or not (root / "configs/experiments/angular_moments" / f"{preset}.json").is_file():
            parser.error(f"Unknown preset {preset}")
    extra = args.weaver_args[1:] if args.weaver_args[:1] == ["--"] else args.weaver_args
    if any(value == "--seed" or value.startswith("--seed=") for value in extra):
        parser.error("Use --seeds instead of passing --seed to Weaver")
    print(f"{'Executing' if args.execute else 'Plan only:'} {len(args.presets) * len(args.seeds)} runs", file=sys.stderr)
    for preset in args.presets:
        for seed in args.seeds:
            comment = f"{args.comment_prefix}_{preset}_seed{seed}"
            command = ["bash", str(root / "scripts/run_angular_experiment.sh"), preset, "--seed", str(seed), *extra]
            print("COMMENT=" + shlex.quote(comment) + " " + shlex.join(command), flush=True)
            if args.execute:
                subprocess.run(command, cwd=root, env={**os.environ, "COMMENT": comment}, check=True)


if __name__ == "__main__":
    main()
