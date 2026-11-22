#!/usr/bin/env python3
"""Run one JetClass-II arm, or a sequence of matched arms and seeds."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aio.models.jets.relative_experiments import PILOT, PRESETS, canonical_preset


def main():
    p = argparse.ArgumentParser(description=__doc__, epilog="Append Weaver options such as -o num_layers 2 or --weights FILE.")
    p.add_argument("preset", type=canonical_preset, choices=[*PRESETS, "suite"],
                   help="suite runs " + ", ".join(PILOT) + " in order")
    p.add_argument("--seed", type=int, action="append", help="repeat for multiple seeds")
    p.add_argument("--num-epochs", type=int, default=os.getenv("EPOCHS", "80"))
    p.add_argument("--train-percentage", type=int, default=os.getenv("TRAIN_PERCENTAGE", "10"))
    p.add_argument("--comment-prefix", default="relative")
    p.add_argument("--dry-run", action="store_true", default=os.getenv("AIO_DRY_RUN") == "1")
    args, extra = p.parse_known_args()
    if extra[:1] == ["--"]:
        extra = extra[1:]
    seeds = args.seed or [int(os.getenv("SEED", "123"))]
    if min(seeds) < 0 or len(set(seeds)) != len(seeds) or args.num_epochs < 1 or not 1 <= args.train_percentage <= 100:
        p.error("Expected distinct nonnegative seeds, positive epochs, and a training percentage from 1 to 100")
    presets = PILOT if args.preset == "suite" else [args.preset]
    single = len(seeds) * len(presets) == 1
    if not single and os.getenv("COMMENT"):
        p.error("Unset COMMENT for multiple runs and use --comment-prefix to give each run its own directory")
    if not single and "--weights" in extra:
        p.error("Use one matching preset and seed with an explicit checkpoint")
    for preset in presets:
        for seed in seeds:
            comment = os.getenv("COMMENT") or f"{args.comment_prefix}_{preset}_seed{seed}"
            cmd = ["bash", "scripts/run_jetclass2.sh", "RelativeParT", "kin",
                   "--train-percentage", str(args.train_percentage), "--seed", str(seed),
                   "-o", "preset", repr(preset), *extra]
            env = dict(os.environ, COMMENT=comment, EPOCHS=str(args.num_epochs),
                       PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(ROOT), os.getenv("PYTHONPATH", "")]))
            print(f"EPOCHS={args.num_epochs} COMMENT={shlex.quote(comment)} {shlex.join(cmd)}", flush=True)
            if not args.dry_run:
                subprocess.run(cmd, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
