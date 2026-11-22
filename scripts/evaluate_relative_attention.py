#!/usr/bin/env python3
"""Run the existing JetClass-II evaluator or summarize its metrics and recovered gap."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions", nargs="+", type=Path, help="All prediction ROOT files for ONE run")
    source.add_argument("--metrics", type=Path, help="Existing results/metrics.json for the arm")
    p.add_argument("--reference", type=Path, help="Matched A metrics.json")
    p.add_argument("--plain", type=Path, help="Matched B metrics.json")
    args = p.parse_args()
    if bool(args.reference) != bool(args.plain):
        p.error("Supply both --reference and --plain for gap recovery")
    if args.predictions:
        from integrations.weaver.eval.jetclass2_eval import evaluate_prediction_files
        files = {f.stem: str(f) for f in args.predictions}
        if len(files) != len(args.predictions):
            p.error("Duplicate prediction filenames")
        metrics = evaluate_prediction_files(prediction_files=files, args=args)["metrics"]
    else:
        metrics = json.loads(args.metrics.read_text())["metrics"]
    key = "rej50_geomean_hadronic"
    report = {k: v for k, v in metrics.items() if k.startswith("rej50")}
    if args.reference:
        a, b = (json.loads(f.read_text())["metrics"][key] for f in (args.reference, args.plain))
        x = metrics[key]
        if any(v is None or not math.isfinite(v) or v <= 0 for v in (a, b, x)) or a == b:
            p.error("Gap recovery needs distinct A and B and finite, positive rej50 for every run")
        report["gap_recovered"] = (math.log(x) - math.log(b)) / (math.log(a) - math.log(b))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
