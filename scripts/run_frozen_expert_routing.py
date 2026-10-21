#!/usr/bin/env python3
"""Complementarity diagnostics and held-out frozen-expert routing experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch

from aio.core.routing.frozen_experts import ExpertCache, ConstantGate, validate_splits
from aio.core.routing.experiment import evaluate_gates, fit_gate, export_jetclass2_metrics, paired_difference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    diagnostic = sub.add_parser("diagnose", help="Inspect complementarity on a development/validation cache")
    diagnostic.add_argument("--cache", type=Path, required=True)
    diagnostic.add_argument("--output", type=Path, required=True)
    fitting = sub.add_parser("fit", help="Fit constant, soft, and hard gates with disjoint fit/validation/test caches")
    for name in ("fit", "validation", "test", "output"):
        fitting.add_argument("--" + name, type=Path, required=True)
    fitting.add_argument("--epochs", type=int, default=40)
    fitting.add_argument("--patience", type=int, default=8)
    fitting.add_argument("--hidden", type=int, default=16, help="0 for a linear input gate")
    fitting.add_argument("--learning-rate", type=float, default=.003)
    fitting.add_argument("--seed", type=int, default=123)
    fitting.add_argument("--jetclass2-metrics", action="store_true", help="Write ROOT predictions and run the existing 188-class evaluator for every method")
    for command in (diagnostic, fitting):
        command.add_argument("--batch-size", type=int, default=1024)
        command.add_argument("--device", default="cpu")
        command.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if min(args.batch_size, args.threads) < 1:
        parser.error("Positive batch size and threads required")
    torch.set_num_threads(args.threads)
    if args.output.exists():
        parser.error("Output already exists; choose a fresh path")
    if args.command == "diagnose":
        cache = ExpertCache(args.cache)
        dummy = {name: ConstantGate(len(cache.metadata["experts"])).to(args.device).eval()
                 for name in ("constant", "soft", "hard")}
        report, _, _, _ = evaluate_gates(cache, dummy, batch_size=args.batch_size, device=args.device)
        for key in ("hard_route_fraction", "mean_soft_weights", "constant_weights", "comparisons"):
            report.pop(key)
        for method in ("constant", "soft", "hard"):
            report["methods"].pop(method)
        for group in report["groups"].values():
            group.pop("hard_route_fraction")
            for method in ("constant", "soft", "hard"):
                group["ce"].pop(method)
        report["cache"] = str(args.cache.resolve())
        report["expert_names"] = [e["name"] for e in cache.metadata["experts"]]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: report[k] for k in ("methods", "diagnostic_only_oracle")}, indent=2))
        return
    if min(args.epochs, args.patience) < 1 or args.hidden < 0 or args.learning_rate <= 0:
        parser.error("Invalid fitting options")
    fit, validation, test = (ExpertCache(getattr(args, name)) for name in ("fit", "validation", "test"))
    validate_splits(fit, validation, test)
    if args.jetclass2_metrics:
        from integrations.weaver.eval.jetclass2_eval import CLASS_NAMES
        if test.metadata["class_names"] != CLASS_NAMES:
            parser.error("--jetclass2-metrics requires canonical JetClass-II class_names")
    args.output.mkdir(parents=True, exist_ok=False)
    gates, fits = {}, {}
    for kind in ("constant", "soft", "hard"):
        gates[kind], fits[kind] = fit_gate(fit, validation, kind=kind, hidden=args.hidden,
            epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
            learning_rate=args.learning_rate, seed=args.seed, device=args.device)
        print(f"{kind}: selected epoch {fits[kind]['best_epoch']}, validation CE {fits[kind]['validation_ce']:.6f}", flush=True)
    val_report, _, _, _ = evaluate_gates(validation, gates, batch_size=args.batch_size, device=args.device)
    expert_names = ["expert:" + e["name"] for e in test.metadata["experts"]]
    best_single = min(expert_names, key=lambda name: val_report["methods"][name]["ce"])
    # Test outcomes are computed only after all fits and selections have finished.
    report, losses, choices, names = evaluate_gates(test, gates, batch_size=args.batch_size, device=args.device)
    report.update(best_single_selected_on_validation=best_single, validation=val_report,
                  fitting=fits, expert_names=[e["name"] for e in test.metadata["experts"]],
                  settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  cache_manifests={key: cache.metadata for key, cache in (("fit", fit), ("validation", validation), ("test", test))},
                  interpretation="Per-jet frozen-expert feasibility test. Soft uses all experts; hard chooses one. No regional order-routing, latency, or seed-stability claim.")
    for method in ("soft", "hard"):
        report["comparisons"][method + " minus validation_best_single"] = paired_difference(losses[:, names.index(method)], losses[:, names.index(best_single)])
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    np.savez_compressed(args.output / "per_event.npz", losses=losses, method_names=np.asarray(names),
                        hard_choices=choices, event_ids=test.event_ids, labels=test.labels)
    checkpoint = dict(format="aio-frozen-router-v1", hidden=args.hidden, metadata=fit.metadata,
                      seed=args.seed, fits=fits,
                      states={name: {k: v.cpu() for k, v in gate.state_dict().items()} for name, gate in gates.items()})
    torch.save(checkpoint, args.output / "router.pt")
    if args.jetclass2_metrics:
        export_jetclass2_metrics(test, gates, args.output, batch_size=args.batch_size, device=args.device, seed=args.seed)
    print(json.dumps(report["methods"], indent=2))
    print(f"Saved report and deployable hard gate to {args.output}")


if __name__ == "__main__":
    main()
