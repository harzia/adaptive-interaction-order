#!/usr/bin/env python3
"""Synthetic optimization check; not a JetClass-II physics result."""
import argparse
import json
import torch
from aio.models.jets.representation_experiments import build_experiment, experiment_family
from benchmark_representation_order import synthetic_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="regions8_rank2")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    torch.set_num_threads(2)
    torch.manual_seed(123)
    overrides = {"trim": False} if experiment_family(args.preset) == "part" else {}
    model = build_experiment(args.preset, num_classes=3, fc_params=[], **overrides).to(args.device)
    inputs = synthetic_batch(12, 24, device=args.device)
    labels = torch.arange(12, device=args.device) % 3
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    history = []
    for _ in range(args.steps):
        opt.zero_grad(set_to_none=True)
        logits = model(*inputs)
        ce = torch.nn.functional.cross_entropy(logits, labels)
        (ce + getattr(model, "last_aux_loss", 0.)).backward()
        for name, p in model.named_parameters():
            if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all()):
                raise RuntimeError(f"Missing/non-finite gradient: {name}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()
        history.append(float(ce.detach()))
    print(json.dumps(dict(preset=args.preset, steps=args.steps, first_ce=history[0],
                          last_ce=history[-1], diagnostics={k: float(v) for k, v in getattr(model, "last_diagnostics", {}).items()},
                          note="Synthetic fixed-batch optimization only"), indent=2))


if __name__ == "__main__":
    main()
