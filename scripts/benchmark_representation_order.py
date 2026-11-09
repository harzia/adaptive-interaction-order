#!/usr/bin/env python3
"""Profile the actual model, including geometry, pooling, transport and readout.

CPU results check execution only. Use CUDA on the target hardware for cost matching.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
import torch
from aio.models.jets.representation_experiments import experiment_config, experiment_family, build_experiment


def synthetic_batch(batch_size, particles, input_dim=7, device="cpu"):
    """Jet-like synthetic momenta, solely for functional/timing checks, not physics."""
    spatial = torch.randn(batch_size, particles, 3, device=device)
    spatial[..., 0] += 3
    spatial = spatial * torch.rand(batch_size, particles, 1, device=device).add(0.1) * 10
    energy = spatial.square().sum(-1).add(0.1).sqrt()
    vectors = torch.cat((spatial, energy[..., None]), -1).transpose(1, 2)
    features = torch.randn(batch_size, input_dim, particles, device=device)
    mask = torch.ones(batch_size, 1, particles, device=device, dtype=torch.bool)
    return features, vectors, mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presets", nargs="+")
    parser.add_argument("--angular", action="store_true", help="Use the matched ParT/angular experiment presets")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--particles", type=int, default=128)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--amp", action="store_true", help="bf16 autocast; geometric operations remain fp32/fp64")
    parser.add_argument("--backward", action="store_true", help="Time forward+backward, excluding optimizer")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.presets is None:
        args.presets = (["part_d2", "part_d4", "part_d8", "moments_l1_d2", "moments_l2_d2"] if args.angular else
                        ["full_sv", "regions16_sv", "regions8_sv", "regions16_rank2", "regions8_rank2", "regions8_wide"])
    if min(args.steps, args.batch_size, args.particles, args.threads) < 1 or args.warmup < 0:
        parser.error("Positive steps, sizes, threads and nonnegative warmup required")
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(123)
    inputs = synthetic_batch(args.batch_size, args.particles, device=device)
    labels = torch.arange(args.batch_size, device=device) % 188
    rows = []
    for preset in args.presets:
        torch.manual_seed(123)
        if args.angular:
            from aio.models.jets.angular_experiments import experiment_config as angular_config, build_experiment as angular_build
            family, config = angular_config(preset)
            builder = angular_build
        else:
            config, family, builder = experiment_config(preset), experiment_family(preset), build_experiment
        # Fixed N for comparable shape timing: training-time random ParT trimming
        # is disabled here only. Real training keeps the original preset behavior.
        model = builder(preset, **({"trim": False} if family in ("part", "angular_part") else {})).to(device)
        model.train(args.backward)

        def step():
            with torch.set_grad_enabled(args.backward):
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.amp):
                    logits = model(*inputs)
                    if args.backward:
                        loss = torch.nn.functional.cross_entropy(logits, labels) + getattr(model, "last_aux_loss", 0.)
                if args.backward:
                    loss.backward()
                    model.zero_grad(set_to_none=True)
            return logits

        for _ in range(args.warmup):
            step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        for _ in range(args.steps):
            logits = step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        seconds = (time.perf_counter() - start) / args.steps
        if not torch.isfinite(logits).all():
            raise RuntimeError(f"Non-finite logits in {preset}")
        row = dict(preset=preset, parameters=sum(p.numel() for p in model.parameters()),
                   milliseconds_per_batch=seconds * 1000, jets_per_second=args.batch_size / seconds,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                   family=family,
                   region_source=config.get("region_source", "learned") if family == "region" else None,
                   includes_cpu_ca_clustering=config.get("region_source") == "ca",
                   message_components_per_head=(config["scalars"] + 4 * config["vectors"] + 16 * config["tensors"]
                                                if family == "region" else config["embed_dims"][-1] // config["num_heads"]),
                   diagnostics={k: float(v) for k, v in getattr(model, "last_diagnostics", {}).items()})
        if args.angular:
            row.update(particle_blocks=config["num_layers"], class_blocks=config["num_cls_layers"],
                       scalar_width=config["embed_dims"][-1],
                       moment_order=config.get("moment_order", 0), moment_channels=config.get("moment_channels", 0),
                       product_degree=config.get("moment_product_degree", 0),
                       persistent=config.get("moment_persistent", False),
                       added_value_components_per_head=(config.get("moment_channels", 0) *
                           (9 if config.get("moment_order") == 2 else 4)))
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model
    report = dict(torch=torch.__version__, python=platform.python_version(), device=str(device),
                  hardware=torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
                  batch_size=args.batch_size, particles=args.particles, amp=args.amp,
                  backward=args.backward, steps=args.steps, results=rows,
                  note="Synthetic inputs; not task performance. Capacity controls require measured cost matching. Angular values may change SDPA kernel selection; no FLOP or speedup claim is inferred.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
