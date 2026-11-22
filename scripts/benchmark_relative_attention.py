#!/usr/bin/env python3
"""Profile the actual model, including pair features, rotations and class readout.

CPU results check execution only. Use CUDA on the target hardware for cost matching.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aio.models.jets.particle_transformer import ParticleTransformer
from aio.models.jets.relative_experiments import PILOT, PRESETS, canonical_preset, relative_config
from aio.models.jets.relative_particle_transformer import RelativeParticleTransformer


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
    parser.add_argument("--presets", nargs="+", type=canonical_preset, choices=PRESETS,
                        default=["part_pair_mlp64_d4", *PILOT])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--particles", type=int, default=128)
    parser.add_argument("--inputs", type=Path, help="Time one batch from a saved seven-feature kin diagnostic sample")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--amp", action="store_true", help="bf16 autocast; geometric operations remain fp32/fp64")
    parser.add_argument("--backward", action="store_true", help="Time forward+backward, excluding optimizer")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.particles, args.threads) < 1 or args.warmup < 0:
        parser.error("Positive steps, sizes, threads and nonnegative warmup required")
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(123)
    if args.inputs:
        data = torch.load(args.inputs, map_location="cpu", weights_only=True)
        inputs = tuple(data[k][:args.batch_size].to(device) for k in ("features", "vectors", "mask"))
        if inputs[0].shape[1] != 7 or len(inputs[0]) == 0:
            parser.error("--inputs expects a nonempty JetClass-II kin diagnostic sample")
        args.batch_size, _, args.particles = inputs[0].shape
    else:
        inputs = synthetic_batch(args.batch_size, args.particles, device=device)
    labels = torch.arange(args.batch_size, device=device) % 188
    rows = []
    for preset in args.presets:
        torch.manual_seed(123)
        config = relative_config(preset, trim=False)
        relative = "score" in config
        cls = RelativeParticleTransformer if relative else ParticleTransformer
        model = cls(input_dim=7, num_classes=188, **config).to(device)
        # The final two kin features are the unstandardized pf_points.
        model_kwargs = {"points": inputs[0][:, -2:]} if relative else {}
        model.train(args.backward)

        def step():
            with torch.set_grad_enabled(args.backward):
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.amp):
                    logits = model(*inputs, **model_kwargs)
                    if args.backward:
                        loss = torch.nn.functional.cross_entropy(logits, labels)
                if args.backward:
                    loss.backward()
                    model.zero_grad(set_to_none=True)
            return logits.detach()

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
                   particle_blocks=len(model.blocks),
                   mean_valid_particles=inputs[2].float().sum(-1).mean().item(),
                   message_components_per_head=getattr(model.blocks[0].attn, "value_width", model.blocks[0].head_dim))
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model
    report = dict(torch=torch.__version__, python=platform.python_version(), device=str(device),
                  hardware=torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
                  batch_size=args.batch_size, particles=args.particles, amp=args.amp,
                  backward=args.backward, steps=args.steps, results=rows,
                  input_sample=str(args.inputs) if args.inputs else "synthetic",
                  note="Timing only. Capacity controls require measured cost matching. Angular values may change SDPA kernel selection; no FLOP or speedup claim is inferred.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
