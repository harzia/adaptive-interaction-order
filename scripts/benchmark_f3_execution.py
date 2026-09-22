#!/usr/bin/env python3
"""Warmed, unprofiled synthetic F3 training benchmark (no Weaver/data dependency)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aio.models.jets.particle_transformer_aio import AIOParticleTransformer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--feature-dim", type=int, default=7, help="JetClassII_kin has 7 features")
    parser.add_argument("--anchor-chunk", type=int, default=4096)
    parser.add_argument("--jet-group-size", type=int, default=16)
    parser.add_argument("--message-backend", choices=("auto", "dense", "sparse"), default="auto")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    if args.batch_size < 1 or args.seq_len < 3 or args.steps < 1 or args.warmup < 10:
        parser.error("batch-size/steps must be positive, seq-len >= 3, and warmup >= 10")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu --batch-size 2 --seq-len 8 for a smoke test")
    if device.type == "cpu":
        torch.set_num_threads(1)
    torch.manual_seed(args.seed)

    B, N = args.batch_size, args.seq_len
    lengths = torch.linspace(max(3, N // 2), N, B, device=device).long()
    mask = (torch.arange(N, device=device)[None] < lengths[:, None])[:, None]
    features = torch.randn(B, args.feature_dim, N, device=device).masked_fill(~mask, 0)
    momentum = torch.randn(B, 3, N, device=device) * 20
    energy = (momentum.square().sum(1, keepdim=True) + 1).sqrt()
    vectors = torch.cat([momentum, energy], dim=1).masked_fill(~mask, 0)
    labels = torch.randint(188, (B,), device=device)

    # Match the dense pilot's model widths/stage, but hold shapes fixed for fair execution comparisons.
    model = AIOParticleTransformer(
        input_dim=args.feature_dim, num_classes=188, pair_input_dim=4,
        use_pre_activation_pair=True, embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64], num_heads=8, num_layers=8, num_cls_layers=2,
        fc_params=[(512, 0.1)], activation="gelu", trim=False,
        f3_after=4, D=32, H3=2, d_r=32, anchor_chunk=args.anchor_chunk,
        jet_group_size=args.jet_group_size, message_backend=args.message_backend,
        ckpt=not args.no_checkpoint,
    ).to(device).train()
    model.calibrate(features, v=vectors, mask=mask)
    with torch.no_grad():
        model.hob.motif.U.weight.normal_(std=0.02)      # exercise the nonzero message/gradient paths
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.precision == "bf16"):
            output = model(features, v=vectors, mask=mask)
            loss = torch.nn.functional.cross_entropy(output.float(), labels)
        loss.backward()
        optimizer.step()
        return loss.detach()

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(args.warmup):
        step()
    optimizer.zero_grad(set_to_none=True)
    sync()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    for _ in range(args.steps):
        loss = step()
    sync()
    elapsed = time.perf_counter() - start
    if not torch.isfinite(loss).item():
        raise RuntimeError("Nonfinite loss in benchmark")
    report = {
        "config": vars(args), "torch": torch.__version__,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "optimizer": "AdamW", "trim": False, "synthetic_inputs": True,
        "seconds_per_step": elapsed / args.steps,
        "entries_per_second": B * args.steps / elapsed,
        "peak_allocated_MiB": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
        "peak_reserved_MiB": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else None,
        "last_loss": loss.item(),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
