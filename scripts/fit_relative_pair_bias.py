#!/usr/bin/env python3
"""Compare linear and fixed rotary fits to a trained part_d4's pair embedding."""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aio.core.geometry.relative_attention import frequency_ladder
from aio.models.jets.angular_experiments import build_experiment
from aio.models.jets.particle_transformer import pairwise_lv_fts_pp
from aio.models.jets.relative_particle_transformer import geometry_coordinates


def sample_pairs(vectors, coords, mask, jet_indices, cap, rng):
    raw, offsets = [], []
    for idx in jet_indices:
        valid = mask[idx, 0].bool()
        v = vectors[idx, :, valid]
        x = coords[idx, valid]
        n = v.size(1)
        if n == 0:
            continue
        pairs = torch.randperm(n*n, generator=rng)[:cap]
        i, j = pairs // n, pairs % n
        raw.append(pairwise_lv_fts_pp(v[None, :, i], v[None, :, j])[0].T)
        offsets.append(x[j] - x[i])
    if not raw:
        raise ValueError("Both jet splits need valid particles")
    return torch.cat(raw), torch.cat(offsets)


def regress(train, test, target, truth):
    def intercept(x):
        return torch.cat((torch.ones_like(x[:, :1]), x), dim=1).double()
    w = torch.linalg.lstsq(intercept(train), target.double(), driver="gelsd").solution
    error = (truth - intercept(test) @ w).square().mean(0)
    var = truth.double().var(0, unbiased=False)
    return [dict(rmse=e.sqrt().item(), r2=(1 - e/v).item() if v > 0 else None)
            for e, v in zip(error, var)]


@torch.no_grad()
def compare(checkpoint, inputs, cap=128, seed=123):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {}
    for k, val in payload.get("state_dict", payload).items():
        while k.startswith(("module.", "_orig_mod.", "mod.")):
            k = k.split(".", 1)[1]
        state[k] = val
    model = build_experiment("part_d4").eval()
    model.load_state_dict(state, strict=True)
    data = torch.load(inputs, map_location="cpu", weights_only=True)
    v, mask = data["vectors"].float(), data["mask"]
    features = data["features"]
    if features.shape[1] != 7:
        raise ValueError("Use an exported seven-feature JetClass-II kin sample")
    coords = geometry_coordinates(features[:, -2:].float(), v)
    rng = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(v), generator=rng)
    if len(v) < 2 or cap < 1:
        raise ValueError("Need at least two jets and a positive pair cap")
    cut = max(1, int(.75 * len(v)))
    train, dx = sample_pairs(v, coords, mask, order[:cut], cap, rng)
    test, dxt = sample_pairs(v, coords, mask, order[cut:], cap, rng)
    def bias(raw):
        return torch.cat([model.pair_embed.embed(chunk.T[None])[0].T for chunk in raw.split(8192)])
    target, truth = bias(train), bias(test)
    result = dict(train_jets=cut, heldout_jets=len(v)-cut, train_pairs=len(train), heldout_pairs=len(test),
                  linear=regress(train, test, target, truth), rotary=[])
    for h, w in enumerate(frequency_ladder(truth.shape[1])):
        def basis(x):
            phases = x @ w.T
            return torch.cat((phases.cos(), phases.sin()), -1)
        result["rotary"].extend(regress(basis(dx), basis(dxt), target[:, h:h+1], truth[:, h:h+1]))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True, help="Tensor sample from export_representation_inputs.py")
    p.add_argument("--pairs-per-jet", type=int, default=128)
    p.add_argument("--seed", type=int, default=123)
    a = p.parse_args()
    print(json.dumps(compare(a.checkpoint, a.inputs, a.pairs_per_jet, a.seed), indent=2, allow_nan=False))
