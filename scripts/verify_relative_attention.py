#!/usr/bin/env python3
"""Fit the proposal's fixed frequency ladders to synthetic relative kernels."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aio.core.geometry.relative_attention import frequency_ladder


def fit_r2(train, test, y, yt):
    design = np.column_stack((np.ones(len(train)), train))
    weights = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = yt - (weights[0] + test @ weights[1:])
    return float(1 - np.sum(residual**2) / np.sum((yt - yt.mean())**2))


def kernel_fits(samples=40000, seed=123):
    rng = np.random.default_rng(seed)
    radius = np.exp(rng.uniform(np.log(.01), np.log(1.2), 2 * samples))
    angle = rng.uniform(-np.pi, np.pi, 2 * samples)
    xy = radius[:, None] * np.stack((np.cos(angle), np.sin(angle)), -1)
    dt = rng.uniform(-7, 7, 2 * samples)
    def fit(basis, target):
        return fit_r2(basis[:samples], basis[samples:], target[:samples], target[samples:])
    results = {}
    for d in (3, 4):
        omega = frequency_ladder(1, directions=d, momentum=False)[0, :, :2].numpy()
        octaves = np.cos(xy @ omega.T).reshape(len(radius), 8, d).mean(-1)
        results[f"angular_{d}"] = fit(octaves, -np.log(radius))
        if d == 3:
            results["angular_equal"] = fit(octaves.sum(-1, keepdims=True), -np.log(radius))
    w = frequency_ladder(1, angular=False)[0, :, 2].numpy()
    momentum = np.cos(dt[:, None] * w)
    results["momentum_abs"] = fit(momentum, abs(dt))
    results["momentum_lnz"] = fit(momentum, -np.logaddexp(0, abs(dt)))
    cos = np.cos(radius)
    results["harmonics_l2"] = fit(np.stack((cos, (3*cos**2 - 1)/2), -1), -np.log(radius))
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--samples", type=int, default=40000, help="Pairs per fit and per held-out split")
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()
    if args.samples < 10:
        p.error("Use at least ten samples per split")
    print(json.dumps(kernel_fits(args.samples, args.seed), indent=2))
