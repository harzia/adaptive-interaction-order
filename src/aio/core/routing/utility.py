"""Relation utility (spec §3.1) on flat anchor lists: U = -dL/dm per anchor; per-sample summaries."""
from __future__ import annotations
import torch
from .anchors import scatter_sym

def make_gate(anc, dtype=torch.float32):           # ones [A], leaf, requires_grad
    return torch.ones(anc.shape[0], dtype=dtype, device=anc.device, requires_grad=True)


def utility(gate):                                 # after loss.backward(): U = -dL/dm  [A]
    return -gate.grad


def utility_matrix(U, anc, B, N):                  # [A] -> symmetric [B,N,N]
    return scatter_sym(U[:, None], anc, B, N)[..., 0]


def summarise_utility(U, anc, B, fracs=(0.05, 0.1, 0.2, 0.4)):
    """Per-sample summaries (the concentration claim is within a sample, not pooled over a batch, where
    large samples dominate).  Returns per-sample tensors: share of positive utility in that sample's top-f%
    anchors, fraction of negative-utility anchors, total positive utility; plus the fraction of samples with
    no positive utility."""
    b = anc[:, 0]; out = {f"top{int(f*100)}pct_share": torch.full((B,), float("nan")) for f in fracs}
    out["frac_negative"] = torch.full((B,), float("nan")); out["total_positive"] = torch.zeros(B)
    for sb in range(B):
        u = U[b == sb]
        if u.numel() == 0: continue
        pos = u.clamp_min(0); out["frac_negative"][sb] = (u < 0).float().mean(); out["total_positive"][sb] = pos.sum()
        if pos.sum() > 0:
            cum = torch.cumsum(torch.sort(pos, descending=True).values, 0) / pos.sum()
            for f in fracs: out[f"top{int(f*100)}pct_share"][sb] = cum[max(1, round(f * u.numel())) - 1]
    out["frac_samples_no_positive"] = (out["total_positive"] <= 0).float().mean()
    return out