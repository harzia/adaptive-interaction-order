"""Factor maps the operator reads (spec §2.2): ê = W_b e; g = MLP_g(ê); n = MLP_n(h);
r̄ = MLP_r(ê, W(h_i+h_j), |Vh_i−Vh_j|); c^b = MLP_c(h).  All O(n^2) or cheaper; all nonlinear.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import torch, torch.nn as nn


def mlp(d_in, d_h, d_out):
    return nn.Sequential(nn.Linear(d_in, d_h), nn.GELU(), nn.Linear(d_h, d_out))


class Factors(nn.Module):
    """ê = W_b e;  g = MLP_g(ê);  n = MLP_n(h);  r̄ = MLP_r(ê, W(h_i+h_j), |Vh_i−Vh_j|);  c^b = MLP_c(h)."""

    def __init__(self, d, d_e, D=32, d_r=32, d_b=16, d_v=16, R=4, bond_ids=()):
        super().__init__()
        self.Wb = nn.Linear(d_e, d_b, bias=False)
        self.g, self.n = mlp(d_b, D, D), mlp(d, D, D)
        self.W, self.V = nn.Linear(d, d_v, bias=False), nn.Linear(d, d_v, bias=False)
        self.r = mlp(d_b + 2 * d_v, d_r, d_r)
        self.c = nn.ModuleDict({str(b): mlp(d, R, R) for b in bond_ids})

    def forward(self, h, e, need_rbar=True):       # h [B,N,d]; e [B,N,N,d_e] symmetric
        e_hat = self.Wb(0.5 * (e + e.transpose(1, 2)))
        rbar = None
        if need_rbar:
            wh, vh = self.W(h), self.V(h)
            rbar = self.r(torch.cat([e_hat, wh[:, :, None] + wh[:, None], (vh[:, :, None] - vh[:, None]).abs()], -1))
        return rbar, self.g(e_hat), self.n(h), {int(b): m(h) for b, m in self.c.items()}

