"""Toy host for tests: L pair-biased attention blocks with the injection semantics of spec §2.5."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import torch, torch.nn as nn

from ..blocks.higher_order import HigherOrderBlock, Injection, attend_with_message
from ..interactions.motifs import TRIANGLE


class _Attn(nn.Module):
    def __init__(self, d, H, d_r):
        super().__init__()
        self.H, self.dh = H, d // H
        self.qkv, self.out, self.inj = nn.Linear(d, 3 * d), nn.Linear(d, d), Injection(d_r, H, d)

    def forward(self, h, pair_bias, mask, delta=None):
        B, N, d = h.shape
        q, k, v = self.qkv(h).view(B, N, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
        am, msg_fn = pair_bias.masked_fill(~mask[:, None, None, :], -1e9), None
        if delta is not None:
            delta_a, anc = delta
            am = am + self.inj.bias(delta_a, anc, B, N); msg_fn = lambda a: self.inj.message(a, delta_a, anc)
        return h + self.out(attend_with_message(q, k, v, am, msg_fn).transpose(1, 2).reshape(B, N, d))


class ToyHost(nn.Module):
    """L blocks; Δr from the output of block `f3_after`, injected into every later block."""

    def __init__(self, d_in, d_e_raw, d=32, H=4, L=4, f3_after=2, d_e=16, spec=TRIANGLE, **hob):
        super().__init__()
        self.embed, self.pair = nn.Linear(d_in, d), nn.Sequential(nn.Linear(d_e_raw, d_e), nn.GELU(), nn.Linear(d_e, d_e))
        self.bias, self.blocks, self.f3_after = nn.Linear(d_e, H, bias=False), nn.ModuleList(_Attn(d, H, 8) for _ in range(L)), f3_after
        self.f3 = HigherOrderBlock(d, d_e, spec=spec, D=16, H=2, d_r=8, d_b=8, d_v=8, anchor_chunk=64, **hob)
        self.head = nn.Linear(d, 1)

    def forward(self, x, e_raw, mask, use_f3=True, rho=None, anc=None, gate=None, cand_mask=None):
        h, e = self.embed(x), self.pair(e_raw)
        pb, delta, aux = self.bias(e).permute(0, 3, 1, 2), None, None
        for li, blk in enumerate(self.blocks):
            h = blk(h, pb, mask, delta)
            if use_f3 and li + 1 == self.f3_after:
                delta_a, anc, aux = self.f3(h, e, mask, rho, anc, gate, cand_mask)
                delta = (delta_a, anc) if anc.shape[0] else None
        m = mask.to(h.dtype)[..., None]
        return self.head((h * m).sum(1) / m.sum(1).clamp_min(1)).squeeze(-1), aux
