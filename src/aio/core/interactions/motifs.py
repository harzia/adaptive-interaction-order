"""Motifs: spec -> contraction (spec §2.3).  A motif is a tuple of PoolSpec; order = 2 + #pools.

    F(i,j) = U( LN( ⊙_v W_v P^{S_v}_ij ) ),  contracted over shared bond indices.

Identical summed pools are computed once; pools sharing S share one base term per chunk; specs that
are not i<->j invariant are evaluated on the swapped anchor and averaged; U is zero-initialised.
"""
from __future__ import annotations
import functools, math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import torch, torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .pools import PoolSpec, NEG, bonded, term, node_pool_at, summed_triangle_dense, _logit_stats
from ..routing.anchors import anchors, gather

TRIANGLE        = (PoolSpec("ij", True),)            # Phase-1 F3 (attended)
SUMMED_TRIANGLE = (PoolSpec("ij", False),)           # dense matmul comparator (Pairmixer-style)
TAIL            = (PoolSpec("i", False),)
DOUBLE_TRIANGLE = (PoolSpec("ij", False), PoolSpec("ij", False))
TRIANGLE_TAIL   = (PoolSpec("ij", False), PoolSpec("i", False))
CHAIN           = (PoolSpec("i", False, (1,)), PoolSpec("j", False, (1,)))
K4              = (PoolSpec("ij", False, (1,)), PoolSpec("ij", False, (1,)))

order = lambda spec: 2 + len(spec)
is_symmetric = lambda spec: all(p.S in ("ij", "") for p in spec)


class Motif(nn.Module):
    def __init__(self, spec: Sequence[PoolSpec], d_r, D, H, d_out, d_mix=None, anchor_chunk=256, ckpt=True,
                 dense_matmul=False, diagnostics=False):
        super().__init__()
        assert D % H == 0 and len(spec) > 0
        self.spec, self.D, self.H, self.da = tuple(spec), D, H, D // H
        self.chunk, self.ckpt, self.dense_matmul, self.diagnostics = anchor_chunk, ckpt, dense_matmul, diagnostics
        self.uniq: List[PoolSpec] = []; self.slot: List[int] = []        # identical summed pools computed once
        for p in self.spec:
            if not p.attended and p in self.uniq: self.slot.append(self.uniq.index(p))
            else: self.uniq.append(p); self.slot.append(len(self.uniq) - 1)
        self.Wq = nn.ModuleDict({str(u): nn.Linear(d_r, D, bias=False) for u, p in enumerate(self.uniq) if p.attended})
        single = len(self.spec) == 1
        self.d_mix = D if single else (d_mix or D)
        self.mix = nn.ModuleList(nn.Identity() if single else nn.Linear(D, self.d_mix, bias=False) for _ in self.spec)
        self.norm, self.U = nn.LayerNorm(self.d_mix), nn.Linear(self.d_mix, d_out, bias=False)
        nn.init.zeros_(self.U.weight)                                      # Δr = 0 at init
        L = {b: "efghklmnop"[k] for k, b in enumerate(sorted({b for p in self.spec for b in p.bonds}))}
        self.einsum = ",".join("bad"[1:] + "".join(L[b] for b in p.bonds) for p in self.spec) + "->ad"

    def _chunk(self, S, members, q_c, anc_c, g, n, cmask, cfac):
        b, i, j = anc_c.unbind(1)
        sdt = torch.float32 if g.dtype in (torch.float16, torch.bfloat16) else g.dtype
        t = term(S, g, n, b, i, j, dtype=sdt)                               # [Ac,N,D] fp32
        ar = torch.arange(t.shape[1], device=t.device)[None]
        km = cmask[b] & (ar != i[:, None]) & (ar != j[:, None])            # valid candidates k ∉ {i,j}
        Ac, N, D = t.shape; outs, stats = [], []
        for u, q in zip(members, q_c):
            if q is None:
                w = km[..., None].to(sdt); stats.append(t.new_zeros(Ac, 0))
            else:
                s = (q.to(sdt).view(Ac, 1, self.H, self.da) * t.view(Ac, N, self.H, self.da)).sum(-1)
                s = (s / math.sqrt(self.da)).masked_fill(~km[..., None], NEG)
                a = torch.softmax(s, 1) * km[..., None].to(sdt)
                stats.append(_logit_stats(s, a, km) if self.diagnostics else t.new_zeros(Ac, 0))
                w = a[..., None].expand(Ac, N, self.H, self.da).reshape(Ac, N, D)
            tb = bonded(t, [cfac[bd][b].to(sdt) for bd in self.uniq[u].bonds])
            outs.append((w.view(*w.shape, *[1] * len(self.uniq[u].bonds)) * tb).sum(1))   # [Ac,D,R..] fp32
        if self.diagnostics:                                                # masked std via sums: no host sync
            kmf = km[..., None].to(t.dtype); cnt = (kmf.sum() * D).clamp_min(1)
            mu = (t * kmf).sum() / cnt; tstd = (((t - mu) ** 2 * kmf).sum() / cnt).sqrt().detach()
        else:
            tstd = t.new_zeros(())
        return (*outs, *stats, tstd)

    def pools(self, rbar, g, n, mask, cmask, cfac, anc):
        vals: List[Optional[torch.Tensor]] = [None] * len(self.uniq); diag: Dict[int, list] = {}; tstd = []
        groups: Dict[str, List[int]] = {}
        for u, p in enumerate(self.uniq):
            if p.node_path: vals[u] = node_pool_at(p, g, n, mask, cmask, cfac, anc)
            elif self.dense_matmul and not p.attended: vals[u] = gather(summed_triangle_dense(g, n, mask, cmask, p.bonds, cfac), anc)
            else: groups.setdefault(p.S, []).append(u)
        ck = self.ckpt and self.training and torch.is_grad_enabled()
        for S, members in groups.items():
            q = [self.Wq[str(u)](rbar) if self.uniq[u].attended else None for u in members]   # rbar: [A,d_r]
            parts = [[] for _ in members]
            for s0 in range(0, anc.shape[0], self.chunk):
                sl = slice(s0, s0 + self.chunk)
                args = (S, members, [None if x is None else x[sl] for x in q], anc[sl], g, n, cmask, cfac)
                out = checkpoint(self._chunk, *args, use_reentrant=False) if ck else self._chunk(*args)
                for k, u in enumerate(members):
                    parts[k].append(out[k])
                    if self.uniq[u].attended: diag.setdefault(u, []).append(out[len(members) + k])
                tstd.append(out[-1])
            for k, u in enumerate(members): vals[u] = torch.cat(parts[k], 0)
        D_ = {u: torch.cat(v, 0) for u, v in diag.items()}                  # [A, 2H] per attended pool (empty if off)
        stats = {"logit_std": {u: x[:, :self.H] for u, x in D_.items()}, "entropy": {u: x[:, self.H:] for u, x in D_.items()},
                 "term_std": torch.stack(tstd).mean() if tstd else g.new_zeros(())}
        return vals, stats

    def contract(self, vals):                        # mix on D, elementwise product, contract bonds -> [A,d_mix]
        mixed = [torch.movedim(self.mix[k](torch.movedim(vals[self.slot[k]], 1, -1)), -1, 1) for k in range(len(self.spec))]
        dt = functools.reduce(torch.promote_types, [m.dtype for m in mixed])
        return torch.einsum(self.einsum, *[m.to(dt) for m in mixed])

    def forward(self, rbar, g, n, mask, cfac=None, anc=None, gate=None, cand_mask=None):
        """rbar is per anchor, [A,d_r], aligned with anc (a dense [B,N,N,d_r] is accepted and gathered once).
        Returns delta_a [A,d_out] for the anchors in anc (default: all valid pairs), anc, aux."""
        cfac, cmask = cfac or {}, mask if cand_mask is None else (mask & cand_mask)
        anc = anchors(mask) if anc is None else anc
        if rbar is not None and rbar.dim() == 4: rbar = gather(rbar, anc)
        if anc.shape[0] == 0:                                                # zero-selection fast path
            return g.new_zeros(0, self.U.out_features), anc, {"pools": [], "stats": None}
        vals, stats = self.pools(rbar, g, n, mask, cmask, cfac, anc)
        z = self.contract(vals)
        if not is_symmetric(self.spec):                                      # evaluate swapped anchor, average
            z = 0.5 * (z + self.contract(self.pools(rbar, g, n, mask, cmask, cfac, anc[:, [0, 2, 1]])[0]))
        d = self.U(self.norm(z))
        if gate is not None: d = d * gate[:, None].to(d.dtype)
        return d, anc, {"pools": vals, "z": z, "stats": stats}