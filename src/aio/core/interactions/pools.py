"""The pool primitive (spec §2.3).

    t^S_ijk = n_k ⊙ ∏_{u∈S} g_uk                      S ⊆ {i,j}
    w_ijk   = 1  |  softmax_{k∉{i,j}}(q_ij·t/√d_a)     summed | attended
    P^S_ij  = Σ_k w_ijk (∏_b c^b_k[r_b]) t^S_ijk         bonded: one extra index per bond

Three evaluation paths, same semantics: the chunked anchor path (any pool; term formed once per chunk of
anchors over all k), the node-pool path (summed |S|≤1 pools, once per node, endpoint-corrected), and the
dense batched-matmul path for summed triangle pools (all pairs, Pairmixer-style).
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import torch, torch.nn as nn

NEG = -1e9


# ----------------------------------------------------------------- specs
@dataclass(frozen=True)
class PoolSpec:
    S: str = "ij"                 # 'ij' | 'i' | 'j' | ''
    attended: bool = True
    bonds: Tuple[int, ...] = ()

    @property
    def node_path(self):          # summed single-endpoint pools: dense O(n^2), endpoint-corrected
        return not self.attended and self.S != "ij"


# ----------------------------------------------------------------- pool evaluation
def bonded(t, cs):
    """Append one index per bond factor: t[L..., D, R..] × c[L..., R] -> t[L..., D, R.., R]."""
    for c in cs:
        t = t.unsqueeze(-1) * c.view(*c.shape[:-1], *[1] * (t.dim() - c.dim() + 1), c.shape[-1])
    return t


def term(S, g, n, b, i, j, dtype=None):
    """t^S over all k for a chunk of anchors: [Ac,N,D].  The gathered rows (not the dense g, n) are cast to
    `dtype` BEFORE the product, so under bf16/fp16 autocast the triple product is formed in fp32 (spec §2.3)."""
    c = (lambda x: x) if dtype is None else (lambda x: x.to(dtype))
    nk = c(n[b])
    return {"ij": lambda: c(g[b, i]) * c(g[b, j]) * nk, "i": lambda: c(g[b, i]) * nk,
            "j": lambda: c(g[b, j]) * nk, "": lambda: nk}[S]()


def node_pool_at(p, g, n, mask, cmask, cfac, anc):
    """Summed S∈{'i','j',''} pool: once per node over candidate k, then the excluded endpoint term
    subtracted — only if that endpoint was itself a candidate (cmask)."""
    B, N, _, D = g.shape; b, i, j = anc.unbind(1); nb = len(p.bonds)
    cs = lambda kk: [cfac[bd][b, kk] for bd in p.bonds]
    was = lambda kk: cmask[b, kk].to(g.dtype).view(-1, *[1] * (1 + nb))
    if p.S == "":
        m = (bonded(n, [cfac[bd] for bd in p.bonds]) * cmask.view(B, N, *[1] * (1 + nb))).sum(1)[b]
        return m - bonded(n[b, i], cs(i)) * was(i) - bonded(n[b, j], cs(j)) * was(j)
    km = (cmask[:, None] & ~torch.eye(N, dtype=torch.bool, device=g.device)[None]).view(B, N, N, *[1] * (1 + nb))
    m = (bonded(g * n[:, None], [cfac[bd][:, None].expand(B, N, N, -1) for bd in p.bonds]) * km).sum(2)   # [B,u,D,R..]
    u, k = (i, j) if p.S == "i" else (j, i)
    return m[b, u] - bonded(g[b, u, k] * n[b, k], cs(k)) * was(k)


def summed_triangle_dense(g, n, mask, cmask=None, bonds=(), cfac=None):
    """All-pairs summed triangle P_ij = Σ_{k valid, k∉{i,j}} g_ik ⊙ n_k ⊙ g_jk (⊙ bonds) as a batched
    matmul per channel — cubic, but hardware-efficient.  The Pairmixer-style comparator (§4.1)."""
    B, N, _, D = g.shape; cfac = cfac or {}; cmask = mask if cmask is None else cmask
    gm = g * cmask[:, None, :, None].to(g.dtype)                                       # non-candidate k -> 0
    Gn = bonded(gm * n[:, None], [cfac[b][:, None].expand(B, N, N, -1) for b in bonds])  # [B,i,k,D,R..]
    L = "efgh"[:len(bonds)]
    P = torch.einsum(f"bikd{L},bjkd->bijd{L}", Gn, gm)
    ar = torch.arange(N, device=g.device)
    e = bonded(g[:, ar, ar] * n * cmask[..., None].to(g.dtype), [cfac[b] for b in bonds])   # k=endpoint terms
    gv = g.view(B, N, N, D, *[1] * len(bonds))
    P = P - e[:, :, None] * gv - e[:, None, :] * gv
    pm = (mask[:, :, None] & mask[:, None, :] & ~torch.eye(N, dtype=torch.bool, device=g.device)[None])
    return P * pm.view(B, N, N, *[1] * (1 + len(bonds))).to(P.dtype)


def _logit_stats(s, a, km):
    """Per anchor and head: within-anchor logit std and normalised attention entropy -> [Ac, 2H]."""
    kmf = km[..., None].to(s.dtype); cnt = kmf.sum(1).clamp_min(1)
    mu = (s * kmf).sum(1) / cnt
    std = (((s - mu[:, None]) ** 2 * kmf).sum(1) / cnt).sqrt()
    ent = -(a * torch.log(a.clamp_min(1e-12)) * kmf).sum(1) / torch.log(cnt.clamp_min(2))
    return torch.cat([std, ent], -1).detach()