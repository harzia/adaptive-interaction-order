"""Flat anchor lists (spec §2.6).  anc[A,3] = (b,i,j), i<j, valid.  Dense = all valid pairs; routed =
the per-sample top-⌊ρE⌋ sub-list.  The motif evaluates exactly the anchors it is given.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import torch, torch.nn as nn


def anchors(mask):
    """All valid unordered pairs as a flat list [A,3] = (b,i,j), i<j."""
    N = mask.shape[1]
    tri = torch.triu(torch.ones(N, N, dtype=torch.bool, device=mask.device), 1)
    return (mask[:, :, None] & mask[:, None, :] & tri[None]).nonzero()


def topk_indices(scores, anc, B, rho):
    """Per-sample top-⌊ρE_b⌋ of a flat anchor list (§2.6): indices into `anc` (and into anything aligned
    with it, e.g. r̄).  Budgets are computed in double precision, never in the score dtype."""
    b = anc[:, 0]
    k = torch.floor(rho * torch.bincount(b, minlength=B).double()).long()
    perm = torch.argsort(scores, descending=True)
    perm = perm[torch.argsort(b[perm], stable=True)]            # grouped by sample, score-descending within
    bs = b[perm]
    counts = torch.bincount(bs, minlength=B); starts = torch.cumsum(counts, 0) - counts
    rank = torch.arange(perm.numel(), device=anc.device) - starts[bs]
    return perm[rank < k[bs]]


def topk_anchors(scores, anc, B, rho):
    """Selected sub-list of anchors (see topk_indices)."""
    return anc[topk_indices(scores, anc, B, rho)]


def gather(x, anc):                                # x [B,N,N,C] -> [A,C]
    return x[anc[:, 0], anc[:, 1], anc[:, 2]]


def scatter_sym(vals, anc, B, N):                  # [A,C] -> [B,N,N,C], both orientations
    b, i, j = anc.unbind(1)
    out = vals.new_zeros(B * N * N, vals.shape[1])
    out = out.index_add(0, (b * N + i) * N + j, vals).index_add(0, (b * N + j) * N + i, vals)
    return out.view(B, N, N, -1)