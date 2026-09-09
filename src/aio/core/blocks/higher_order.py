"""Higher-order block (spec §6 interface, §2.6 router) and injection into the host (spec §2.5).

HigherOrderBlock: factors + motif + router; F(h, e_pair, mask; spec) -> (delta_a, anc, aux).
Injection: b3(Δr) scattered from the anchor list into the [B,H,N,N] logits; the message aggregated
per head over the anchor list before M_h is applied (linearity), so no [B,N,N,d] tensor exists.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import torch, torch.nn as nn

from ..interactions.factors import Factors
from ..interactions.motifs import Motif, TRIANGLE
from ..routing.anchors import anchors, topk_indices, gather


class HigherOrderBlock(nn.Module):
    """§6 interface: F(h, e_pair, mask; spec) -> (delta_a, anc, aux).  Includes the router (§2.6).

    use_router=False (Phase 1, dense): router parameters are frozen so DDP sees no unused trainable
    parameters.  Specs with no attended pool and no router never compute r̄, and the anchor-state
    parameters are frozen for the same reason.  Alternatively wrap DDP with find_unused_parameters=True."""

    def __init__(self, d, d_e, spec=TRIANGLE, D=32, H=2, d_r=32, d_b=16, d_v=16, R=4, d_mix=None,
                 anchor_chunk=256, ckpt=True, dense_matmul=False, use_router=False, diagnostics=False):
        super().__init__()
        self.spec = tuple(spec)
        self.factors = Factors(d, d_e, D, d_r, d_b, d_v, R, sorted({b for p in self.spec for b in p.bonds}))
        self.motif = Motif(self.spec, d_r, D, H, d_r, d_mix, anchor_chunk, ckpt, dense_matmul, diagnostics)
        self.router = nn.Sequential(nn.LayerNorm(d_r), nn.Linear(d_r, 1))
        self.use_router = use_router
        self.needs_rbar = use_router or any(p.attended for p in self.spec)
        for prm in self.router.parameters(): prm.requires_grad_(use_router)
        for mod in (self.factors.r, self.factors.W, self.factors.V):
            for prm in mod.parameters(): prm.requires_grad_(self.needs_rbar)
        self.register_buffer("calibrated", torch.tensor(False))

    def forward(self, h, e_pair, mask, rho=None, anc=None, gate=None, cand_mask=None):
        e_hat, g, n, cfac = self.factors(h, e_pair)                          # dense: ê [B,N,N,d_b], g [B,N,N,D]
        anc_all = anchors(mask) if anc is None else anc
        rbar = self.factors.rbar_at(e_hat, h, anc_all) if self.needs_rbar else None   # [A,d_r], never dense
        if anc is None and rho is not None and rho < 1:
            assert self.use_router, "routing requested but the router is frozen (use_router=False)"
            sel = topk_indices(self.router(rbar).squeeze(-1).detach().float(), anc_all, mask.shape[0], rho)
            anc_all, rbar = anc_all[sel], rbar[sel]                          # W_q runs on the selection only
        delta_a, anc_out, aux = self.motif(rbar, g, n, mask, cfac, anc_all, gate, cand_mask)
        aux["rbar"] = rbar
        return delta_a, anc_out, aux
    
    @torch.no_grad()
    def calibrate(self, h, e_pair, mask, target=1.0, force=False):
        """
        One-time real-batch initialization calibration.
        1. Rescale MLP_g output channels to unit std on valid off-diagonal legs.
        2. Rescale MLP_n output channels to unit std on valid nodes.
        3. Recompute the F3 logits.
        4. Rescale each W_q head so mean within-anchor logit std ~= target.
        `calibrated` is persistent in the state dict, so resumed checkpoints
        automatically skip this.
        """
        if bool(self.calibrated.item()) and not force:
            return {"skipped": True}
        old_diagnostics = self.motif.diagnostics
        self.motif.diagnostics = True
        try:
            _, g, n, _ = self.factors(h, e_pair)
            B, N = mask.shape
            eye = torch.eye(N, dtype=torch.bool, device=mask.device)
            leg_mask = mask[:, :, None] & mask[:, None, :] & ~eye[None]
            if not bool(leg_mask.any().item()):
                raise RuntimeError("F3 calibration batch contains no valid pair legs.")
            if not bool(mask.any().item()):
                raise RuntimeError("F3 calibration batch contains no valid nodes.")
            g_std = g[leg_mask].float().std(dim=0, unbiased=False).clamp_min(1e-6)
            n_std = n[mask].float().std(dim=0, unbiased=False).clamp_min(1e-6)

            report = {"g_std_before": g_std.mean(), "n_std_before": n_std.mean(),}
            for linear, std in (
                (self.factors.g[-1], g_std),
                (self.factors.n[-1], n_std),
            ):
                linear.weight.div_(std[:, None].to(linear.weight.dtype))
                if linear.bias is not None:
                    linear.bias.div_(std.to(linear.bias.dtype))

            _, _, aux = self.forward(h, e_pair, mask)
            H = self.motif.H
            for pool_id, logit_std in aux["stats"]["logit_std"].items():
                per_head = logit_std.float().mean(dim=0).clamp_min(1e-6)
                Wq = self.motif.Wq[str(pool_id)]
                Wq_view = Wq.weight.view(H, -1, Wq.weight.shape[1])
                scale = (target / per_head).to(Wq.weight.dtype)
                Wq_view.mul_(scale[:, None, None])
                report[f"logit_std_before_pool{pool_id}_per_head"] = per_head

            _, g_after, n_after, _ = self.factors(h, e_pair)
            report["g_std_after"] = (
                g_after[leg_mask].float().std(dim=0, unbiased=False).mean()
            )
            report["n_std_after"] = (
                n_after[mask].float().std(dim=0, unbiased=False).mean()
            )
            _, _, aux = self.forward(h, e_pair, mask)
            for pool_id in aux["stats"]["logit_std"]:
                report[f"logit_std_after_pool{pool_id}_per_head"] = (
                    aux["stats"]["logit_std"][pool_id].float().mean(dim=0)
                )
                report[f"entropy_after_pool{pool_id}_per_head"] = (
                    aux["stats"]["entropy"][pool_id].float().mean(dim=0)
                )
            report["term_std_after"] = aux["stats"]["term_std"].float()
            self.calibrated.fill_(True)
            return report

        finally:
            self.motif.diagnostics = old_diagnostics

class Injection(nn.Module):
    def __init__(self, d_r, n_heads, d_model):
        super().__init__()
        self.H, self.dh = n_heads, d_model // n_heads
        self.b3, self.M = nn.Linear(d_r, n_heads, bias=False), nn.Linear(d_r, d_model, bias=False)

    def bias(self, delta_a, anc, B, N):                       # [A,d_r] -> [B,H,N,N] additive logits
        b, i, j = anc.unbind(1); v = self.b3(delta_a)
        out = v.new_zeros(B * N * N, self.H)
        out = out.index_add(0, (b * N + i) * N + j, v).index_add(0, (b * N + j) * N + i, v)
        return out.view(B, N, N, self.H).permute(0, 3, 1, 2)

    def message(self, attn, delta_a, anc, chunk=16384):       # attn [B,H,N,N], [A,d_r] -> [B,H,N,dh]
        B, H, N, _ = attn.shape
        d = delta_a.to(attn.dtype); agg = attn.new_zeros(B * N, H, d.shape[1])
        for s0 in range(0, anc.shape[0], chunk):                            # bounds the [Ac,H,d_r] temporary
            b, i, j = anc[s0:s0 + chunk].unbind(1); dc = d[s0:s0 + chunk, None]
            agg = agg.index_add(0, b * N + i, attn[b, :, i, j][..., None] * dc) \
                     .index_add(0, b * N + j, attn[b, :, j, i][..., None] * dc)      # [B*N,H,d_r]
        Wm = self.M.weight.view(H, self.dh, -1).to(attn.dtype)
        return torch.einsum("nhr,hdr->nhd", agg, Wm).view(B, N, H, self.dh).permute(0, 2, 1, 3)


def attend_with_message(q, k, v, attn_mask, msg_fn=None, dropout_p=0.0, training=False):
    """q,k,v [B,H,N,dh]; attn_mask additive [B,H,N,N] or None; msg_fn(attn) -> [B,H,N,dh] or None."""
    s = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if attn_mask is not None: s = s + attn_mask
    sdt = torch.float32 if s.dtype in (torch.float16, torch.bfloat16) else s.dtype
    a = torch.nan_to_num(torch.softmax(s.to(sdt), -1)).to(v.dtype)              # fully-masked rows -> 0
    if dropout_p > 0 and training: a = nn.functional.dropout(a, dropout_p)
    out = a @ v
    return out + msg_fn(a) if msg_fn is not None else out