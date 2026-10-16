"""Learned rank-0/1/2 Q/K/V with global-frame transport and regional pooling.

Rank-2 here means a full learned 4x4 Lorentz tensor, NOT a moment correction,
an angular l=2 irrep, or a geometric-algebra bivector.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from .lorentz import gather_tokens, inverse_lorentz


class GeometricAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, scalars=8, vectors=2, tensors=0):
        super().__init__()
        if min(scalars, vectors, tensors) < 0 or num_heads < 1:
            raise ValueError("Invalid representation multiplicities")
        self.scalars, self.vectors, self.tensors = scalars, vectors, tensors
        self.num_heads = num_heads
        self.head_dim = scalars + 4 * vectors + 16 * tensors
        if self.head_dim == 0:
            raise ValueError("At least one message component is required")
        self.qkv = nn.Linear(hidden_dim, 3 * num_heads * self.head_dim)
        self.out = nn.Linear(num_heads * self.head_dim, hidden_dim)
        g = torch.tensor([1., -1., -1., -1.])
        self.register_buffer("metric", torch.cat((torch.ones(scalars), g.repeat(vectors),
                             (g[:, None] * g[None, :]).flatten().repeat(tensors))))

    def transport(self, x, matrices, valid):
        """Transform [B,N,H,D] by one matrix per token, on every geometric index."""
        parts = [x[..., :self.scalars]]
        pos = self.scalars
        if self.vectors:
            v = x[..., pos:pos + 4 * self.vectors].reshape(*x.shape[:-1], self.vectors, 4)
            v = torch.einsum("ztij,zthcj->zthci", matrices, v)
            parts.append((v * valid[..., None, None, None]).flatten(-2))
            pos += 4 * self.vectors
        if self.tensors:
            t = x[..., pos:].reshape(*x.shape[:-1], self.tensors, 4, 4)
            t = torch.einsum("ztia,zthcab,ztjb->zthcij", matrices, t, matrices)
            parts.append((t * valid[..., None, None, None, None]).flatten(-3))
        return torch.cat(parts, -1)

    def forward(self, h, frames, frame_valid, mask, *, assignments=None,
                mass=None, region_mask=None, indices=None, neighbor_mask=None,
                tensor_gate=None, pair_bias=None, neighbor_pair_bias=None):
        if neighbor_pair_bias is not None and (indices is None or pair_bias is not None):
            raise ValueError("neighbor_pair_bias requires neighbor attention and no dense pair_bias")
        shape = (*h.shape[:2], 3, self.num_heads, self.head_dim)
        q_local, k_local, v_local = self.qkv(h).reshape(shape).unbind(2)
        dtype = torch.float64 if frames.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=h.device.type, enabled=False):
            matrices = inverse_lorentz(frames.to(dtype))
            q, k, v = [self.transport(x.to(dtype), matrices, frame_valid)
                       for x in (q_local, k_local, v_local)]
            if assignments is not None:
                if pair_bias is not None:
                    raise ValueError("Regional pair bias needs an explicit coarse-token definition")
                if indices is not None:
                    raise ValueError("Regional and neighbor modes are mutually exclusive")
                a = assignments.to(dtype)
                pooling = a / mass.to(dtype).clamp_min(1e-8)[:, None, :]
                q, k, v = [torch.einsum("bnk,bnhd->bkhd", pooling, x) for x in (q, k, v)]
                if tensor_gate is not None:
                    if not self.tensors or tensor_gate.shape != mass.shape:
                        raise ValueError("tensor_gate requires tensors and shape [B,K]")
                    start = self.scalars + 4 * self.vectors
                    q, k, v = [torch.cat((x[..., :start], x[..., start:] * tensor_gate[..., None, None]), -1)
                               for x in (q, k, v)]
                key_mask = region_mask
            else:
                key_mask = mask
                if tensor_gate is not None:
                    raise ValueError("Region gates are only defined for coarse attention")

            if indices is None:
                logits = torch.einsum("bihd,bjhd->bhij", q * self.metric, k) / math.sqrt(self.head_dim)
                if pair_bias is not None:
                    if pair_bias.shape != logits.shape:
                        raise ValueError("pair_bias must have shape [B,H,N,N]")
                    logits = logits + pair_bias.to(dtype)
                if assignments is not None:
                    logits = logits + mass.to(dtype).clamp_min(1e-8).log()[:, None, None, :]
                logits = logits.masked_fill(~key_mask[:, None, None, :], -torch.finfo(dtype).max)
                weight = logits.softmax(-1) * key_mask[:, None, None, :]
                weight = weight / weight.sum(-1, keepdim=True).clamp_min(1e-8)
                output = torch.einsum("bhij,bjhd->bihd", weight, v)
            else:
                keys, values = gather_tokens(k, indices), gather_tokens(v, indices)
                logits = ((q * self.metric)[:, :, None] * keys).sum(-1) / math.sqrt(self.head_dim)
                if pair_bias is not None:
                    expected = (h.shape[0], self.num_heads, h.shape[1], h.shape[1])
                    if pair_bias.shape != expected:
                        raise ValueError("pair_bias must have shape [B,H,N,N]")
                    local_bias = pair_bias.gather(3, indices[:, None].expand(-1, self.num_heads, -1, -1))
                    logits = logits + local_bias.permute(0, 2, 3, 1).to(dtype)
                if neighbor_pair_bias is not None:
                    expected = (h.shape[0], self.num_heads, h.shape[1], indices.shape[-1])
                    if neighbor_pair_bias.shape != expected:
                        raise ValueError("neighbor_pair_bias must have shape [B,H,N,L]")
                    logits = logits + neighbor_pair_bias.permute(0, 2, 3, 1).to(dtype)
                logits = logits.masked_fill(~neighbor_mask[..., None], -torch.finfo(dtype).max)
                weight = logits.softmax(2) * neighbor_mask[..., None]
                weight = weight / weight.sum(2, keepdim=True).clamp_min(1e-8)
                output = (weight[..., None] * values).sum(2)
            if assignments is not None:
                output = torch.einsum("bnk,bkhd->bnhd", a, output)
            output = self.transport(output, frames.to(dtype), frame_valid).flatten(-2)
        return self.out(output.to(h.dtype)) * mask[..., None]
