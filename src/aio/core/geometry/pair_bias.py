"""ParT-inspired pair features evaluated in each receiving particle's local frame.

The feature definitions follow particle_transformer.pairwise_lv_fts_pp, with
floors applied before roots/atan2 to keep frame gradients finite at self pairs.
The small pointwise MLP is NOT the published ParT BatchNorm embedding.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from .lorentz import gather_tokens


def pair_features(receiver, sender, eps=1e-8):
    """Broadcast time-first four-vectors; return (..., 4): log kt,z,delta,m2."""
    def coordinates(p):
        e, x, y, z = p.unbind(-1)
        pt2 = x.square() + y.square()
        regular = pt2 > eps**2
        phi = torch.atan2(torch.where(regular, y, 0.), torch.where(regular, x, 1.))
        rapidity = 0.5 * ((e + z).clamp_min(eps).log() - (e - z).clamp_min(eps).log())
        return pt2.clamp_min(eps**2).sqrt(), rapidity, phi

    pti, yi, phii = coordinates(receiver)
    ptj, yj, phij = coordinates(sender)
    dphi = (phii - phij + math.pi) % (2 * math.pi) - math.pi
    delta = ((yi - yj).square() + dphi.square()).clamp_min(eps**2).sqrt()
    ptmin = torch.minimum(pti, ptj)
    total = receiver + sender
    m2 = total[..., 0].square() - total[..., 1:].square().sum(-1)
    return torch.stack(((ptmin * delta).clamp_min(eps).log(),
                        (ptmin / (pti + ptj).clamp_min(eps)).clamp_min(eps).log(),
                        delta.log(), m2.clamp_min(eps).log()), -1)


class LocalFramePairBias(nn.Module):
    """Shared across blocks; output [B,H,receiver,sender], generally asymmetric.

Invalid receiving frames disable the bias for that row. Sender frame validity
does not matter: this branch transports physical momenta, not sender latents.
The complete matrix is O(N^2), including its feature construction. Supplying
indices instead constructs only [B,H,N,L] local biases, with O(NL) features.
"""
    def __init__(self, num_heads, hidden_dim=32):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("pair_bias_dim must be positive")
        self.net = nn.Sequential(nn.Linear(4, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                 nn.Linear(hidden_dim, num_heads))

    def forward(self, p, frames, frame_valid, mask, *, indices=None, neighbor_mask=None):
        with torch.autocast(device_type=p.device.type, enabled=False):
            # Evaluate features in fp32 (fp64 for numerical checks), including
            # differentiable frames. Padded momenta have already been zeroed.
            if indices is None:
                local_sender = torch.einsum("biac,bjc->bija", frames, p)
                valid = mask[:, :, None] & mask[:, None, :] & frame_valid[:, :, None]
            else:
                if neighbor_mask is None or neighbor_mask.shape != indices.shape:
                    raise ValueError("Sparse pair bias requires neighbor_mask matching indices")
                local_sender = torch.einsum("biac,bijc->bija", frames, gather_tokens(p, indices))
                valid = neighbor_mask & mask[:, :, None] & frame_valid[:, :, None]
            local_receiver = torch.einsum("biac,bic->bia", frames, p)[:, :, None]
            features = pair_features(local_receiver, local_sender)
            features = features.masked_fill(~valid[..., None], 0)
        bias = self.net(features.to(self.net[0].weight.dtype))
        return bias.masked_fill(~valid[..., None], 0).permute(0, 3, 1, 2)
