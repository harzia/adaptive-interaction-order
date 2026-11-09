"""Angular energy-flow features in one common frame, with SO(3)-typed algebra.

The quadrupole is stored in an orthonormal five-component STF basis. It is not a
general 4x4 Lorentz tensor. Keeping ParT's detector inputs does not confer full
Lorentz equivariance or IRC safety on the complete model.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def stf_basis():
    basis = torch.zeros(5, 3, 3, dtype=torch.float64)
    basis[0] = torch.diag(torch.tensor([1., -1., 0.], dtype=torch.float64)) / math.sqrt(2)
    basis[1] = torch.diag(torch.tensor([1., 1., -2.], dtype=torch.float64)) / math.sqrt(6)
    for channel, (i, j) in enumerate(((0, 1), (0, 2), (1, 2)), 2):
        basis[channel, i, j] = basis[channel, j, i] = 1 / math.sqrt(2)
    return basis


def encode_stf(matrix, basis):
    """The basis contraction implicitly symmetrizes and removes the trace."""
    return torch.einsum("...ij,lij->...l", matrix, basis.to(matrix))


def decode_stf(components, basis):
    return torch.einsum("...l,lij->...ij", components, basis.to(components))


def jet_angular_geometry(vectors, mask, *, frame="rest", rest_mass_fraction_min=1e-6):
    """px,py,pz,E [B,4,N] -> unit directions and energy fractions [B,N,...].

    Rest-frame arithmetic uses float64. A whole jet falls back to the detector
    frame if M^2/E^2 is below the threshold, the total is non-timelike, or the
    candidate boost yields negative particle energies. Zero spatial momenta have
    zero angular features (a monopole only). This fallback is detector referenced.
    """
    if frame not in ("rest", "detector") or not 0 < rest_mass_fraction_min < 1:
        raise ValueError("Invalid angular frame or rest mass threshold")
    if vectors.ndim != 3 or vectors.shape[1] != 4 or mask.shape != (len(vectors), 1, vectors.shape[-1]):
        raise ValueError("Expected vectors [B,4,N] and mask [B,1,N]")
    valid = mask[:, 0].bool()
    if not valid.any(1).all():
        raise ValueError("Every jet needs at least one valid particle")
    dtype = torch.float64 if vectors.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=vectors.device.type, enabled=False):
        p = vectors.double().transpose(1, 2).masked_fill(~valid[..., None], 0)
        if not torch.isfinite(p).all() or (p[..., 3] < 0).any():
            raise ValueError("Valid momenta must be finite with nonnegative energy")
        total = p.sum(1)
        if (total[:, 3] <= 0).any():
            raise ValueError("Every jet needs positive total energy")
        # Normalize before boosting: common input scaling cannot affect fallback.
        p = p / total[:, None, 3:4]
        beta = total[:, :3] / total[:, 3:4]
        beta2 = beta.square().sum(-1)
        mass_fraction = 1 - beta2
        rest_valid = mass_fraction > rest_mass_fraction_min
        gamma = mass_fraction.clamp_min(rest_mass_fraction_min).rsqrt()
        bp = (p[..., :3] * beta[:, None]).sum(-1)
        spatial = p[..., :3] + ((gamma.square() / (gamma + 1))[:, None] * bp
                               - gamma[:, None] * p[..., 3])[..., None] * beta[:, None]
        energy = gamma[:, None] * (p[..., 3] - bp)
        rest_valid = rest_valid & ((energy >= 0) | ~valid).all(1)
        used_rest = rest_valid if frame == "rest" else torch.zeros_like(rest_valid)
        spatial = torch.where(used_rest[:, None, None], spatial, p[..., :3])
        energy = torch.where(used_rest[:, None], energy, p[..., 3]).masked_fill(~valid, 0)
        direction_norm = torch.linalg.vector_norm(spatial, dim=-1)
        direction_valid = valid & (direction_norm > 1e-12)
        direction = (spatial / direction_norm.clamp_min(1e-12)[..., None]).masked_fill(~direction_valid[..., None], 0)
        fraction = energy / energy.sum(1, keepdim=True)
        quadrupole = encode_stf(direction[..., :, None] * direction[..., None, :], stf_basis().to(direction))
    return dict(direction=direction.to(dtype), quadrupole=quadrupole.to(dtype),
                energy_fraction=fraction.to(dtype), direction_valid=direction_valid,
                used_rest=used_rest, mass_fraction=mass_fraction.to(dtype))


class AngularEmbedding(nn.Module):
    """Shared amplitude a=z*softplus(f(h)) seeds all retained angular degrees."""
    def __init__(self, hidden_dim, heads, channels, order):
        super().__init__()
        self.heads, self.channels, self.order = heads, channels, order
        self.amplitude = nn.Linear(hidden_dim, heads * channels)

    def forward(self, h, geometry, mask):
        dtype = self.amplitude.weight.dtype
        with torch.autocast(device_type=h.device.type, enabled=False):
            a = F.softplus(self.amplitude(h.to(dtype))).reshape(*h.shape[:2], self.heads, self.channels)
            a = a * geometry["energy_fraction"].to(a)[..., None, None]
            state = [a[..., None], a[..., None] * geometry["direction"].to(a)[:, :, None, None]]
            if self.order == 2:
                state.append(a[..., None] * geometry["quadrupole"].to(a)[:, :, None, None])
        return tuple(x.masked_fill(~mask[:, :, None, None, None], 0) for x in state)


class AngularUpdate(nn.Module):
    """Channel mixing -> shared attention -> scalar contractions / typed products.

    State shape per degree: [B,N,H,C,2*l+1]. All nonlinear geometric algebra runs
    in parameter precision (normally fp32, including under AMP). Only channel
    indices have learned linear maps; spatial indices obey the specified algebra.
    """
    def __init__(self, hidden_dim, heads, channels, order=2, product_degree=2, carry=True):
        super().__init__()
        if order not in (1, 2) or product_degree not in (1, 2) or min(heads, channels) < 1:
            raise ValueError("order/product_degree must be 1 or 2 and channels/heads positive")
        self.heads, self.channels, self.order = heads, channels, order
        self.product_degree, self.carry = product_degree, carry
        self.widths = [1, 3] + ([5] if order == 2 else [])
        self.value_mix = nn.ParameterList([
            nn.Parameter(torch.eye(channels).repeat(heads, 1, 1)) for _ in self.widths])
        self.register_buffer("basis", stf_basis(), persistent=False)
        self.register_buffer("pairs", torch.triu_indices(channels, channels), persistent=False)
        # Linear scalarization S, V.n, T.Q plus invariant normalization scale.
        features = channels * len(self.widths) + 1
        if product_degree == 2:
            features += len(self.pairs[0]) * order  # VV, and TT when order=2
        self.scalar_out = nn.Linear(heads * features, hidden_dim, bias=False)
        nn.init.zeros_(self.scalar_out.weight)
        if carry:
            self.state_gate = nn.Linear(hidden_dim, heads * channels * len(self.widths))
            if product_degree == 2 and order == 2:
                self.tv_mix = nn.Parameter(torch.zeros(heads, channels, channels * channels))
                self.vv_mix = nn.Parameter(torch.zeros(heads, channels, channels * channels))

    def values(self, state):
        dtype = self.scalar_out.weight.dtype
        with torch.autocast(device_type=state[0].device.type, enabled=False):
            mixed = [torch.einsum("hoc,bnhcd->bnhod", w, x.to(dtype)) for w, x in zip(self.value_mix, state)]
            return torch.cat([x.flatten(-2) for x in mixed], -1).permute(0, 2, 1, 3)

    def update(self, h, packed, previous, geometry, mask):
        dtype = self.scalar_out.weight.dtype
        with torch.autocast(device_type=h.device.type, enabled=False):
            chunks = packed.to(dtype).split([self.channels * w for w in self.widths], -1)
            messages = [x.reshape(*x.shape[:-1], self.channels, w) for x, w in zip(chunks, self.widths)]
            # One invariant norm per particle/head, shared across all degrees.
            norm2 = sum(x.square().sum(-1).mean(-1) for x in messages)
            scale = (norm2 + 1e-8).sqrt()
            normalized = [x / scale[..., None, None] for x in messages]
            s, v = normalized[:2]
            n = geometry["direction"].to(v)[:, :, None, None]
            features = [s.squeeze(-1), (v * n).sum(-1), scale.log1p()[..., None]]
            if self.order == 2:
                t = normalized[2]
                q = geometry["quadrupole"].to(t)[:, :, None, None]
                features.append((t * q).sum(-1))
            if self.product_degree == 2:
                left, right = self.pairs
                features.append((v[..., left, :] * v[..., right, :]).sum(-1))
                if self.order == 2:
                    features.append((t[..., left, :] * t[..., right, :]).sum(-1))
            delta = self.scalar_out(torch.cat(features, -1).flatten(-2))
            delta = delta.masked_fill(~mask[..., None], 0)
            next_state = None
            if self.carry:
                updates = list(normalized)
                if self.product_degree == 2 and self.order == 2:
                    matrices = decode_stf(t, self.basis)
                    tv = torch.einsum("bnhcij,bnhdj->bnhcdi", matrices, v).flatten(-3, -2)
                    vv = torch.einsum("bnhci,bnhdj,lij->bnhcdl", v, v, self.basis.to(v)).flatten(-3, -2)
                    updates[1] = updates[1] + torch.einsum("hoc,bnhcd->bnhod", self.tv_mix, tv)
                    updates[2] = updates[2] + torch.einsum("hoc,bnhcd->bnhod", self.vv_mix, vv)
                gates = self.state_gate(h.to(dtype)).reshape(*h.shape[:2], self.heads, len(self.widths), self.channels).sigmoid()
                next_state = tuple((old.to(dtype) + gates[..., i, :, None] * update)
                                   .masked_fill(~mask[:, :, None, None, None], 0)
                                   for i, (old, update) in enumerate(zip(previous, updates)))
        return delta.to(h.dtype), next_state
