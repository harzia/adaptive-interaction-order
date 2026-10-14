"""Input-only learned regions: independently checkpointable from message order."""
from __future__ import annotations

import torch
from torch import nn
from .lorentz import angular_neighbors, gather_tokens, levi_civita, make_frames, minkowski_dot


class LearnedRegionGeometry(nn.Module):
    def __init__(self, input_dim, num_regions, hidden_dim=32, local_neighbors=8,
                 temperature=0.5, balance_weight=0.01, entropy_weight=0.001,
                 locality_weight=0.01):
        super().__init__()
        if num_regions < 0 or temperature <= 0:
            raise ValueError("num_regions >= 0 and temperature > 0 required")
        self.spec = dict(input_dim=input_dim, num_regions=num_regions,
                         hidden_dim=hidden_dim, local_neighbors=local_neighbors,
                         temperature=temperature, balance_weight=balance_weight,
                         entropy_weight=entropy_weight, locality_weight=locality_weight)
        self.num_regions = num_regions
        self.local_neighbors = local_neighbors
        self.temperature = temperature
        self.balance_weight = balance_weight
        self.entropy_weight = entropy_weight
        self.locality_weight = locality_weight
        self.encoder = nn.Sequential(nn.Linear(input_dim + 4, hidden_dim), nn.GELU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.GELU())
        self.frame_weights = nn.Linear(hidden_dim, 3)
        self.assignment = nn.Linear(hidden_dim, num_regions) if num_regions else None
        self.register_buffer("epsilon", levi_civita(), persistent=False)

    def forward(self, features, p, mask, references=None):
        # Features are scalars under the intended symmetry. Supplied JetClass-II
        # lab features only preserve the detector's residual SO(2), not all boosts.
        with torch.autocast(device_type=p.device.type, enabled=False):
            dtype = torch.float64 if p.dtype == torch.float64 else torch.float32
            p, features = p.to(dtype), features.to(dtype)
            if references is None:
                references = p.new_tensor([[1., 0., 0., 0.], [0., 0., 0., 1.]])
                references = references[None].expand(p.shape[0], -1, -1)
            references = references.to(p)
            if references.shape != (p.shape[0], 2, 4):
                raise ValueError("references must have shape [B,2,4]: unit time and beam vectors")
            time, beam = references.unbind(1)
            indices, neighbor_mask, direction = angular_neighbors(
                p, mask, time, self.local_neighbors)
            energy = minkowski_dot(p, time[:, None]).clamp_min(1e-8)
            unit_p = p / energy[..., None]
            total_e = (energy * mask).sum(1, keepdim=True).clamp_min(1e-8)
            descriptor = torch.stack((energy.log(),
                                      minkowski_dot(p, p) / energy.square(),
                                      minkowski_dot(unit_p, beam[:, None]),
                                      energy / total_e), -1)
            descriptor = descriptor.masked_fill(~mask[..., None], 0)
            h = self.encoder(torch.cat((features, descriptor), -1))
            weights = self.frame_weights(h)
            local = gather_tokens(unit_p, indices) * neighbor_mask[..., None]
            local = local.sum(2) / neighbor_mask.sum(2).clamp_min(1)[..., None]
            # A convex physical direction plus the unit observer is strictly timelike.
            # Its boost relative to that observer is bounded without a lab-frame clamp.
            v0 = time[:, None] + weights[..., 0:1].sigmoid() * (unit_p + local) / 2
            v1 = beam[:, None] + 0.25 * weights[..., 1:2].tanh() * unit_p
            v2 = unit_p + 0.5 * weights[..., 2:3].sigmoid() * local
            frames, frame_valid = make_frames(torch.stack((v0, v1, v2), -2), self.epsilon)
            frame_valid = frame_valid & mask
            result = dict(frames=frames, frame_valid=frame_valid,
                          indices=indices, neighbor_mask=neighbor_mask)
            if self.assignment is None:
                result.update(assignments=None, mass=None, region_mask=None,
                              regularization=h.sum() * 0, diagnostics={})
                return result
            assignment = (self.assignment(h) / self.temperature).softmax(-1)
            assignment = assignment * mask[..., None]
            mass = assignment.sum(1)
            total = mask.sum(1, keepdim=True).clamp_min(1)
            occupancy = mass / total
            balance = ((occupancy * self.num_regions - 1).square()).mean()
            entropy = -(assignment * assignment.clamp_min(1e-12).log()).sum() / mask.sum().clamp_min(1)
            means = torch.einsum("bnk,bnd->bkd", assignment, direction) / mass.clamp_min(1e-8)[..., None]
            delta = direction[:, :, None] - means[:, None]
            spread = -minkowski_dot(delta, delta)
            locality = (assignment * spread.clamp_min(0)).sum() / mask.sum().clamp_min(1)
            regularization = (self.balance_weight * balance + self.entropy_weight * entropy
                              + self.locality_weight * locality)
            diagnostics = dict(balance=balance.detach(), assignment_entropy=entropy.detach(),
                               locality=locality.detach(), min_region_mass=mass.min().detach(),
                               invalid_frame_fraction=((mask & ~frame_valid).sum() / mask.sum().clamp_min(1)).detach())
            result.update(assignments=assignment, mass=mass, region_mask=mass > 1e-6,
                          regularization=regularization, diagnostics=diagnostics)
            return result
