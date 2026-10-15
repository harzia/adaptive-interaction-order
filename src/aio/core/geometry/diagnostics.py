"""Offline diagnostics for learned assignments and tensor-channel interventions."""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F


def assignment_statistics(assignments, mask, momenta):
    """Per-jet and per-region statistics; momenta [B,N,4] are time-first.

    Angular dispersion is 1-|mean unit direction|^2 in the supplied lab frame,
    NOT a Lorentz-invariant angle or an RMS angle in radians. Soft slot occupancy
    alone cannot diagnose collapse: uniform assignments occupy every slot.
    """
    a = assignments.detach().float().masked_fill(~mask[..., None], 0)
    k = a.shape[-1]
    count = mask.sum(1).clamp_min(1)
    mass = a.sum(1)
    occupancy = mass / count[:, None]
    entropy = -(a * a.clamp_min(1e-12).log()).sum(-1)
    hard = F.one_hot(a.argmax(-1), k) * mask[..., None]
    hard_mass = hard.sum(1)
    gram = torch.einsum("bnk,bnl->bkl", a, a)
    norms = gram.diagonal(dim1=-2, dim2=-1).clamp_min(1e-20).sqrt()
    overlap = gram / (norms[:, :, None] * norms[:, None, :])
    off_diagonal = ~torch.eye(k, dtype=torch.bool, device=a.device)
    mean_overlap = (overlap[:, off_diagonal].mean(-1) if k > 1 else a.new_zeros(a.shape[0]))
    spatial = momenta.detach().float()[..., 1:].masked_fill(~mask[..., None], 0)
    nonzero = spatial.square().sum(-1) > 1e-20
    direction = F.normalize(spatial, dim=-1, eps=1e-10)
    angular_weight = a * nonzero[..., None]
    angular_mass = angular_weight.sum(1)
    mean_direction = torch.einsum("bnk,bnd->bkd", angular_weight, direction) / angular_mass.clamp_min(1e-8)[..., None]
    dispersion = (1 - mean_direction.square().sum(-1)).clamp(0, 1)
    dispersion = dispersion.masked_fill(angular_mass == 0, 0)
    return dict(
        region_mass=mass, region_fraction=occupancy, hard_region_mass=hard_mass,
        effective_occupied_regions=(-(occupancy * occupancy.clamp_min(1e-12).log()).sum(-1)).exp(),
        hard_occupied_regions=(hard_mass > 0).sum(-1),
        assignment_entropy=(entropy * mask).sum(1) / count,
        normalized_assignment_entropy=(entropy * mask).sum(1) / count / max(math.log(k), 1e-12),
        mean_max_assignment=(a.max(-1).values * mask).sum(1) / count,
        mean_region_overlap=mean_overlap, angular_dispersion=dispersion,
        angular_mass=angular_mass)


def tensor_interventions(model, features, vectors, mask, labels):
    """Label-dependent diagnostics, not a deployable router or a speed benchmark.

    Turn off a region's rank-2 Q,K,V in every block, first jointly and then one
    region at a time. Delta CE = CE(intervened)-CE(original); positive means the
    tensor contribution helped this example. Derivative estimates use -dCE/dgate.
    Interventions are not additive and gate=0 is not a retrained SV model.
    """
    if model.training:
        raise ValueError("Tensor diagnostics require eval mode")
    if not model.num_regions or not all(b.global_attention.tensors for b in model.blocks):
        raise ValueError("Tensor interventions require a regional rank-2 model")
    shape = (features.shape[0], model.num_regions)
    gate = torch.ones(shape, dtype=features.dtype, device=features.device, requires_grad=True)
    with torch.enable_grad():
        logits = model(features, vectors, mask, tensor_gate=gate)
        losses = F.cross_entropy(logits, labels, reduction="none")
        gradient, = torch.autograd.grad(losses.sum(), gate)
    base_losses = losses.detach()
    base_logits = logits.detach()
    per_region = []
    with torch.no_grad():
        off = model(features, vectors, mask, tensor_gate=torch.zeros_like(gate))
        for region in range(model.num_regions):
            one_off = torch.ones_like(gate)
            one_off[:, region] = 0
            ablated = model(features, vectors, mask, tensor_gate=one_off)
            per_region.append(F.cross_entropy(ablated, labels, reduction="none") - base_losses)
    return dict(base_ce=base_losses, all_tensors_off_delta_ce=F.cross_entropy(off, labels, reduction="none") - base_losses,
                region_off_delta_ce=torch.stack(per_region, -1),
                first_order_region_delta_ce=-gradient.detach(),
                all_tensors_off_changed_prediction=(off.argmax(-1) != base_logits.argmax(-1)),
                logits=base_logits)
