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
    # Center before taking products: subtracting nearly equal second moments is
    # inaccurate for narrow boosted jets. These are lab-frame direction moments,
    # assignment-weighted like the model's pooling, not energy-weighted multipoles.
    u = F.normalize(spatial.double(), dim=-1, eps=1e-10)
    w = angular_weight.double()
    center = torch.einsum("bnk,bnd->bkd", w, u) / angular_mass.double().clamp_min(1e-8)[..., None]
    residual = u[:, :, None] - center[:, None]
    covariance = torch.einsum("bnk,bnki,bnkj->bkij", w, residual, residual)
    covariance = covariance / angular_mass.double().clamp_min(1e-8)[..., None, None]
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0)
    trace = eigenvalues.sum(-1)
    shape_valid = (angular_mass > 0) & (trace > 1e-12)
    anisotropy = ((eigenvalues[..., -1] - eigenvalues[..., 0]) / trace.clamp_min(1e-12))
    anisotropy = anisotropy.masked_fill(~shape_valid, 0)
    result = dict(
        region_mass=mass, region_fraction=occupancy, hard_region_mass=hard_mass,
        effective_occupied_regions=(-(occupancy * occupancy.clamp_min(1e-12).log()).sum(-1)).exp(),
        hard_occupied_regions=(hard_mass > 0).sum(-1),
        assignment_entropy=(entropy * mask).sum(1) / count,
        normalized_assignment_entropy=(entropy * mask).sum(1) / count / max(math.log(k), 1e-12),
        mean_max_assignment=(a.max(-1).values * mask).sum(1) / count,
        mean_region_overlap=mean_overlap, angular_dispersion=dispersion,
        angular_mass=angular_mass, constituent_count=mask.sum(1),
        region_mask=mass > 1e-6,
        effective_region_constituents=mass.square() / a.square().sum(1).clamp_min(1e-12),
        angular_covariance_eigenvalues=eigenvalues.float(),
        angular_covariance_trace=trace.float(), angular_anisotropy=anisotropy.float(),
        angular_shape_valid=shape_valid)
    # Coordinates for reproducible assignment maps, centred on the input jet.
    p = momenta.detach().double().masked_fill(~mask[..., None], 0)
    total = p.sum(1)
    rapidity = .5 * ((p[..., 0] + p[..., 3]).clamp_min(1e-10).log()
                     - (p[..., 0] - p[..., 3]).clamp_min(1e-10).log())
    jet_y = .5 * ((total[:, 0] + total[:, 3]).clamp_min(1e-10).log()
                  - (total[:, 0] - total[:, 3]).clamp_min(1e-10).log())
    phi = torch.atan2(p[..., 2], p[..., 1])
    jet_phi = torch.atan2(total[:, 2], total[:, 1])
    pt = p[..., 1:3].square().sum(-1).sqrt()
    result.update(rapidity_offset=(rapidity - jet_y[:, None]).float().masked_fill(~mask, 0),
                  phi_offset=((phi - jet_phi[:, None] + math.pi) % (2 * math.pi) - math.pi).float().masked_fill(~mask, 0),
                  constituent_pt_fraction=(pt / pt.sum(1).clamp_min(1e-10)[:, None]).float())
    return result


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


@torch.no_grad()
def communication_interventions(model, features, vectors, mask, labels):
    """Remove global messages or existing pair biases in a trained checkpoint.

The local graph and particle readout remain. Global gating occurs after attention
output projection in every block, including its bias. It saves no computation.
This tests reliance, not the performance of a model retrained without a branch.
"""
    if model.training or model.for_inference:
        raise ValueError("Communication diagnostics require eval mode and raw logits")
    baseline = model(features, vectors, mask)
    ce = F.cross_entropy(baseline, labels, reduction="none")
    interventions = {"global_off": {"global_gate": 0.}}
    if model.pair_bias is not None:
        scope = model.pair_bias_scope
        interventions["pair_all_off"] = {"pair_bias_scope": "none"}
        if scope in ("both", "local"):
            interventions["pair_local_off"] = {"pair_bias_scope": "global" if scope == "both" else "none"}
        if scope in ("both", "global"):
            interventions["pair_global_off"] = {"pair_bias_scope": "local" if scope == "both" else "none"}
    result = dict(base_ce=ce, logits=baseline)
    for name, options in interventions.items():
        logits = model(features, vectors, mask, **options)
        result[name + "_delta_ce"] = F.cross_entropy(logits, labels, reduction="none") - ce
        result[name + "_changed_prediction"] = logits.argmax(-1) != baseline.argmax(-1)
    return result
