"""Hard region cuts of a Cambridge/Aachen tree, using FastJet E-scheme merging.

This is a hierarchy control, not a LundNet implementation. All input constituents
are reclustered into one tree, then its exclusive K-subjet cut supplies the regions.
No primary-branch selection, grooming, or new node features enter the classifier.
Clustering uses detector-frame rapidity/azimuth, not full Lorentz invariance.
"""
from __future__ import annotations

import math
import warnings
import torch
from .regions import LearnedRegionGeometry


@torch.no_grad()
def ca_assignments(momenta, mask, num_regions, radius=1000.):
    """Batched FastJet clustering for training; no per-particle Python objects.

Group jets by min(K,N), because FastJet's exclusive cut size is shared across
an Awkward batch. Preserve original constituent indices even with padding holes.
Detailed node variables are extracted separately by ca_tree_regions offline.
"""
    try:
        import fastjet as fj
        import awkward as ak
        import numpy as np
    except ImportError as exc:
        raise ImportError("C/A controls require: pip install -r requirements-lund.txt") from exc
    if num_regions < 1 or not math.isfinite(radius) or radius <= 0:
        raise ValueError("Positive num_regions and finite positive C/A radius required")
    if momenta.ndim != 3 or momenta.shape[-1] != 4 or mask.shape != momenta.shape[:2]:
        raise ValueError("Expected momenta [B,N,4] and mask [B,N]")
    p = momenta.detach().double().cpu().numpy()
    valid = mask.detach().bool().cpu().numpy()
    if not valid.any(1).all() or not np.isfinite(p[valid]).all() or (p[..., 0][valid] < 0).any():
        raise ValueError("C/A needs finite nonnegative-energy constituents and nonempty jets")
    batch, n = valid.shape
    a = np.zeros((batch, n, num_regions), dtype=np.float64)
    records = ak.unflatten(
        ak.zip({name: p[..., component][valid]
                for name, component in (("E", 0), ("px", 1), ("py", 2), ("pz", 3))}),
        valid.sum(1))
    # Compact positions returned by FastJet -> original input positions.
    original_indices = np.argsort(~valid, axis=1, kind="stable")
    targets = np.minimum(valid.sum(1), num_regions)
    definition = fj.JetDefinition(fj.cambridge_algorithm, radius, fj.E_scheme)
    for target in np.unique(targets):
        ids = np.flatnonzero(targets == target)
        sequence = fj.ClusterSequence(ak.to_packed(records[ids]), definition)
        if not np.all(ak.to_numpy(ak.num(sequence.inclusive_jets(), axis=1)) == 1):
            raise ValueError("C/A radius did not produce one tree; check inputs or increase ca_radius")
        # FastJet 3.5.2's Python wrapper incorrectly warns for a JetDefinition
        # object even when its algorithm is C/A. Suppress only that known warning.
        formatter = warnings.formatwarning
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="dcut and exclusive jets for jet-finders other than kt, C/A or genkt.*", category=UserWarning)
                jets = sequence.exclusive_jets(n_jets=int(target))
                members = sequence.exclusive_jets_constituent_index(njets=int(target))
        finally:
            warnings.formatwarning = formatter
        order = ak.argsort(jets.px**2 + jets.py**2, axis=1, ascending=False, stable=True)
        members = members[order]
        b, r, compact = ak.broadcast_arrays(
            ak.Array(ids)[:, None, None],
            ak.Array(np.broadcast_to(np.arange(target), (len(ids), target)))[:, :, None], members)
        b, r, compact = [ak.to_numpy(ak.flatten(x, axis=None)) for x in (b, r, compact)]
        a[b, original_indices[b, compact], r] = 1
    if not np.array_equal(a.sum(-1), valid):
        raise RuntimeError("C/A cut did not partition the supplied constituents")
    return torch.from_numpy(a).to(device=momenta.device, dtype=momenta.dtype)


@torch.no_grad()
def ca_tree_regions(momenta, mask, num_regions, radius=1000.):
    """Time-first momenta [B,N,4]; return assignments and cut-node Lund variables.

FastJet runs on the CPU on detached, double-precision momenta. The large radius
keeps a jet's supplied constituents in one C/A tree; we explicitly check this.
Empty slots when N<K have zero mass. kt/mass use INPUT momentum units (JetClass-II
normally rescales momenta), so kt_fraction is supplied for scale-free inspection.
"""
    try:
        import fastjet as fj
    except ImportError as exc:
        raise ImportError("C/A controls require: pip install -r requirements-lund.txt") from exc
    if num_regions < 1 or not math.isfinite(radius) or radius <= 0:
        raise ValueError("Positive num_regions and finite positive C/A radius required")
    if momenta.ndim != 3 or momenta.shape[-1] != 4 or mask.shape != momenta.shape[:2]:
        raise ValueError("Expected momenta [B,N,4] and mask [B,N]")
    p = momenta.detach().double().cpu()
    valid = mask.detach().bool().cpu()
    if not valid.any(1).all() or not torch.isfinite(p[valid]).all():
        raise ValueError("C/A needs at least one finite valid constituent per jet")
    if (p[..., 0][valid] < 0).any():
        raise ValueError("C/A requires nonnegative constituent energies")
    b, n = valid.shape
    a = torch.zeros(b, n, num_regions, dtype=torch.float64)
    fields = {name: torch.zeros(b, num_regions, dtype=torch.float64)
              for name in ("lund_delta", "lund_z", "lund_kt", "lund_kt_fraction", "lund_mass")}
    split_valid = torch.zeros(b, num_regions, dtype=torch.bool)
    definition = fj.JetDefinition(fj.cambridge_algorithm, radius, fj.E_scheme)
    for batch in range(b):
        particles = []
        for index in valid[batch].nonzero(as_tuple=True)[0].tolist():
            e, px, py, pz = p[batch, index].tolist()
            particle = fj.PseudoJet(px, py, pz, e)
            particle.set_user_index(index)
            particles.append(particle)
        sequence = fj.ClusterSequence(particles, definition)
        if len(sequence.inclusive_jets()) != 1:
            raise ValueError("C/A radius did not produce one tree; check inputs or increase ca_radius")
        nodes = fj.sorted_by_pt(sequence.exclusive_jets(min(num_regions, len(particles))))
        total_pt = sum(node.pt() for node in particles)
        for region, node in enumerate(nodes):
            for constituent in node.constituents():
                a[batch, constituent.user_index(), region] = 1
            fields["lund_mass"][batch, region] = math.sqrt(max(node.m2(), 0.))
            left, right = fj.PseudoJet(), fj.PseudoJet()
            if node.has_parents(left, right):
                delta = left.delta_R(right)
                soft_pt = min(left.pt(), right.pt())
                fields["lund_delta"][batch, region] = delta
                fields["lund_z"][batch, region] = soft_pt / max(left.pt() + right.pt(), 1e-12)
                fields["lund_kt"][batch, region] = soft_pt * delta
                fields["lund_kt_fraction"][batch, region] = soft_pt * delta / max(total_pt, 1e-12)
                split_valid[batch, region] = True
    # Every valid constituent appears exactly once, including jets shorter than K.
    if not torch.equal(a.sum(-1), valid.double()):
        raise RuntimeError("C/A cut did not partition the supplied constituents")
    fields.update(assignments=a, lund_split_valid=split_valid)
    return {name: value.to(device=momenta.device,
                           dtype=torch.bool if value.dtype == torch.bool else momenta.dtype)
            for name, value in fields.items()}


class CambridgeAachenGeometry(LearnedRegionGeometry):
    """Keep learned particle frames and the local graph; replace only grouping."""
    def __init__(self, input_dim, num_regions, hidden_dim=32, local_neighbors=8,
                 temperature=0.5, balance_weight=0.01, entropy_weight=0.001,
                 locality_weight=0.01, radius=1000.):
        if num_regions < 1:
            raise ValueError("C/A geometry requires num_regions > 0")
        # No unused assignment parameters or grouping penalties in the fixed cut.
        super().__init__(input_dim, 0, hidden_dim, local_neighbors, temperature, 0., 0., 0.)
        self.num_regions = num_regions
        self.radius = radius
        self.spec.update(num_regions=num_regions, region_source="ca", ca_radius=radius)

    def forward(self, features, p, mask, references=None):
        geometry = super().forward(features, p, mask, references)
        assignments = ca_assignments(p, mask, self.num_regions, self.radius)
        mass = assignments.sum(1)
        geometry.update(assignments=assignments, mass=mass, region_mask=mass > 0)
        geometry["diagnostics"].update(
            occupied_regions=(mass > 0).sum(1).float().mean(),
            invalid_frame_fraction=((mask & ~geometry["frame_valid"]).sum()
                                    / mask.sum().clamp_min(1)).detach())
        return geometry
