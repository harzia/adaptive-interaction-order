"""Pure-PyTorch Lorentz operations. Internal convention: (E, px, py, pz), +---.

Frames follow LLoCa's local-canonicalization construction, but this is an independent
small implementation, not the authors' pretrained model or a benchmark reproduction.
Only proper orthochronous transformations are claimed (the oriented fourth axis
uses the Levi-Civita tensor). Degenerate frames are flagged, never randomly repaired.
"""
from __future__ import annotations

from itertools import permutations
import torch


def metric_like(x):
    return x.new_tensor([1., -1., -1., -1.])


def minkowski_dot(x, y):
    return (x * y * metric_like(x)).sum(-1)


def inverse_lorentz(frame):
    g = metric_like(frame)
    return frame.transpose(-1, -2) * g[:, None] * g[None, :]


def levi_civita():
    eps = torch.zeros(4, 4, 4, 4, dtype=torch.float64)
    for p in permutations(range(4)):
        inversions = sum(p[i] > p[j] for i in range(4) for j in range(i + 1, 4))
        eps[p] = (-1) ** inversions
    return eps


def make_frames(vectors, epsilon, threshold=1e-10):
    """Orthonormalize (..., 3, 4) vectors, returning global->local frames and validity.

    Arithmetic is float64 even under AMP. Invalid frames are identity placeholders;
    callers MUST zero their geometric channels, since the placeholders are not
    covariant. Scalar processing remains defined. Validity uses invariant norms.
    """
    with torch.autocast(device_type=vectors.device.type, enabled=False):
        v0, v1, v2 = vectors.double().unbind(-2)
        n0 = minkowski_dot(v0, v0)
        e0 = v0 / n0.clamp_min(threshold).sqrt()[..., None]
        u1 = v1 - minkowski_dot(v1, e0)[..., None] * e0
        n1 = -minkowski_dot(u1, u1)
        e1 = u1 / n1.clamp_min(threshold).sqrt()[..., None]
        u2 = (v2 - minkowski_dot(v2, e0)[..., None] * e0
              + minkowski_dot(v2, e1)[..., None] * e1)
        n2 = -minkowski_dot(u2, u2)
        e2 = u2 / n2.clamp_min(threshold).sqrt()[..., None]
        g = metric_like(e0)
        e3 = -torch.einsum("ijkl,...j,...k,...l->...i", epsilon.double(),
                           e0 * g, e1 * g, e2 * g)
        basis = torch.stack((e0, e1, e2, e3), -2)
        frame = basis * g[..., None] * g
        valid = (n0 > threshold) & (n1 > threshold) & (n2 > threshold)
        identity = torch.eye(4, device=frame.device, dtype=frame.dtype)
        frame = torch.where(valid[..., None, None], frame, identity)
    return frame.to(vectors.dtype), valid


def gather_tokens(x, indices):
    """Gather [B,N,...] at [B,Q,L] without materializing an N*N feature tensor."""
    batch = torch.arange(x.shape[0], device=x.device)[:, None, None]
    return x[batch, indices]


def angular_neighbors(p, mask, time_reference, count):
    """Fixed kNN in the supplied observer's rest frame; self is eligible.

    The distance matrix is O(N^2), constructed once per forward; attention later
    gathers only O(N*count) neighbors. This is not an end-to-end linear algorithm.
    """
    if count < 1:
        raise ValueError("local_neighbors must be positive")
    energy = minkowski_dot(p, time_reference[:, None]).clamp_min(1e-8)
    direction = p / energy[..., None] - time_reference[:, None]
    norm = (-minkowski_dot(direction, direction)).clamp_min(1e-12).sqrt()
    direction = direction / norm[..., None]
    cosine = -torch.einsum("bni,bmi->bnm", direction * metric_like(p), direction)
    distance = (2 - 2 * cosine).clamp_min(0)
    distance = distance.masked_fill(~mask[:, None, :], float("inf"))
    indices = distance.topk(min(count, p.shape[1]), dim=-1, largest=False).indices
    neighbor_mask = gather_tokens(mask, indices) & mask[..., None]
    return indices, neighbor_mask, direction
