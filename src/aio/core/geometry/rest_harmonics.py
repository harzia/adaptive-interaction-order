"""Rest-frame dipole and quadrupole features for the harmonic score control."""
import math
import torch


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


def rest_harmonics(vectors, mask):
    """Map px,py,pz,E [B,4,N] to dipole and quadrupole components [B,N,8].

    Rest-frame arithmetic uses float64. A whole jet falls back to the detector
    frame if M^2/E^2 is below the threshold, the total is non-timelike, or the
    candidate boost yields negative particle energies. Zero spatial momenta have
    zero angular features (a monopole only). This fallback is detector referenced.
    """
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
        rest_valid = mass_fraction > 1e-6
        gamma = mass_fraction.clamp_min(1e-6).rsqrt()
        bp = (p[..., :3] * beta[:, None]).sum(-1)
        spatial = p[..., :3] + ((gamma.square() / (gamma + 1))[:, None] * bp
                               - gamma[:, None] * p[..., 3])[..., None] * beta[:, None]
        energy = gamma[:, None] * (p[..., 3] - bp)
        rest_valid = rest_valid & ((energy >= 0) | ~valid).all(1)
        spatial = torch.where(rest_valid[:, None, None], spatial, p[..., :3])
        direction_norm = torch.linalg.vector_norm(spatial, dim=-1)
        direction_valid = valid & (direction_norm > 1e-12)
        direction = (spatial / direction_norm.clamp_min(1e-12)[..., None]).masked_fill(~direction_valid[..., None], 0)
        quadrupole = encode_stf(direction[..., :, None] * direction[..., None, :], stf_basis().to(direction))
    return torch.cat((direction, quadrupole), -1).to(dtype)
