"""JetClass-II relative attention on the matched four-block ParT backbone."""
from functools import partial
import math

import torch
from torch.nn import functional as F

from aio.core.geometry.rest_harmonics import rest_harmonics
from aio.core.geometry.relative_attention import RelativeAttention
from .particle_transformer import ParticleTransformer, pairwise_lv_fts_pp


def geometry_coordinates(points, vectors):
    # JetClass-II supplies unstandardized axis offsets and p^mu * 500 / jet_pt.
    deta, dphi = points.unbind(1)
    dphi = math.pi - (math.pi - dphi).remainder(math.tau)
    fraction = torch.linalg.vector_norm(vectors[:, :2], dim=1) / 500.
    return torch.stack((deta, dphi, fraction.clamp_min(1e-12).log()), -1)


def pair_features(receivers, senders, scale_invariant=False):
    # Use actual four-momenta and ParT's eps, rapidity, wrapped phi, self pairs
    # and native order (ln kt, ln z, ln Delta, ln m^2).
    features = pairwise_lv_fts_pp(receivers.unsqueeze(-1), senders.unsqueeze(-2))
    if scale_invariant:
        features = features[:, 1:3]
    return features.movedim(1, -1).contiguous()


def origin_vector(v, coords, mask):
    # A massless reference along the data jet axis with pt/jet_pt = 1.
    first = mask[:, 0].int().argmax(-1)
    b = torch.arange(len(v), device=v.device)
    px, py, pz, _ = v[b, :, first].unbind(-1)
    eta = torch.asinh(pz / torch.hypot(px, py).clamp_min(1e-12)) - coords[b, first, 0]
    phi = torch.atan2(py, px) - coords[b, first, 1]
    return 500 * torch.stack((phi.cos(), phi.sin(), eta.sinh(), eta.cosh()), 1).unsqueeze(-1)


class RelativeParticleTransformer(ParticleTransformer):
    def __init__(self, input_dim, num_classes, *, score="linear", rotate_values=True,
                 bank="full", value_bank=None, angular=True, momentum=True, directions=3,
                 freeze_frequencies=False, scale_invariant_pairs=False,
                 class_at_origin=False, control_rank=8, block_params=None,
                 cls_block_params=None, **kwargs):
        self.score = score
        self.scale_invariant_pairs = scale_invariant_pairs
        self.class_at_origin = class_at_origin
        factory = partial(RelativeAttention, score=score, rotate_values=rotate_values,
                          bank=bank, value_bank=value_bank, angular=angular, momentum=momentum,
                          directions=directions, freeze_frequencies=freeze_frequencies,
                          scale_invariant_pairs=scale_invariant_pairs, control_rank=control_rank)
        block_params = dict(block_params or {}, attention_factory=factory)
        if class_at_origin:
            cls_block_params = dict(cls_block_params or {}, attention_factory=factory)
        super().__init__(input_dim, num_classes, pair_embed_dims=None, block_params=block_params,
                         cls_block_params=cls_block_params, **kwargs)
        if self.cls_blocks is None or self.include_global_token or self.for_segmentation:
            raise ValueError("RelativeParT requires the ParT class-attention readout")

    def forward(self, x, v, mask, *, points, export_embed=False):
        # Trim/permutate coordinates, features and four-vectors with one index map.
        x, packed, mask, _ = self.trimmer(x, torch.cat((v, points), 1), mask)
        v, points = packed.masked_fill(~mask, 0).split((4, 2), dim=1)
        with torch.autocast(device_type=v.device.type, enabled=False):
            dtype = torch.float64 if v.dtype == torch.float64 else torch.float32
            v = v.to(dtype)
            coords = geometry_coordinates(points.to(dtype), v).masked_fill(~mask.transpose(1, 2), 0)
            pair = pair_features(v, v, self.scale_invariant_pairs) if self.score == "linear" else None
            harmonics = rest_harmonics(v, mask) if self.score == "harmonic" else None
        padding = ~mask[:, 0]
        # Preserve the baseline's finite wrap padding for its input BatchNorm.
        x = x.masked_fill(~mask & ~torch.isfinite(x), 0)
        x = self.embed(x).masked_fill(padding.unsqueeze(-1), 0)
        geometry = dict(coords=coords, pair=pair, harmonics=harmonics)
        for block in self.blocks:
            x = block(x, padding_mask=padding, attention_kwargs=geometry)
        if self.class_at_origin:
            cls = self.cls_token.expand(len(x), -1, -1)
            class_pair = None
            if self.score == "linear":
                origin = origin_vector(v, coords, mask)
                class_pair = pair_features(origin, torch.cat((origin, v), -1), self.scale_invariant_pairs)
            geometry = dict(coords=F.pad(coords, (0, 0, 1, 0)), pair=class_pair,
                            harmonics=None if harmonics is None else F.pad(harmonics, (0, 0, 1, 0)))
            for block in self.cls_blocks:
                cls = block(x, x_cls=cls, padding_mask=padding, attention_kwargs=geometry)
            embedding = self.norm(cls[:, 0])
        else:
            embedding = self._forward_aggregator(x, padding)
        if self.fc is None:
            return embedding
        output = self.fc(embedding)
        if self.for_inference:
            output = output.softmax(dim=1)
        return torch.cat((output, embedding), dim=1) if export_embed else output
