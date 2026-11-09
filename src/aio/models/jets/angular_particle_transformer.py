"""ParT with persistent scalar/dipole/quadrupole value channels.

Unchanged particle Q/K, pair embedding, FFNs, and class blocks. Typed values are
concatenated into the SAME attention operation, then contracted before the FFN.
No regions, token compression, learned frames, or router are introduced here.
"""
from __future__ import annotations

import math
import torch
from torch import nn

from .particle_transformer import ParticleTransformer, build_sparse_tensor
from aio.core.geometry.angular_moments import AngularEmbedding, AngularUpdate, jet_angular_geometry


class AngularParticleTransformer(ParticleTransformer):
    def __init__(self, input_dim, num_classes=None, *, moment_order=2, moment_channels=2,
                 moment_product_degree=2, moment_persistent=True, moment_scale=1.,
                 moment_frame="rest", rest_mass_fraction_min=1e-6, **kwargs):
        if moment_order not in (1, 2) or moment_product_degree not in (1, 2) or moment_channels < 1:
            raise ValueError("Invalid moment order, product degree, or channels")
        if not math.isfinite(moment_scale) or moment_frame not in ("rest", "detector") or not 0 < rest_mass_fraction_min < 1:
            raise ValueError("Invalid moment scale/frame/rest threshold")
        if kwargs.get("version", 1) != 1 or kwargs.get("include_global_token", False) or kwargs.get("for_segmentation", False):
            raise ValueError("Initial angular experiments require ParT v1 particle blocks and class readout")
        if kwargs.get("compile_model", False):
            raise ValueError("Angular experiments currently use eager execution; compile_model is unsupported")
        block = kwargs.get("block_params") or {}
        if block.get("headwise_attn_output_gate", False) or block.get("elementwise_attn_output_gate", False):
            raise ValueError("Additional attention-output gating is outside the matched angular experiment")
        super().__init__(input_dim, num_classes, **kwargs)
        if not len(self.blocks) or self.cls_blocks is None:
            raise ValueError("At least one particle block and class-attention readout required")
        dim, heads = self.blocks[0].embed_dim, self.blocks[0].num_heads
        self.moment_order, self.moment_channels = moment_order, moment_channels
        self.moment_product_degree, self.moment_persistent = moment_product_degree, moment_persistent
        self.moment_scale, self.moment_frame = float(moment_scale), moment_frame
        self.rest_mass_fraction_min = rest_mass_fraction_min
        # Append after super initialization: common ParT parameters keep exactly
        # the baseline initialization at the same random seed.
        self.moment_embedding = AngularEmbedding(dim, heads, moment_channels, moment_order)
        self.moment_layers = nn.ModuleList([
            AngularUpdate(dim, heads, moment_channels, moment_order, moment_product_degree,
                          carry=moment_persistent and i < len(self.blocks) - 1)
            for i in range(len(self.blocks))])
        self.last_diagnostics = {}

    def load_part_backbone(self, state):
        """Explicit warm-start helper. Reject missing/mismatched baseline keys.

        Pass the unwrapped ordinary ParT state dict, not a tensor-model checkpoint.
        This does not freeze the baseline and is not the default training recipe.
        """
        common = {k for k in self.state_dict() if not k.startswith(("moment_embedding.", "moment_layers."))}
        if set(state) != common:
            raise ValueError("Expected exactly the matching ordinary ParT state_dict keys")
        merged = self.state_dict()
        merged.update(state)
        self.load_state_dict(merged, strict=True)

    def _forward_encoder(self, x, v=None, mask=None, uu=None, uu_idx=None):
        if self.moment_scale == 0:
            # Exact baseline execution path, including its attention kernel/RNG.
            self.last_diagnostics = {}
            return super()._forward_encoder(x, v=v, mask=mask, uu=uu, uu_idx=uu_idx)
        if v is None:
            raise ValueError("Angular attention requires particle four-momenta")
        if mask is None:
            mask = torch.ones_like(x[:, :1], dtype=torch.bool)
        if x.ndim != 3 or v.shape != (len(x), 4, x.shape[-1]) or mask.shape != (len(x), 1, x.shape[-1]):
            raise ValueError("Inconsistent particle features, momenta, or mask")
        mask = mask.bool()
        # Keep finite wrap-padding exactly as in ordinary ParT: replacing it by
        # zeros would change input BatchNorm statistics during training. Only
        # non-finite padded entries are sanitized; geometry masks all padding.
        x = x.masked_fill(~mask & ~torch.isfinite(x), 0)
        v = v.masked_fill(~mask & ~torch.isfinite(v), 0)
        with torch.set_grad_enabled(x.requires_grad or v.requires_grad):
            if not self.for_inference and uu_idx is not None:
                uu = build_sparse_tensor(uu, uu_idx, x.size(-1))
            x, v, mask, uu = self.trimmer(x, v, mask, uu)
        valid = mask[:, 0]
        padding_mask = ~valid
        geometry = jet_angular_geometry(v, mask, frame=self.moment_frame,
                                        rest_mass_fraction_min=self.rest_mass_fraction_min)
        x = self.embed(x).masked_fill(~valid[..., None], 0)
        attn_mask = self.pair_embed(v, uu=uu, mask=mask) if self.pair_embed is not None else None
        state = self.moment_embedding(x, geometry, valid)
        for idx, (block, angular) in enumerate(zip(self.blocks, self.moment_layers)):
            if idx and not self.moment_persistent:
                state = self.moment_embedding(x, geometry, valid)
            next_state = None

            def update_scalar(hidden, messages):
                nonlocal next_state
                delta, next_state = angular.update(hidden, messages, state, geometry, valid)
                return hidden + self.moment_scale * delta

            x = block(x, padding_mask=padding_mask,
                      attn_mask=attn_mask if self.block_ids_with_attn_mask[idx] else None,
                      extra_values=angular.values(state), extra_update=update_scalar)
            if next_state is not None:
                state = next_state
        self.last_diagnostics = dict(rest_frame_fraction=geometry["used_rest"].float().mean().detach(),
            zero_direction_fraction=(valid & ~geometry["direction_valid"]).sum().float().div(valid.sum()).detach())
        return x, padding_mask
