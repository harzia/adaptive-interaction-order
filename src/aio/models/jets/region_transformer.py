"""Learned-region representation-order pilot, using LLoCa-style attention.

This is an independent small backbone, not official LLoCa-ParT. Original presets
use no pair bias and mean readout. Optional reference ablations add a local-frame
pair-feature MLP and/or the repository's ParT class-attention readout.
"""
from __future__ import annotations

import torch
from torch import nn
from aio.core.geometry.attention import GeometricAttention
from aio.core.geometry.regions import LearnedRegionGeometry
from aio.core.geometry.pair_bias import LocalFramePairBias


class RegionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, scalars, vectors, tensors, ffn_dim,
                 dropout, coarse):
        super().__init__()
        self.coarse = coarse
        self.local_norm = nn.LayerNorm(hidden_dim)
        # Shared local representation across ALL arms, including the full reference.
        self.local_attention = GeometricAttention(hidden_dim, num_heads, 8, 2, 0)
        self.global_norm = nn.LayerNorm(hidden_dim)
        self.global_attention = GeometricAttention(hidden_dim, num_heads, scalars, vectors, tensors)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, ffn_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(ffn_dim, hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, geometry, mask, tensor_gate=None, pair_bias=None,
                neighbor_pair_bias=None, pair_bias_scope="both", global_gate=1.):
        args = (geometry["frames"], geometry["frame_valid"], mask)
        h = h + self.dropout(self.local_attention(
            self.local_norm(h), *args, indices=geometry["indices"],
            neighbor_mask=geometry["neighbor_mask"],
            pair_bias=pair_bias if pair_bias_scope in ("both", "local") else None,
            neighbor_pair_bias=neighbor_pair_bias))
        regional = dict(assignments=geometry["assignments"], mass=geometry["mass"],
                        region_mask=geometry["region_mask"], tensor_gate=tensor_gate) if self.coarse else {}
        global_output = self.global_attention(
            self.global_norm(h), *args,
            pair_bias=pair_bias if pair_bias_scope in ("both", "global") else None, **regional)
        h = h + self.dropout(global_output) * global_gate
        h = h + self.dropout(self.ffn(self.ffn_norm(h)))
        return h * mask[..., None]


class RegionTransformer(nn.Module):
    def __init__(self, input_dim, num_classes, *, num_regions=0, hidden_dim=64,
                 num_heads=4, num_layers=4, ffn_dim=128, scalars=8, vectors=2,
                 tensors=0, local_neighbors=8, geometry_dim=32, dropout=0.,
                 assignment_temperature=0.5, balance_weight=0.01,
                 entropy_weight=0.001, locality_weight=0.01,
                 fc_params=(), for_inference=False, pair_bias=False, pair_bias_dim=32,
                 readout="mean", readout_layers=2, region_source="learned",
                 ca_radius=1000., pair_bias_scope="both"):
        super().__init__()
        if num_classes < 2 or input_dim < 0 or num_layers < 1:
            raise ValueError("Invalid input_dim, num_classes, or num_layers")
        if min(hidden_dim, num_heads, ffn_dim, geometry_dim, local_neighbors) < 1:
            raise ValueError("Hidden sizes and neighborhood/head counts must be positive")
        if pair_bias_scope not in ("both", "local", "global"):
            raise ValueError("pair_bias_scope must be both, local, or global")
        if pair_bias and num_regions and pair_bias_scope != "local":
            raise ValueError("Dense pair bias requires full particle attention; use pair_bias_scope='local' for regions")
        if region_source not in ("learned", "ca") or (region_source == "ca" and not num_regions):
            raise ValueError("region_source must be learned or ca; ca requires positive num_regions")
        if readout not in ("mean", "class_attention"):
            raise ValueError("readout must be mean or class_attention")
        if readout == "class_attention" and (readout_layers < 1 or hidden_dim % num_heads):
            raise ValueError("Class attention requires positive layers and hidden_dim divisible by num_heads")
        self.input_dim = input_dim
        self.num_regions = num_regions
        self.region_source = region_source
        self.pair_bias_scope = pair_bias_scope
        self.for_inference = for_inference
        geometry_class, geometry_kwargs = LearnedRegionGeometry, {}
        if region_source == "ca":
            from aio.core.geometry.lund import CambridgeAachenGeometry
            geometry_class, geometry_kwargs = CambridgeAachenGeometry, {"radius": ca_radius}
        self.geometry = geometry_class(
            input_dim, num_regions, geometry_dim, local_neighbors, assignment_temperature,
            balance_weight, entropy_weight, locality_weight, **geometry_kwargs)
        self.embed = nn.Sequential(nn.Linear(input_dim + 4, hidden_dim), nn.GELU(),
                                   nn.Linear(hidden_dim, hidden_dim))
        self.blocks = nn.ModuleList([
            RegionBlock(hidden_dim, num_heads, scalars, vectors, tensors, ffn_dim,
                        dropout, coarse=num_regions > 0) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        head, width = [], hidden_dim
        for units, probability in fc_params:
            head += [nn.Linear(width, units), nn.GELU(), nn.Dropout(probability)]
            width = units
        self.head = nn.Sequential(*head, nn.Linear(width, num_classes))
        # Create optional modules after the original ones so old presets and
        # their initialization/state_dict remain unchanged under a fixed seed.
        self.pair_bias = LocalFramePairBias(num_heads, pair_bias_dim) if pair_bias else None
        self.readout = readout
        self.cls_token = None
        self.cls_blocks = None
        if readout == "class_attention":
            from aio.models.jets.particle_transformer import Block
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            self.cls_blocks = nn.ModuleList([
                Block(embed_dim=hidden_dim, num_heads=num_heads, ffn_ratio=4,
                      dropout=0., attn_dropout=0., activation_dropout=0.)
                for _ in range(readout_layers)])
        self.geometry_frozen = False
        self.last_aux_loss = None
        self.last_diagnostics = {}

    def freeze_geometry(self):
        self.geometry.requires_grad_(False)
        self.geometry_frozen = True

    def save_geometry(self, path):
        """Share this checkpoint across different representation-order models at fixed K."""
        torch.save({"format": "aio-region-geometry-v1", "spec": self.geometry.spec,
                    "state_dict": self.geometry.state_dict()}, path)

    def load_geometry(self, path, freeze=True):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") != "aio-region-geometry-v1":
            raise ValueError("Not an exported region-geometry checkpoint")
        if payload["spec"] != self.geometry.spec:
            raise ValueError(f"Geometry configuration mismatch: {payload['spec']} != {self.geometry.spec}")
        self.geometry.load_state_dict(payload["state_dict"], strict=True)
        if freeze:
            self.freeze_geometry()

    def encode(self, features, v, mask, *, references=None, tensor_gate=None,
               global_gate=1., pair_bias_scope=None):
        """Weaver tensors: features [B,F,N], v [B,4,N] in (px,py,pz,E), mask [B,1,N].

        Optional references [B,2,4] are in INTERNAL (E,px,py,pz) order. For Lorentz
        tests transform those references along with momenta and use scalar features.
        The normal JetClass-II input feature pipeline has only residual SO(2) symmetry.
        """
        if features.ndim != 3 or features.shape[1] != self.input_dim:
            raise ValueError(f"features must have shape [B,{self.input_dim},N]")
        batch, _, n = features.shape
        if n == 0 or v.shape != (batch, 4, n) or mask.shape != (batch, 1, n):
            raise ValueError("Inconsistent features, four-momenta, or mask shapes")
        valid = mask[:, 0].bool()
        if not valid.any(1).all():
            raise ValueError("Each jet must contain at least one valid particle")
        dtype = self.embed[0].weight.dtype
        x = features.transpose(1, 2).to(dtype).masked_fill(~valid[..., None], 0)
        p = v[:, [3, 0, 1, 2]].transpose(1, 2).to(dtype).masked_fill(~valid[..., None], 0)
        geometry = self.geometry(x, p, valid, references=references)
        self.last_aux_loss = geometry["regularization"]
        self.last_diagnostics = geometry["diagnostics"]
        with torch.autocast(device_type=x.device.type, enabled=False):
            local_p = torch.einsum("bnij,bnj->bni", geometry["frames"], p)
            local_p = local_p * geometry["frame_valid"][..., None]
            local_p = local_p.sign() * local_p.abs().log1p()
        h = self.embed(torch.cat((x, local_p), -1)) * valid[..., None]
        scope = self.pair_bias_scope if pair_bias_scope is None else pair_bias_scope
        if scope not in ("both", "local", "global", "none"):
            raise ValueError("Invalid pair_bias_scope override")
        if self.pair_bias is not None and self.num_regions and scope not in ("local", "none"):
            raise ValueError("Regional models support local pair bias only")
        bias, neighbor_bias = None, None
        if self.pair_bias is not None and scope != "none":
            if scope == "local":
                neighbor_bias = self.pair_bias(
                    p, geometry["frames"], geometry["frame_valid"], valid,
                    indices=geometry["indices"], neighbor_mask=geometry["neighbor_mask"])
            else:
                bias = self.pair_bias(p, geometry["frames"], geometry["frame_valid"], valid)
        for block in self.blocks:
            h = block(h, geometry, valid, tensor_gate=tensor_gate, pair_bias=bias,
                      neighbor_pair_bias=neighbor_bias, pair_bias_scope=scope, global_gate=global_gate)
        if self.cls_blocks is None:
            embedding = (self.norm(h) * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)
        else:
            cls = self.cls_token.expand(batch, -1, -1)
            for block in self.cls_blocks:
                cls = block(h, x_cls=cls, padding_mask=~valid)
            embedding = self.norm(cls[:, 0])
        return embedding, geometry

    def forward(self, features, v, mask, *, references=None, tensor_gate=None,
                return_aux=False, global_gate=1., pair_bias_scope=None):
        embedding, geometry = self.encode(features, v, mask, references=references,
                                          tensor_gate=tensor_gate, global_gate=global_gate,
                                          pair_bias_scope=pair_bias_scope)
        logits = self.head(embedding)
        if self.for_inference:
            logits = logits.softmax(-1)
        return (logits, geometry) if return_aux else logits
