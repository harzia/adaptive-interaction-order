"""Relative particle attention in jet-axis (delta eta, delta phi, log pt fraction)."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def frequency_ladder(heads, *, bank="full", values=False, directions=3,
                     angular=True, momentum=True):
    """Return initial frequencies [heads, planes, 3], in radians per coordinate."""
    if bank not in ("small", "full") or directions < 1:
        raise ValueError("Expected a full or small bank and positive direction count")
    rows = []
    for h in range(heads):
        if bank == "small":
            # Slide three adjacent octaves across heads, including both ends.
            start_a = round(5 * h / max(1, heads - 1))
            start_t = round(3 * h / max(1, heads - 1))
            a = [start_a + 1] if values else range(start_a, start_a + 3)
            t = range(start_t, start_t + 3)
        else:
            step = 2 if values else 1
            a, t = range(0, 8, step), range(0, 6, step)
        row = []
        if angular:
            for k in a:
                w = math.tau * 2**k / 3.2
                for d in range(directions):
                    alpha = math.pi * (d + h / heads) / directions
                    row.append([w * math.cos(alpha), w * math.sin(alpha), 0.])
        if momentum:
            row.extend([0., 0., math.tau * 2**k / 16] for k in t)
        rows.append(row)
    return torch.tensor(rows, dtype=torch.float32).reshape(heads, -1, 3)


def rotate_planes(x, cos, sin):
    a, b = x.unflatten(-1, (-1, 2)).unbind(-1)
    return torch.stack((cos * a - sin * b, sin * a + cos * b), -1).flatten(-2)


class RelativeAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, *,
                 score="linear", rotate_values=True, bank="full", value_bank=None,
                 directions=3, angular=True, momentum=True, freeze_frequencies=False,
                 scale_invariant_pairs=False, control_rank=8, use_qk_norm=False,
                 headwise_attn_output_gate=False, elementwise_attn_output_gate=False,
                 norm_layer=None):
        super().__init__()
        if use_qk_norm or headwise_attn_output_gate or elementwise_attn_output_gate:
            raise ValueError("Relative attention expects the baseline ParT v1 attention options")
        if score not in ("linear", "rotary", "harmonic", "low_rank"):
            raise ValueError(f"Unknown score mode {score!r}")
        if scale_invariant_pairs and score != "linear":
            raise ValueError("Scale-invariant pair subset requires linear pair scores")
        if not (angular or momentum) or control_rank < 1:
            raise ValueError("Select a frequency family and a positive control rank")
        self.heads = num_heads
        self.content = embed_dim // num_heads
        self.score = score
        self.dropout = dropout
        self.scale_invariant_pairs = scale_invariant_pairs
        # Keep disabled coordinate families absent even after frequency updates.
        self.axes = slice(None) if angular and momentum else (slice(0, 2) if angular else slice(2, 3))
        def make_bank(name, enabled, which, values):
            if not enabled:
                return None
            w = frequency_ladder(num_heads, bank=which, values=values, directions=directions,
                                 angular=angular, momentum=momentum)[..., self.axes].contiguous()
            if freeze_frequencies:
                self.register_buffer(name, w)
            else:
                self.register_parameter(name, nn.Parameter(w))
            return w
        ws = make_bank("omega_score", score == "rotary", bank, False)
        wv = make_bank("omega_value", rotate_values, value_bank or bank, True)
        self.score_width = self.content + (2 * ws.size(1) if ws is not None else 0)
        self.value_width = self.content + (2 * wv.size(1) if wv is not None else 0)
        self.q_proj = nn.Linear(embed_dim, num_heads * self.score_width, bias=bias)
        self.k_proj = nn.Linear(embed_dim, num_heads * self.score_width, bias=bias)
        self.v_proj = nn.Linear(embed_dim, num_heads * self.value_width, bias=bias)
        self.out_proj = nn.Linear(num_heads * self.value_width, embed_dim, bias=bias)
        if score == "linear":
            self.pair_proj = nn.Linear(2 if scale_invariant_pairs else 4, num_heads)
        elif score == "rotary":
            self.gain = nn.Parameter(torch.ones(num_heads, 1, 1))
        elif score == "harmonic":
            # Independent weights for the dipole and quadrupole inner products.
            self.degree_gain = nn.Parameter(torch.ones(num_heads, 2))
        else:
            self.geom_proj = nn.Sequential(nn.Linear(3, 32), nn.GELU(),
                                          nn.Linear(32, 2 * num_heads * control_rank))

    def _phases(self, coords, omega, dtype):
        # A bf16 phase at the fine octaves can be wrong by an entire radian.
        with torch.autocast(device_type=coords.device.type, enabled=False):
            theta = torch.einsum("bnc,hmc->bhnm", coords[..., self.axes].to(omega), omega)
            return theta.cos().to(dtype), theta.sin().to(dtype)

    def forward(self, query, key, value, *, coords, key_padding_mask, pair=None,
                harmonics=None, attn_mask=None):
        def project(layer, x):
            return layer(x).unflatten(-1, (self.heads, -1)).transpose(1, 2)
        q, k, v = project(self.q_proj, query), project(self.k_proj, key), project(self.v_proj, value)
        nq = q.size(2)  # class attention uses the first (origin) coordinate as query
        mask = ~key_padding_mask[:, None, None, :]
        if self.score == "linear":
            bias = self.pair_proj(pair).permute(0, 3, 1, 2).to(q.dtype)
            mask = bias.masked_fill(~mask, float("-inf"))
        elif self.score == "rotary":
            c, s = self._phases(coords, self.omega_score, q.dtype)
            qp = rotate_planes(q[..., self.content:], c[:, :, :nq], s[:, :, :nq])
            kp = rotate_planes(k[..., self.content:], c, s)
            q = torch.cat((q[..., :self.content], self.gain.to(q.dtype) * qp), -1)
            k = torch.cat((k[..., :self.content], kp), -1)
        elif self.score == "harmonic":
            y = harmonics.to(q.dtype)[:, None].expand(-1, self.heads, -1, -1)
            g1, g2 = self.degree_gain.to(q.dtype).unbind(-1)
            yq = torch.cat((y[..., :3] * g1[:, None, None], y[..., 3:] * g2[:, None, None]), -1)
            q, k = torch.cat((q, yq[:, :, :nq]), -1), torch.cat((k, y), -1)
        else:
            geom = self.geom_proj(coords.to(query.dtype)).unflatten(-1, (self.heads, -1)).transpose(1, 2)
            gq, gk = geom.chunk(2, -1)
            q, k = torch.cat((q, gq[:, :, :nq]), -1), torch.cat((k, gk), -1)
        if self.value_width > self.content:
            c, s = self._phases(coords, self.omega_value, v.dtype)
            v = torch.cat((v[..., :self.content], rotate_planes(v[..., self.content:], c, s)), -1)
        u = F.scaled_dot_product_attention(q, k, v, mask, self.dropout if self.training else 0.)
        if self.value_width > self.content:
            relative = rotate_planes(u[..., self.content:], c[:, :, :nq], -s[:, :, :nq])
            u = torch.cat((u[..., :self.content], relative), -1)
        return self.out_proj(u.transpose(1, 2).flatten(2)), None
