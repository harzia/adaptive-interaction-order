"""ParT adapter (spec §6).  No monkeypatching: particle blocks are converted in place to AIOBlock /
AIOAttention (instance-level `__class__` swap), class blocks and any other ParticleTransformer in the
process are untouched.  AIOBlock.forward mirrors upstream Block.forward exactly, including the shared
tail (head scaling, post-attention norm, dropout, drop-path, residual) for BOTH branches.

Names assumed from upstream ParT: trimmer, embed, pair_embed, blocks, cls_blocks, cls_token, norm, fc,
mask convention True = padded inside Block.  Verify with `parity_check` in a process that has never
imported this module's model, using identical weights and inputs.
"""
from __future__ import annotations
import torch, torch.nn.functional as F

from aio.core import HigherOrderBlock, Injection, TRIANGLE, SUMMED_TRIANGLE, TAIL, attend_with_message
from aio.models.jets.particle_transformer import (ParticleTransformer, PairEmbed, Attention, Block,
                                                  _canonical_mask, _none_or_dtype, build_sparse_tensor)


# ---------------------------------------------------------------- pair-feature tap
class PairEmbedTap(PairEmbed):
    """`self.embed` ends with [Conv1d(64->H), BatchNorm1d]; the 64-d post-GELU hidden is `embed[:-2]`.
    Returns cat([hidden, head_bias]) so the existing dense/sparse scatter logic writes both."""

    @classmethod
    def convert(cls, pe):
        pe.__class__ = cls
        pe.hidden_dim, pe.head_dim_out = pe.embed[-2].in_channels, pe.out_dim
        pe.out_dim = pe.hidden_dim + pe.head_dim_out
        return pe

    def _embed_pairs(self, x, uu):
        hid = self.embed[:-2](x); out = self.embed[-2:](hid)
        if uu is not None:
            fh = self.fts_embed[:-2](uu); hid, out = hid + fh, out + self.fts_embed[-2:](fh)
        return torch.cat([hid, out], 1)

    def split(self, y):                             # (B,64+H,P,P) -> e_pair (B,P,P,64), bias (B,H,P,P)
        return y[:, :self.hidden_dim].permute(0, 2, 3, 1).contiguous(), y[:, self.hidden_dim:]


# ---------------------------------------------------------------- attention / block
class AIOAttention(Attention):
    force_explicit = False          # set True to run the explicit path even without a message (parity checks)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None, msg_fn=None):
        bsz, tgt_len, _ = query.shape; src_len = key.shape[1]
        kpm = _canonical_mask(key_padding_mask, "key_padding_mask", _none_or_dtype(attn_mask), "attn_mask", query.dtype)
        am = _canonical_mask(attn_mask, "attn_mask", None, "", query.dtype, check_other=False)
        if kpm is not None:
            kpm = kpm.view(bsz, 1, 1, src_len).expand(-1, self.num_heads, -1, -1)
            am = kpm if am is None else am + kpm
        q, k, v = F._in_projection_packed(query, key, value, self.in_proj.weight, self.in_proj.bias)
        q = self.q_norm(q.view(bsz, tgt_len, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(k.view(bsz, src_len, self.num_heads, self.head_dim)).transpose(1, 2)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        p = self.dropout if self.training else 0.0
        if msg_fn is None and self.use_sdpa and not self.force_explicit:
            out = F.scaled_dot_product_attention(q, k, v, am, p)                 # unchanged host path
        else:
            out = attend_with_message(q, k, v, am, msg_fn, p, self.training)  # AIO path
        out = out.transpose(1, 2).contiguous()
        if self.headwise_attn_output_gate:
            out = out * torch.sigmoid(self.gate_proj(query)).reshape(bsz, tgt_len, self.num_heads, 1)
        elif self.elementwise_attn_output_gate:
            out = out * torch.sigmoid(self.gate_proj(query)).reshape(bsz, tgt_len, self.num_heads, self.head_dim)
        return self.out_proj(out.reshape(bsz, tgt_len, self.embed_dim)), None


class AIOBlock(Block):
    def forward(self, x, x_cls=None, padding_mask=None, attn_mask=None, msg_fn=None):
        if x_cls is not None:                                                   # class attention (upstream)
            with torch.no_grad():
                padding_mask = torch.cat((torch.zeros_like(padding_mask[:, :1]), padding_mask), dim=1)
            residual = x_cls
            u = self.pre_attn_norm(torch.cat((x_cls, x), dim=1))
            x = self.attn(x_cls, u, u, key_padding_mask=padding_mask)[0]
        else:
            if self.c_mask is not None and attn_mask is not None: attn_mask = torch.mul(self.c_mask, attn_mask)
            residual = x
            x = self.pre_attn_norm(x)
            x = self.attn(x, x, x, key_padding_mask=padding_mask, attn_mask=attn_mask, msg_fn=msg_fn)[0]
        # --- shared tail, both branches (upstream) ---
        if self.c_attn is not None:
            bsz, tgt_len, _ = x.size()
            x = (x.view(bsz, tgt_len, self.num_heads, self.head_dim) * self.c_attn.view(1, 1, self.num_heads, 1)).reshape(bsz, tgt_len, self.embed_dim)
        x = self.post_attn_norm(x); x = self.dropout(x); x = self.drop_path1(self.ls1(x)); x = x + residual
        residual = x; x = self.pre_fc_norm(x)
        if self.fc1_g is None: x = self.act(self.fc1(x))
        else: x = self.act(self.fc1_g(x)) * self.fc1(x)
        x = self.act_dropout(x); x = self.post_fc_norm(x); x = self.fc2(x); x = self.dropout(x)
        x = self.drop_path2(self.ls2(x))
        if self.w_resid is not None: residual = torch.mul(self.w_resid, residual)
        return x + residual


# ---------------------------------------------------------------- model
SPECS = {"triangle": TRIANGLE, "summed_triangle": SUMMED_TRIANGLE, "tail": TAIL}


class AIOParticleTransformer(ParticleTransformer):
    """ParT + one higher-order stage: Δr from the output of particle block `f3_after` (1-based), injected
    into blocks f3_after+1..L.  Particle blocks are converted in place; class blocks untouched.

    Only `_forward_encoder` is overridden, so the inherited `forward` and any embed-export path that calls
    `_forward_encoder` + `_forward_aggregator` both go through the stage.  Runtime knobs are attributes
    (set them on the module; the Weaver wrapper's forward signature is unchanged):
        use_f3 (True)   rho (None = dense)   anc / gate / cand_mask (None)   — see spec §2.6, §3.1, §5.
    `spec` may be a tuple of PoolSpec or one of the names in SPECS (Weaver -o options are strings)."""

    def __init__(self, *a, f3_after=4, spec="triangle", D=32, H3=2, d_r=32, anchor_chunk=256, dense_matmul=False,
                 use_router=False, diagnostics=False, **kw):
        super().__init__(*a, **kw)
        spec = SPECS[spec] if isinstance(spec, str) else spec
        tap = PairEmbedTap.convert(self.pair_embed)
        for blk in self.blocks:
            blk.__class__, blk.attn.__class__ = AIOBlock, AIOAttention
        d, H = self.blocks[0].embed_dim, self.blocks[0].num_heads
        self.f3_after = f3_after
        self.hob = HigherOrderBlock(d, tap.hidden_dim, spec=spec, D=D, H=H3, d_r=d_r, anchor_chunk=anchor_chunk,
                                    dense_matmul=dense_matmul, use_router=use_router, diagnostics=diagnostics)
        self.inj = Injection(d_r, H, d)
        self.use_f3, self.rho, self.anc, self.gate, self.cand_mask = True, None, None, None, None
        self.last = None

    def load_baseline(self, state_dict):
        """Warm start from a completed baseline checkpoint (§4.4 continued-training control only — not needed
        when training from scratch).  The only keys allowed to be missing are the new modules'."""
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        bad = [k for k in missing if not (k.startswith("hob.") or k.startswith("inj."))]
        assert not bad and not unexpected, f"checkpoint mismatch: missing={bad} unexpected={unexpected}"
        return missing

    def _prelude(self, x, v, mask, uu, uu_idx):
        with torch.no_grad():
            if not self.for_inference and uu_idx is not None:
                uu = build_sparse_tensor(uu, uu_idx, x.size(-1))
            x, v, mask, uu = self.trimmer(x, v, mask, uu)                    # hooks AFTER the trimmer
            padding_mask = ~mask.squeeze(1)                                   # (B,P): True = padded
        x = self.embed(x).masked_fill(padding_mask[..., None], 0)            # (B,P,C)
        e_pair, attn_mask = self.pair_embed.split(self.pair_embed(v, uu, mask=mask))   # AIO tap
        return x, padding_mask, e_pair, attn_mask

    def _forward_encoder(self, x, v=None, mask=None, uu=None, uu_idx=None):
        """Mirror of upstream _forward_encoder; the lines marked AIO are the additions.  Returns (x, padding_mask)."""
        x, padding_mask, e_pair, attn_mask = self._prelude(x, v, mask, uu, uu_idx)
        msg_fn, self.last = None, None
        for li, block in enumerate(self.blocks):
            x = block(x, padding_mask=padding_mask, attn_mask=attn_mask, msg_fn=msg_fn)
            if self.use_f3 and li + 1 == self.f3_after:                   # AIO: Δr from this block's output
                B, N = padding_mask.shape
                delta_a, anc_sel, aux = self.hob(x, e_pair, ~padding_mask, self.rho, self.anc, self.gate, self.cand_mask)
                self.last = {"delta_a": delta_a, "anc": anc_sel, **aux}
                if anc_sel.shape[0]:                                      # zero-selection: host untouched
                    attn_mask = attn_mask + self.inj.bias(delta_a, anc_sel, B, N)
                    msg_fn = lambda a, d=delta_a, s=anc_sel: self.inj.message(a, d, s)
        return x, padding_mask

    @torch.no_grad()
    def calibrate(self, x, v=None, mask=None, uu=None, uu_idx=None, target=1.0, force=False):
        """Run once at init (per seed), on a real batch, in fp32: block-4 states and post-trimmer pair features go
        to HigherOrderBlock.calibrate.  Sets hob.calibrated (saved in the state dict; a second call is a no-op)."""
        was_training = self.training; self.eval()
        with torch.autocast(device_type="cuda", enabled=False):
            x, padding_mask, e_pair, attn_mask = self._prelude(x.float(), None if v is None else v.float(), mask, uu, uu_idx)
            for li, block in enumerate(self.blocks):
                x = block(x, padding_mask=padding_mask, attn_mask=attn_mask)
                if li + 1 == self.f3_after: break
            rep = self.hob.calibrate(x.float(), e_pair.float(), ~padding_mask, target=target, force=force)
        self.train(was_training)
        return rep


@torch.no_grad()
def parity_check(aio: AIOParticleTransformer, base: ParticleTransformer, batch, rtol=None, atol=None):
    """Three checks, with the contract each one can actually promise.

      identity   AIO(use_f3=False) vs AIO(use_f3=True), both on the explicit-attention path: same code,
                 zero branch -> expected bitwise equal (atol 0).  Requires U3 == 0.
      adapter    AIO(use_f3=False) vs upstream baseline, both on SDPA: checks the mirrored forward.
                 Expected equal or at roundoff.
      deployment AIO(use_f3=True, explicit path in blocks 5-8) vs baseline (SDPA): differs by backend
                 roundoff only; compared with tolerances (SDPA kernels are documented not to be bitwise).

    Both models must already hold the same weights — trained baseline or shared random init; this helper
    verifies that instead of copying.  `batch` is the positional input tuple your forward takes.  Runs in eval
    mode with the trimmer deterministic."""
    dt = next(aio.parameters()).dtype
    rtol = rtol or (1e-2 if dt in (torch.float16, torch.bfloat16) else 1e-5)
    atol = atol or (1e-2 if dt in (torch.float16, torch.bfloat16) else 1e-6)
    assert float(aio.hob.motif.U.weight.abs().max()) == 0.0, "U3 is not zero; identity check is meaningless"
    bs, as_ = base.state_dict(), aio.state_dict()
    for k in bs: assert torch.equal(bs[k], as_[k]), f"weights differ: {k}"
    aio.eval(); base.eval()
    per_block = {"aio": [], "base": []}
    hooks = [blk.register_forward_hook(lambda m, i, o, key=key: per_block[key].append(o.detach()))
             for key, blocks in (("aio", aio.blocks), ("base", base.blocks)) for blk in blocks]
    def run(model, use_f3=None):
        key = "aio" if model is aio else "base"; per_block[key].clear()
        if use_f3 is not None: aio.use_f3 = use_f3
        y = model(*batch)
        return y.detach(), list(per_block[key])
    for blk in aio.blocks: blk.attn.force_explicit = True
    y_off_x, b_off_x = run(aio, use_f3=False); y_on_x, b_on_x = run(aio, use_f3=True)
    for blk in aio.blocks: blk.attn.force_explicit = False
    y_off, b_off = run(aio, use_f3=False); y_on, b_on = run(aio, use_f3=True); y_b, b_b = run(base)
    aio.use_f3 = True
    for h in hooks: h.remove()
    mx = lambda a, b: float((a - b).abs().max())
    rep = {"identity_logits": mx(y_on_x, y_off_x), "identity_blocks": [mx(a, b) for a, b in zip(b_on_x, b_off_x)],
           "adapter_logits": mx(y_off, y_b), "adapter_blocks": [mx(a, b) for a, b in zip(b_off, b_b)],
           "deployment_logits": mx(y_on, y_b), "deployment_blocks": [mx(a, b) for a, b in zip(b_on, b_b)]}
    rep["identity_pass"] = rep["identity_logits"] == 0.0 and max(rep["identity_blocks"]) == 0.0
    rep["adapter_pass"] = torch.allclose(y_off, y_b, rtol=rtol, atol=atol)
    rep["deployment_pass"] = torch.allclose(y_on, y_b, rtol=rtol, atol=atol)
    return rep