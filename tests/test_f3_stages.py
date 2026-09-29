"""Stage scheduling, backward compatibility, and gradients through fresh F3 computations.

Run with PYTHONPATH=src python -m pytest tests/test_f3_stages.py.
Set AIO_TEST_DEVICE=cuda to exercise the target GPU; no dataset or Weaver is required.
"""
import copy
import os
import shlex
import subprocess
from pathlib import Path
from types import MethodType
from unittest.mock import patch

import pytest
import torch

from aio.models.jets.particle_transformer import ParticleTransformer
from aio.models.jets.particle_transformer_aio import AIOParticleTransformer


DEVICE = torch.device(os.environ.get("AIO_TEST_DEVICE", "cpu"))
torch.set_num_threads(1)


@pytest.fixture(scope="module", autouse=True)
def cuda_reference_precision():
    if DEVICE.type != "cuda":
        yield
        return
    matmul, conv = torch.backends.cuda.matmul, torch.backends.cudnn.conv
    old_matmul, old_conv = matmul.fp32_precision, conv.fp32_precision
    old_benchmark = torch.backends.cudnn.benchmark
    try:
        matmul.fp32_precision = conv.fp32_precision = "ieee"
        torch.backends.cudnn.benchmark = False
        yield
    finally:
        matmul.fp32_precision, conv.fp32_precision = old_matmul, old_conv
        torch.backends.cudnn.benchmark = old_benchmark


def config():
    return dict(input_dim=4, num_classes=3, embed_dims=(16,), pair_embed_dims=(12, 12, 12),
                num_heads=2, num_layers=8, num_cls_layers=1, fc_params=[], trim=False)


def model(**kwargs):
    torch.manual_seed(731)
    mod = AIOParticleTransformer(**config(), D=8, H3=2, d_r=4, anchor_chunk=7,
                                jet_group_size=2, **kwargs).to(DEVICE)
    mod.require_calibration = False
    with torch.no_grad():
        mod.hob.motif.U.weight.normal_(std=0.1)
    return mod


def batch():
    torch.manual_seed(732)
    mask = (torch.arange(6, device=DEVICE)[None] < torch.tensor([6, 4], device=DEVICE)[:, None])[:, None]
    x = torch.randn(2, 4, 6, device=DEVICE).masked_fill(~mask, 0)
    p = torch.randn(2, 3, 6, device=DEVICE)
    v = torch.cat([p, (p.square().sum(1, keepdim=True) + 1).sqrt()], dim=1).masked_fill(~mask, 0)
    return x, v, mask


def original_encoder(self, x, v=None, mask=None, uu=None, uu_idx=None):
    """The pre-patch single-stage loop, independent of the new scheduling logic."""
    x, padding_mask, e_pair, attn_mask = self._prelude(x, v, mask, uu, uu_idx)
    msg_fn = None
    for li, block in enumerate(self.blocks):
        x = block(x, padding_mask=padding_mask, attn_mask=attn_mask, msg_fn=msg_fn)
        if self.use_f3 and li + 1 == self.f3_after:
            B, N = padding_mask.shape
            delta, anc, _ = self.hob(x, e_pair, ~padding_mask, self.rho, self.anc, self.gate, self.cand_mask)
            if anc.shape[0]:
                attn_mask = attn_mask + self.inj.bias(delta, anc, B, N)
                pairs = self.inj.dense_pairs(delta, anc, B, N)
                msg_fn = lambda a, r=pairs: self.inj.message_dense(a, r)
    return x, padding_mask


@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("ckpt", [False, True])
def test_default_matches_original_outputs_and_gradients(bf16, ckpt):
    actual = model(ckpt=ckpt).train()
    reference = copy.deepcopy(actual)
    reference._forward_encoder = MethodType(original_encoder, reference)
    assert actual.state_dict().keys() == reference.state_dict().keys()
    data = batch()

    def run(mod):
        torch.manual_seed(733)
        with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=bf16):
            out = mod(*data)
            loss = torch.nn.functional.cross_entropy(out.float(), torch.tensor([0, 2], device=DEVICE))
        loss.backward()
        return out.detach()

    expected, got = run(reference), run(actual)
    # CPU is exact. CUDA indexing-backward reductions may be nondeterministic even
    # for the same graph; use the existing suite's reduction policy for BF16 gradients.
    rtol, atol = ((0, 0) if DEVICE.type == "cpu" else
                  ((1e-2, 1e-3) if bf16 else (1e-5, 1e-5)))
    torch.testing.assert_close(got, expected, rtol=rtol, atol=atol)
    groups = {key: ([], []) for key in ("host", "hob", "inj")}
    for (name, p), (old_name, old_p) in zip(actual.named_parameters(), reference.named_parameters()):
        assert name == old_name
        if p.grad is None or old_p.grad is None:
            assert p.grad is None and old_p.grad is None
        elif DEVICE.type == "cuda" and bf16:
            group = name.split(".")[0] if name.startswith(("hob.", "inj.")) else "host"
            groups[group][0].append(p.grad.flatten().float())
            groups[group][1].append(old_p.grad.flatten().float())
        else:
            torch.testing.assert_close(p.grad, old_p.grad, rtol=rtol, atol=atol, msg=name)
    for name, (new, old) in groups.items():
        if new:
            new, old = torch.cat(new), torch.cat(old)
            error = new - old
            assert torch.isfinite(new).all() and torch.isfinite(old).all(), name
            assert error.norm() <= 0.05 * old.norm() + 1e-6, name
            assert error.abs().max() <= 0.05 * old.abs().max() + 1e-6, name


def observe(mod, *, empty_last=False):
    """Observe actual host masks/messages and F3 inputs; controlled biases expose accidental accumulation."""
    block_inputs, block_outputs, f3_inputs, message_pairs = {}, {}, [], []
    handles = []
    for index, block in enumerate(mod.blocks, 1):
        def pre(module, args, kwargs, index=index):
            block_inputs[index] = (kwargs["attn_mask"].detach().clone(), kwargs.get("msg_fn") is not None)
        def post(module, args, output, index=index):
            block_outputs[index] = output.detach().clone()
        handles.extend([block.register_forward_pre_hook(pre, with_kwargs=True), block.register_forward_hook(post)])
    original_hob = mod.hob.forward
    original_message = mod.inj.message_dense

    def hob(h, e, *args):
        f3_inputs.append((h.detach().clone(), e.detach().clone()))
        delta, anc, aux = original_hob(h, e, *args)
        if empty_last and len(f3_inputs) == 2:
            return delta[:0], anc[:0], aux
        return delta, anc, aux

    def bias(delta, anc, B, N):
        return delta.new_full((B, mod.inj.H, N, N), 0.1 * len(f3_inputs))

    def message(a, pairs):
        message_pairs.append(pairs)
        return original_message(a, pairs)

    try:
        with torch.no_grad(), patch.object(mod.hob, "forward", side_effect=hob), \
                patch.object(mod.inj, "bias", side_effect=bias), \
                patch.object(mod.inj, "message_dense", side_effect=message):
            mod(*batch())
    finally:
        for handle in handles:
            handle.remove()
    return block_inputs, block_outputs, f3_inputs, message_pairs


def test_one_injection_restores_both_channels_and_keeps_block4_source():
    mod = model(f3_inject_blocks=[5]).eval()
    seen, outputs, sources, pairs = observe(mod)
    base = seen[1][0]
    assert len(sources) == len(pairs) == 1
    torch.testing.assert_close(sources[0][0], outputs[4], rtol=0, atol=0)
    for index, (mask, has_message) in seen.items():
        assert has_message == (index == 5)
        torch.testing.assert_close(mask, base + 0.1 if index == 5 else base, rtol=0, atol=0)


def test_refresh_uses_current_nodes_static_pairs_and_replaces_cached_bias():
    mod = model(f3_refresh_after=[6]).eval()
    mod.record_last = True
    seen, outputs, sources, pairs = observe(mod)
    assert len(sources) == 2 and len(pairs) == 4
    torch.testing.assert_close(sources[0][0], outputs[4], rtol=0, atol=0)
    torch.testing.assert_close(sources[1][0], outputs[6], rtol=0, atol=0)
    assert not torch.equal(sources[0][0], sources[1][0])
    torch.testing.assert_close(sources[0][1], sources[1][1], rtol=0, atol=0)
    assert pairs[0] is pairs[1] and pairs[2] is pairs[3] and pairs[0] is not pairs[2]
    base = seen[1][0]
    for index, (mask, has_message) in seen.items():
        assert has_message == (index >= 5)
        offset = 0.0 if index <= 4 else (0.1 if index <= 6 else 0.2)
        torch.testing.assert_close(mask, base + offset, rtol=0, atol=0)
    assert list(mod.last_stages) == [4, 6]
    assert mod.last is mod.last_stages[6]


def test_empty_refresh_clears_old_bias_and_message():
    mod = model(f3_refresh_after=[6]).eval()
    seen, _, sources, pairs = observe(mod, empty_last=True)
    assert len(sources) == 2 and len(pairs) == 2
    base = seen[1][0]
    for index in (7, 8):
        torch.testing.assert_close(seen[index][0], base, rtol=0, atol=0)
        assert not seen[index][1]


@pytest.mark.parametrize("spec,dense_matmul", [("triangle", False), ("summed_triangle", True)])
@pytest.mark.parametrize("bf16", [False, True])
def test_gradients_flow_through_both_refresh_stages(spec, dense_matmul, bf16):
    mod = model(f3_refresh_after=[6], spec=spec, dense_matmul=dense_matmul, ckpt=True).train()
    deltas = []

    def retain(module, args, output):
        output[0].retain_grad()
        deltas.append(output[0])

    handle = mod.hob.register_forward_hook(retain)
    try:
        with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=bf16):
            out = mod(*batch())
            loss = torch.nn.functional.cross_entropy(out.float(), torch.tensor([0, 2], device=DEVICE))
        loss.backward()
    finally:
        handle.remove()
    assert len(deltas) == 2
    for delta in deltas:
        assert delta.grad is not None and torch.isfinite(delta.grad).all()
        assert delta.grad.abs().max() > 0
    for parameter in (mod.hob.factors.g[-1].weight, mod.hob.factors.n[-1].weight, mod.inj.M.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().max() > 0


def test_refresh_preserves_checkpoint_keys_parameter_count_and_one_time_calibration():
    single = model()
    refreshed = model(f3_refresh_after=[6])
    refreshed.load_state_dict(single.state_dict(), strict=True)
    assert sum(p.numel() for p in single.parameters()) == sum(p.numel() for p in refreshed.parameters())
    assert single.state_dict().keys() == refreshed.state_dict().keys()
    with torch.no_grad():
        single.hob.motif.U.weight.zero_()
        refreshed.hob.motif.U.weight.zero_()
    single.calibrate(*batch())
    refreshed.calibrate(*batch())
    assert refreshed.calibrate(*batch()) == {"skipped": True}
    for key, value in single.state_dict().items():
        torch.testing.assert_close(value, refreshed.state_dict()[key], rtol=0, atol=0, msg=key)
    base = ParticleTransformer(**config()).to(DEVICE).eval()
    base.load_state_dict({k: single.state_dict()[k] for k in base.state_dict()}, strict=True)
    with torch.no_grad():
        torch.testing.assert_close(refreshed.eval()(*batch()), base(*batch()), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("kwargs", [
    {"f3_after": [4, 6]}, {"f3_after": True}, {"f3_after": 0}, {"f3_after": 8},
    {"f3_refresh_after": [6, 5]}, {"f3_refresh_after": [4]}, {"f3_refresh_after": [8]},
    {"f3_refresh_after": [6, 6]}, {"f3_refresh_after": [5.5]},
    {"f3_inject_blocks": []}, {"f3_inject_blocks": [4]}, {"f3_inject_blocks": [9]},
    {"f3_inject_blocks": [6, 5]}, {"f3_inject_blocks": [5, 5]},
    {"f3_refresh_after": [6], "f3_inject_blocks": [5]},
    {"f3_refresh_after": [6], "f3_inject_blocks": [7]},
])
def test_invalid_schedules_fail_before_training(kwargs):
    with pytest.raises(ValueError):
        model(**kwargs)


@pytest.mark.parametrize("preset,spec,inject,refresh", [
    ("attended-one", "triangle", [5], []),
    ("attended-reuse", "triangle", [5, 6, 7, 8], []),
    ("summed-one", "summed_triangle", [5], []),
    ("summed-reuse", "summed_triangle", [5, 6, 7, 8], []),
    ("attended-refresh", "triangle", [5, 6, 7, 8], [6]),
    ("summed-refresh", "summed_triangle", [5, 6, 7, 8], [6]),
])
def test_presets_pass_literal_options_and_do_not_reuse_external_comment(preset, spec, inject, refresh):
    import ast
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "AIO_DRY_RUN": "1", "SEED": "123", "EPOCHS": "80", "COMMENT": "old-run"}
    env.pop("AIO_RUN_COMMENT", None)
    proc = subprocess.run(["bash", str(root / "scripts/run_triplet_experiment.sh"), preset],
                          env=env, text=True, capture_output=True, check=True)
    args = shlex.split(proc.stdout)
    options = {args[i + 1]: ast.literal_eval(args[i + 2]) for i, arg in enumerate(args) if arg == "-o"}
    assert args[0] == "EPOCHS=80" and args[1] == f"COMMENT={preset}-b4-10pct-seed123"
    assert options["spec"] == spec and options["f3_inject_blocks"] == inject
    assert options["f3_refresh_after"] == refresh and options["f3_after"] == 4
    assert options["dense_matmul"] == (spec == "summed_triangle")
    assert options["use_router"] is False and options["message_backend"] == "auto"
