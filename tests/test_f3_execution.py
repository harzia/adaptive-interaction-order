"""Execution parity against the pre-optimisation chunk loop at main@36ab697c.

Run with PYTHONPATH=src python -m pytest tests/test_f3_execution.py.
F3 readouts are deliberately nonzero: identity-at-initialisation alone cannot test these paths.
Set AIO_TEST_DEVICE=cuda to repeat the same checks on the target GPU.
"""
import copy
import os
from types import MethodType
from unittest.mock import patch

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from aio.core import Injection, PoolSpec, anchors
from aio.core.interactions.motifs import Motif, TRIANGLE, SUMMED_TRIANGLE, TAIL, CHAIN, K4
from aio.core.interactions.pools import node_pool_at, summed_triangle_dense
from aio.core.routing.anchors import gather
from aio.models.jets.particle_transformer import ParticleTransformer
from aio.models.jets.particle_transformer_aio import AIOParticleTransformer, parity_check


DEVICE = torch.device(os.environ.get("AIO_TEST_DEVICE", "cpu"))
torch.set_num_threads(1)


def legacy_pools(self, rbar, g, n, mask, cmask, cfac, anc):
    """Original full-batch gathers and independent query slices, retained only as a test oracle."""
    vals, diag, groups = [None] * len(self.uniq), {}, {}
    for u, p in enumerate(self.uniq):
        if p.node_path:
            vals[u] = node_pool_at(p, g, n, mask, cmask, cfac, anc)
        elif self.dense_matmul and not p.attended:
            vals[u] = gather(summed_triangle_dense(g, n, mask, cmask, p.bonds, cfac), anc)
        else:
            groups.setdefault(p.S, []).append(u)
    for S, members in groups.items():
        queries = [self.Wq[str(u)](rbar) if self.uniq[u].attended else None for u in members]
        parts = [[] for _ in members]
        for start in range(0, len(anc), self.chunk):
            sl = slice(start, start + self.chunk)
            args = (S, members, [None if q is None else q[sl] for q in queries], anc[sl], g, n, cmask, cfac)
            if self.ckpt and self.training and torch.is_grad_enabled():
                out = checkpoint(self._chunk, *args, use_reentrant=False)
            else:
                out = self._chunk(*args)
            for k, u in enumerate(members):
                parts[k].append(out[k])
                if self.uniq[u].attended:
                    diag.setdefault(u, []).append(out[len(members) + k])
        for k, u in enumerate(members):
            vals[u] = torch.cat(parts[k])
    stats = {u: torch.cat(v) for u, v in diag.items()}
    return vals, {"logit_std": {u: v[:, :self.H] for u, v in stats.items()},
                  "entropy": {u: v[:, self.H:] for u, v in stats.items()}}


def close(actual, expected, bf16=False):
    if actual is None or expected is None:
        assert actual is None and expected is None
        return
    torch.testing.assert_close(actual, expected, rtol=0.04 if bf16 else 2e-5,
                               atol=2e-3 if bf16 else 2e-6)


def close_reduction(actual, expected, tolerance=0.02):
    """BF16 reductions change rounding order; cancellation makes elementwise relative error unhelpful.

    Bound both the relative L2 error and the largest error relative to the tensor's largest value.
    FP32 tests separately compare every element of every gradient with tight tolerances.
    """
    actual, expected = actual.float(), expected.float()
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    if actual.numel():
        error = actual - expected
        assert error.norm() <= tolerance * expected.norm() + 1e-6
        assert error.abs().max() <= tolerance * expected.abs().max() + 1e-6


SPECS = [TRIANGLE, SUMMED_TRIANGLE, TAIL, CHAIN, K4,
         (PoolSpec("ij", True, (1,)), PoolSpec("ij", True, (1,)), PoolSpec("i", True))]


@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("ckpt", [False, True])
@pytest.mark.parametrize("bf16", [False, True])
def test_grouped_pools_outputs_and_gradients(spec, ckpt, bf16, group_size=2, dense_matmul=False):
    torch.manual_seed(104)
    B, N, D, dr = 5, 6, 8, 4
    # The middle group has no anchors, the final group is partial, and jet 1 has no candidate k.
    mask = torch.arange(N, device=DEVICE)[None] < torch.tensor([6, 2, 1, 0, 5], device=DEVICE)[:, None]
    cand = mask.clone()
    cand[:, 3] = False
    anc = anchors(mask)
    anc = anc[torch.randperm(len(anc), device=DEVICE)]   # arbitrary caller order
    bonds = sorted({b for p in spec for b in p.bonds})
    g = torch.randn(B, N, N, D, device=DEVICE) * 0.5
    g = (g + g.transpose(1, 2)) * 0.5
    n = torch.randn(B, N, D, device=DEVICE) * 0.5
    cs = [torch.randn(B, N, 2, device=DEVICE) for _ in bonds]
    if bf16:
        g, n, cs = g.bfloat16(), n.bfloat16(), [c.bfloat16() for c in cs]
    source = [torch.randn(len(anc), dr, device=DEVICE), g, n, *cs,
              torch.rand(len(anc), device=DEVICE) + 0.2]
    model = Motif(spec, dr, D, 2, dr, anchor_chunk=4, ckpt=ckpt,
                  jet_group_size=group_size, dense_matmul=dense_matmul, diagnostics=True).to(DEVICE).train()
    with torch.no_grad():
        model.U.weight.normal_(std=0.1)
    reference = copy.deepcopy(model)
    reference.pools = MethodType(legacy_pools, reference)
    probe = torch.randn(len(anc), dr, device=DEVICE)

    def run(mod):
        inputs = [v.detach().clone().requires_grad_() for v in source]
        r, gg, nn, *tail = inputs
        with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=bf16):
            out, selected, aux = mod(r, gg, nn, mask, dict(zip(bonds, tail[:-1])),
                                     anc=anc, gate=tail[-1], cand_mask=cand)
        grads = torch.autograd.grad(out, [*inputs, *mod.parameters()],
                                    grad_outputs=probe.to(out.dtype), allow_unused=True)
        assert torch.equal(selected, anc)
        assert out.abs().max() > 0
        return out, aux, grads

    expected, old_aux, old_grads = run(reference)
    actual, aux, grads = run(model)
    close(actual, expected, bf16)
    for v, ref in zip(aux["pools"], old_aux["pools"]):
        close(v, ref, bf16)
    for name in ("logit_std", "entropy"):
        for u in aux["stats"][name]:
            close(aux["stats"][name][u], old_aux["stats"][name][u], bf16)
    for v, ref in zip(grads, old_grads):
        close(v, ref, bf16)


@pytest.mark.parametrize("group_size", [0, 1, 16])
def test_grouping_boundaries(group_size):
    test_grouped_pools_outputs_and_gradients(TRIANGLE, True, False, group_size=group_size)


def test_summed_matmul_path():
    test_grouped_pools_outputs_and_gradients(SUMMED_TRIANGLE, True, False, dense_matmul=True)


@pytest.mark.parametrize("selection", ["dense", "sparse", "duplicates", "empty"])
@pytest.mark.parametrize("bf16", [False, True])
def test_dense_messages_reuse_and_gradients(selection, bf16):
    torch.manual_seed(202)
    B, H, N, dr = 3, 2, 7, 4
    mask = torch.tensor([[1, 0, 1, 1, 1, 0, 1], [1] * 7, [1, 1, 0, 0, 0, 0, 0]],
                        dtype=torch.bool, device=DEVICE)
    anc = anchors(mask)
    if selection == "sparse":
        anc = anc[::3].flip(0)
    elif selection == "duplicates":
        anc = torch.cat([anc, anc[::4]])
    elif selection == "empty":
        anc = anc[:0]
    dtype = torch.bfloat16 if bf16 else torch.float32
    delta = torch.randn(len(anc), dr, dtype=dtype, device=DEVICE)
    # Noncontiguous post-dropout attention; each downstream block has different weights.
    weights = [torch.nn.functional.dropout(torch.randn(B, H, N, N, device=DEVICE).softmax(-1),
                                          p=0.25).to(dtype).transpose(-1, -2) for _ in range(3)]
    injection = Injection(dr, H, 8).to(DEVICE)
    probe = torch.randn(3, B, H, N, 4, device=DEVICE)

    def run(dense, fp32_reference=False):
        mod = copy.deepcopy(injection)
        # The FP32 oracle uses exactly the same quantised inputs/weights as the BF16 run.
        if bf16 and fp32_reference:
            with torch.no_grad():
                mod.M.weight.copy_(mod.M.weight.bfloat16().float())
        d = (delta.float() if fp32_reference else delta).detach().clone().requires_grad_()
        aa = [(a.float() if fp32_reference else a).detach().clone().requires_grad_() for a in weights]
        with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=bf16 and not fp32_reference):
            if dense:
                pairs = mod.dense_pairs(d, anc, B, N)
                close(pairs, pairs.transpose(1, 2), bf16)
                assert pairs.diagonal(dim1=1, dim2=2).count_nonzero() == 0
                out = torch.stack([mod.message_dense(a, pairs) for a in aa])
            else:
                out = torch.stack([mod.message(a, d, anc, chunk=5) for a in aa])
        grad_out = probe.to(dtype).to(out.dtype)
        grads = torch.autograd.grad(out, [d, *aa, mod.M.weight], grad_out, allow_unused=True)
        return out, grads

    expected, old_grads = run(False)
    actual, grads = run(True)
    compare = close_reduction if bf16 else close
    compare(actual, expected)
    for v, ref in zip(grads, old_grads):
        # With no anchors, the sparse implementation has no graph edge to d or attn.
        if selection == "empty" and ref is None:
            assert v is None or v.count_nonzero() == 0
        else:
            compare(v, ref)
    if bf16 and selection != "empty":
        precise, precise_grads = run(False, fp32_reference=True)
        close_reduction(actual, precise)
        for v, ref in zip(grads, precise_grads):
            close_reduction(v, ref)


def model_config():
    return dict(input_dim=4, num_classes=3, embed_dims=(16,), pair_embed_dims=(12, 12, 12),
                num_heads=2, num_layers=5, num_cls_layers=1, fc_params=[], trim=False)


def batch():
    mask = (torch.arange(7, device=DEVICE)[None] < torch.tensor([7, 6, 2, 4, 5], device=DEVICE)[:, None])[:, None]
    x = torch.randn(5, 4, 7, device=DEVICE).masked_fill(~mask, 0)
    p = torch.randn(5, 3, 7, device=DEVICE)
    v = torch.cat([p, (p.square().sum(1, keepdim=True) + 1).sqrt()], dim=1).masked_fill(~mask, 0)
    return x, v, mask


@pytest.mark.parametrize("bf16", [False, True])
def test_model_training_outputs_gradients_and_batchnorm(bf16):
    torch.manual_seed(303)
    model = AIOParticleTransformer(**model_config(), f3_after=2, D=8, H3=2, d_r=4,
                                  anchor_chunk=5, jet_group_size=2).to(DEVICE).train()
    model.require_calibration = False
    with torch.no_grad():
        model.hob.motif.U.weight.normal_(std=0.1)
    reference = copy.deepcopy(model)
    reference.message_backend = "sparse"
    reference.hob.motif.pools = MethodType(legacy_pools, reference.hob.motif)
    # Execution options add no checkpoint keys.
    reference.load_state_dict(model.state_dict(), strict=True)
    data = batch()

    def run(mod):
        x, v, mask = data
        x = x.detach().clone().requires_grad_()
        torch.manual_seed(404)                      # same host attention/dropout masks
        with torch.autocast(DEVICE.type, dtype=torch.bfloat16, enabled=bf16):
            out = mod(x, v=v, mask=mask)
            loss = torch.nn.functional.cross_entropy(out.float(), torch.tensor([0, 1, 2, 1, 0], device=DEVICE))
        loss.backward()
        return out, x.grad

    expected, old_xgrad = run(reference)
    with patch.object(model.inj, "dense_pairs", wraps=model.inj.dense_pairs) as build:
        actual, xgrad = run(model)
        assert build.call_count == 1                # shared by all three downstream blocks
    close(actual, expected, bf16)
    close(xgrad, old_xgrad, bf16)
    gradient_groups = {key: ([], []) for key in ("host", "hob", "inj")}
    for (name, prm), (old_name, old_prm) in zip(model.named_parameters(), reference.named_parameters()):
        assert name == old_name
        if not bf16 or prm.grad is None or old_prm.grad is None:
            close(prm.grad, old_prm.grad)
        else:
            group = name.split(".")[0] if name.startswith(("hob.", "inj.")) else "host"
            actual_grads, expected_grads = gradient_groups[group]
            actual_grads.append(prm.grad.flatten())
            expected_grads.append(old_prm.grad.flatten())
    if bf16:
        # Near-null BatchNorm/attention biases are especially cancellation-sensitive in BF16.
        # Check host, motif, and injection gradients separately so the large host cannot hide F3 errors.
        for actual_grads, expected_grads in gradient_groups.values():
            close_reduction(torch.cat(actual_grads), torch.cat(expected_grads), tolerance=0.05)
    for (name, buf), (_, ref) in zip(model.named_buffers(), reference.named_buffers()):
        torch.testing.assert_close(buf, ref, rtol=0, atol=0)
    assert model.hob.motif.Wq["0"].weight.grad.abs().max() > 0
    assert model.inj.M.weight.grad.abs().max() > 0


def test_zero_initialisation_and_calibration():
    torch.manual_seed(505)
    base = ParticleTransformer(**model_config()).to(DEVICE)
    model = AIOParticleTransformer(**model_config(), f3_after=2, D=8, H3=2, d_r=4,
                                  anchor_chunk=5, jet_group_size=2).to(DEVICE)
    model.load_baseline(base.state_dict())
    data = batch()
    for calibrated in (False, True):
        if calibrated:
            model.calibrate(*data)
            assert model.hob.calibrated.item()
        report = parity_check(model, base, data, rtol=1e-5, atol=1e-5)
        assert report["identity_pass"] and report["adapter_pass"] and report["deployment_pass"]


@pytest.mark.parametrize("rho", [None, 0.5, 0.0])
def test_auto_message_policy(rho):
    torch.manual_seed(606)
    model = AIOParticleTransformer(**model_config(), f3_after=2, D=8, H3=2, d_r=4,
                                  anchor_chunk=5, jet_group_size=2, use_router=True).to(DEVICE).eval()
    model.rho = rho
    with torch.no_grad(), patch.object(model.inj, "dense_pairs", wraps=model.inj.dense_pairs) as dense, \
            patch.object(model.inj, "message", wraps=model.inj.message) as sparse:
        out = model(*batch())
        assert torch.isfinite(out).all()
        assert dense.call_count == (1 if rho is None else 0)
        assert sparse.call_count == (3 if rho == 0.5 else 0)


def test_local_gathers_are_bounded():
    """Check checkpoint recomputation also reads group-local factors, not the original batch."""
    model = Motif(TRIANGLE, 4, 8, 2, 4, anchor_chunk=3, jet_group_size=2, ckpt=True).to(DEVICE)
    mask = torch.ones(5, 4, dtype=torch.bool, device=DEVICE)
    anc = anchors(mask)
    seen = []
    original = model._chunk

    def record(S, members, q, ac, g, n, cmask, cfac):
        seen.append((g.shape[0], n.shape[0]))
        return original(S, members, q, ac, g, n, cmask, cfac)

    model._chunk = record
    out, _, _ = model(torch.randn(len(anc), 4, device=DEVICE, requires_grad=True),
                       torch.randn(5, 4, 4, 8, device=DEVICE, requires_grad=True),
                       torch.randn(5, 4, 8, device=DEVICE, requires_grad=True), mask, anc=anc)
    out.sum().backward()
    assert seen and all(g <= 2 and n <= 2 for g, n in seen)
