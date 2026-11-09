"""Physical basis, shared attention, state algebra, and matched ParT integration."""
from __future__ import annotations

import copy
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from aio.core.geometry.angular_moments import (
    AngularEmbedding, AngularUpdate, decode_stf, encode_stf, jet_angular_geometry, stf_basis,
)
from aio.models.jets.particle_transformer import Attention, ParticleTransformer
from aio.models.jets.angular_particle_transformer import AngularParticleTransformer
from aio.models.jets.angular_experiments import CONFIG_ROOT, build_experiment, experiment_config
from aio.models.jets.representation_experiments import experiment_config as old_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def setup():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(29)
    yield
    torch.set_num_threads(previous)


def batch(n=9, dtype=torch.float64):
    p = torch.randn(4, 3, n, dtype=dtype)
    p[:, 0] += 3
    energy = (p.square().sum(1, keepdim=True) + .2).sqrt()
    mask = torch.ones(4, 1, n, dtype=torch.bool)
    mask[1, 0, -2:] = False
    return torch.randn(4, 3, n, dtype=dtype), torch.cat((p, energy), 1), mask


def small(preset="moments_l2_d2", **kwargs):
    return build_experiment(preset, input_dim=3, num_classes=5, embed_dims=[16, 32, 16],
        pair_embed_dims=[8, 8], num_heads=2, fc_params=[], trim=False,
        block_params=dict(dropout=0., attn_dropout=0., activation_dropout=0.), **kwargs)


def activate(model):
    with torch.no_grad():
        for layer in model.moment_layers:
            layer.scalar_out.weight.normal_(std=.1)
            if hasattr(layer, "tv_mix"):
                layer.tv_mix.normal_(std=.07)
                layer.vv_mix.normal_(std=.07)
    return model


def rotation():
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    q[:, 0] *= torch.linalg.det(q)
    return q


def test_stf_basis_and_angular_correlation_identity():
    basis = stf_basis()
    torch.testing.assert_close(basis.flatten(1) @ basis.flatten(1).T, torch.eye(5, dtype=torch.float64))
    n = F.normalize(torch.randn(7, 3, dtype=torch.float64), dim=-1)
    q = encode_stf(n[..., :, None] * n[..., None, :], basis)
    decoded = decode_stf(q, basis)
    torch.testing.assert_close(decoded, n[..., :, None] * n[..., None, :] - torch.eye(3, dtype=n.dtype) / 3)
    weights = torch.randn(7, dtype=n.dtype, requires_grad=True)
    moment = (weights[:, None] * q).sum(0)
    actual = moment.square().sum()
    expected = (weights[:, None] * weights[None] * ((n @ n.T).square() - 1/3)).sum()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, weights, retain_graph=True)[0], torch.autograd.grad(expected, weights)[0])


def test_geometry_rest_boost_scale_permutation_and_degeneracies():
    _, p, mask = batch()
    g = jet_angular_geometry(p, mask)
    assert g["used_rest"].all()
    torch.testing.assert_close(g["energy_fraction"].sum(1), torch.ones(4, dtype=p.dtype))
    perm = torch.randperm(p.shape[-1])
    changed = jet_angular_geometry(p[..., perm] * 1e4, mask[..., perm])
    torch.testing.assert_close(changed["direction"], g["direction"][:, perm])
    torch.testing.assert_close(changed["energy_fraction"], g["energy_fraction"][:, perm])
    moved = p.clone()
    moved[:, 2] = math.cosh(.6) * p[:, 2] + math.sinh(.6) * p[:, 3]
    moved[:, 3] = math.cosh(.6) * p[:, 3] + math.sinh(.6) * p[:, 2]
    boosted = jet_angular_geometry(moved, mask)
    torch.testing.assert_close(boosted["energy_fraction"], g["energy_fraction"], atol=1e-10, rtol=1e-9)
    torch.testing.assert_close(boosted["direction"] @ boosted["direction"].transpose(1, 2),
                               g["direction"] @ g["direction"].transpose(1, 2), atol=1e-10, rtol=1e-9)
    edge = torch.tensor([[[1., 2., 0.], [0., 0., 0.], [0., 0., 0.], [1., 2., 0.]],
                         [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [1., 2., 0.]]], dtype=torch.float64, requires_grad=True)
    edge_mask = torch.tensor([[[True, True, False]], [[True, True, False]]])
    result = jet_angular_geometry(edge, edge_mask)
    assert result["used_rest"].tolist() == [False, True]
    assert not result["direction"][1].any() and not result["quadrupole"][1].any()
    sum(result[k].square().sum() for k in ("direction", "quadrupole", "energy_fraction")).backward()
    assert torch.isfinite(edge.grad).all()
    with pytest.raises(ValueError, match="valid particle"):
        jet_angular_geometry(p, torch.zeros_like(mask))


@pytest.mark.parametrize("dropout", [0., .25])
def test_extra_values_share_exact_weights_dropout_and_gradients(dropout):
    attention = Attention(8, 2, dropout=dropout).double().train()
    attention.use_sdpa = False
    x = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
    extra = torch.randn(2, 2, 5, 9, dtype=torch.float64, requires_grad=True)
    padding = torch.tensor([[False, False, False, True, True], [False] * 5])
    bias = torch.randn(2, 2, 5, 5, dtype=torch.float64)
    torch.manual_seed(11)
    actual, typed = attention(x, x, x, key_padding_mask=padding, attn_mask=bias, extra_values=extra)
    q, k, v = attention.in_proj(x).chunk(3, -1)
    q, k, v = [t.reshape(2, 5, 2, 4).transpose(1, 2) for t in (q, k, v)]
    logits = q @ k.transpose(-1, -2) / 2 + bias
    weights = logits.masked_fill(padding[:, None, None], -torch.inf).softmax(-1)
    torch.manual_seed(11)
    weights = F.dropout(weights, p=dropout)
    expected = attention.out_proj((weights @ v).transpose(1, 2).reshape(2, 5, 8))
    expected_typed = (weights @ extra).transpose(1, 2)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(typed, expected_typed)
    params = (x, extra, *attention.parameters())
    grads = torch.autograd.grad(actual.square().sum() + typed.square().sum(), params, retain_graph=True)
    reference = torch.autograd.grad(expected.square().sum() + expected_typed.square().sum(), params)
    for value, target in zip(grads, reference):
        torch.testing.assert_close(value, target)


def test_sdpa_extended_values_match_explicit_kernel():
    attention = Attention(12, 3).double().eval()
    x = torch.randn(2, 6, 12, dtype=torch.float64)
    extra = torch.randn(2, 3, 6, 18, dtype=torch.float64)
    padding = torch.tensor([[False] * 4 + [True] * 2, [False] * 6])
    actual = attention(x, x, x, key_padding_mask=padding, extra_values=extra)
    attention.use_sdpa = False
    reference = attention(x, x, x, key_padding_mask=padding, extra_values=extra)
    for value, target in zip(actual, reference):
        torch.testing.assert_close(value, target, atol=1e-10, rtol=1e-9)


@pytest.mark.parametrize("order,products", [(1, 2), (2, 1), (2, 2)])
def test_state_algebra_is_rotation_covariant(order, products):
    h = torch.randn(4, 9, 12, dtype=torch.float64)
    _, p, mask = batch()
    valid = mask[:, 0]
    geometry = jet_angular_geometry(p, mask)
    r = rotation()
    moved_p = p.clone()
    moved_p[:, :3] = torch.einsum("ij,bjn->bin", r, p[:, :3])
    moved_geometry = jet_angular_geometry(moved_p, mask)
    embedding = AngularEmbedding(12, 2, 2, order).double()
    layer = AngularUpdate(12, 2, 2, order, products, carry=True).double()
    with torch.no_grad():
        layer.scalar_out.weight.normal_(std=.1)
        if hasattr(layer, "tv_mix"):
            layer.tv_mix.normal_(std=.1)
            layer.vv_mix.normal_(std=.1)
    alpha = torch.randn(4, 2, 9, 9, dtype=torch.float64).masked_fill(~valid[:, None, None], -torch.inf).softmax(-1)
    def run(g):
        state = embedding(h, g, valid)
        messages = torch.einsum("bhij,bhjd->bihd", alpha, layer.values(state))
        return layer.update(h, messages, state, g, valid)
    delta, state = run(geometry)
    moved_delta, moved_state = run(moved_geometry)
    torch.testing.assert_close(delta, moved_delta, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(moved_state[0], state[0])
    torch.testing.assert_close(moved_state[1], state[1] @ r.T, rtol=1e-9, atol=1e-10)
    if order == 2:
        expected = r @ decode_stf(state[2], stf_basis()) @ r.T
        torch.testing.assert_close(decode_stf(moved_state[2], stf_basis()), expected, rtol=1e-9, atol=1e-10)


def test_equal_dipole_different_quadrupole_mechanism():
    planar = torch.tensor([[1., 0., 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.]], dtype=torch.float64)
    tetra = F.normalize(torch.tensor([[1., 1., 1.], [1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]], dtype=torch.float64), dim=-1)
    torch.testing.assert_close(planar.mean(0), tetra.mean(0))
    def quadrupole(n):
        return encode_stf(n[..., :, None] * n[..., None, :], stf_basis()).mean(0)
    assert quadrupole(planar).square().sum() > .1
    assert quadrupole(tetra).square().sum() < 1e-25
    # This is a fixed-moment mechanism check, not an impossibility claim about
    # learned vector networks or a JetClass performance experiment.


def test_disabled_extension_is_exact_part_and_common_initialization_matches():
    torch.manual_seed(50)
    reference = small("part_d2").double().eval()
    torch.manual_seed(50)
    model = small(moment_scale=0.).double().eval()
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
    x, v, mask = batch()
    torch.testing.assert_close(model(x, v, mask), reference(x, v, mask), rtol=0, atol=0)
    model.load_part_backbone(reference.state_dict())
    with pytest.raises(ValueError, match="exactly"):
        model.load_part_backbone({})
    assert type(build_experiment("part_d2")) is ParticleTransformer
    assert isinstance(build_experiment("moments_l2_d2"), AngularParticleTransformer)


def test_zero_residual_preserves_baseline_training_and_wrapped_padding_statistics():
    torch.manual_seed(50)
    reference = small("part_d2").double().train()
    torch.manual_seed(50)
    model = small().double().train()
    x, v, mask = batch()
    # Nonzero finite padding represents the dataset's wrap-padding convention.
    x[1, :, -2:] = x[1, :, :2]
    v[1, :, -2:] = v[1, :, :2]
    torch.testing.assert_close(model(x, v, mask), reference(x, v, mask), atol=1e-10, rtol=1e-9)
    for name, value in reference.named_buffers():
        if "running_" in name or "num_batches_tracked" in name:
            torch.testing.assert_close(model.get_buffer(name), value, atol=0, rtol=0)


def test_nonzero_angular_model_permutation_padding_beam_rotation_and_checkpoint():
    model = activate(small()).double().eval()
    x, v, mask = batch()
    expected = model(x, v, mask)
    model.moment_scale = 0
    assert (expected - model(x, v, mask)).abs().max() > 1e-5
    model.moment_scale = 1
    perm = torch.randperm(x.shape[-1])
    torch.testing.assert_close(model(x[..., perm], v[..., perm], mask[..., perm]), expected, atol=1e-9, rtol=1e-8)
    xp = torch.cat((x, x.new_full((4, 3, 2), float("nan"))), -1)
    vp = torch.cat((v, v.new_full((4, 4, 2), float("nan"))), -1)
    mp = torch.cat((mask, torch.zeros(4, 1, 2, dtype=torch.bool)), -1)
    torch.testing.assert_close(model(xp, vp, mp), expected, atol=1e-9, rtol=1e-8)
    moved = v.clone()
    moved[:, 0] = math.cos(.6) * v[:, 0] - math.sin(.6) * v[:, 1]
    moved[:, 1] = math.sin(.6) * v[:, 0] + math.cos(.6) * v[:, 1]
    torch.testing.assert_close(model(x, moved, mask), expected, atol=1e-9, rtol=1e-8)
    restored = small().double().eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored(x, v, mask), expected, rtol=0, atol=0)


@pytest.mark.parametrize("preset", ["moments_l1_d2", "moments_l2_d2", "moments_l2_noprod_d2", "moments_l2_reset_d2"])
@pytest.mark.parametrize("amp", [False, True])
def test_training_gradients_reach_all_parameters(preset, amp):
    model = activate(small(preset)).train()
    x, v, mask = batch(dtype=torch.float32)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        output = model(x, v, mask)
        loss = F.cross_entropy(output, torch.tensor([0, 1, 2, 3]))
    loss.backward()
    assert torch.isfinite(output).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    assert model.moment_embedding.amplitude.weight.grad.abs().sum() > 0


def test_zero_initialized_residual_starts_learning_without_manual_activation():
    model = small().train()
    x, v, mask = batch(dtype=torch.float32)
    initial = model.moment_embedding.amplitude.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.)
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x, v, mask), torch.tensor([0, 1, 2, 3]))
        loss.backward()
        assert torch.isfinite(loss)
        optimizer.step()
    assert not torch.equal(model.moment_embedding.amplitude.weight, initial)
    assert all(layer.scalar_out.weight.abs().sum() > 0 for layer in model.moment_layers)


def test_geometry_follows_the_same_trimmed_and_permuted_particles():
    model = activate(small()).double().eval()
    reference = copy.deepcopy(model)
    indices = torch.tensor([3, 0, 5, 1])
    class FixedTrimmer(nn.Module):
        def forward(self, x, v, mask, uu):
            assert uu is None
            return x[..., indices], v[..., indices], mask[..., indices], None
    model.trimmer = FixedTrimmer()
    x, v, mask = batch()
    expected = reference(x[..., indices], v[..., indices], mask[..., indices])
    torch.testing.assert_close(model(x, v, mask), expected, rtol=0, atol=0)


def test_weaver_adapter_uses_normal_part_for_baselines_and_preserves_readout():
    from integrations.weaver.AngularParT import get_model, get_loss, get_train_fn, get_evaluate_fn
    cfg = SimpleNamespace(input_dicts={"pf_features": ["a", "b", "c"]},
        input_names=["pf_points", "pf_features", "pf_vectors", "pf_mask"],
        input_shapes=dict(pf_points=(1, 2, 9), pf_features=(1, 3, 9), pf_vectors=(1, 4, 9), pf_mask=(1, 1, 9)))
    x, v, mask = batch(dtype=torch.float32)
    for preset, expected_type in (("part_d2", ParticleTransformer), ("moments_l2_d2", AngularParticleTransformer)):
        model, info = get_model(cfg, preset=preset, num_classes=5, trim=False, export_embed=True)
        model.eval()
        assert type(model.mod) is expected_type
        assert len(model.mod.cls_blocks) == 2 and model.mod.pair_embed is not None
        output = model(torch.zeros(4, 2, 9), x, v, mask)
        assert output.shape == (4, 5 + 48) and torch.isfinite(output).all()
        get_loss(cfg)(output[:, :5], torch.tensor([0, 1, 2, 3])).backward()
        assert info["input_names"] == cfg.input_names
        assert callable(get_train_fn(cfg)) and callable(get_evaluate_fn(cfg))


def test_matched_presets_and_cli_paths(tmp_path):
    _, four = experiment_config("part_d4")
    assert four == old_config("part_small")
    for path in CONFIG_ROOT.glob("*.json"):
        if path.stem == "common":
            continue
        family, config = experiment_config(path.stem)
        assert config["num_cls_layers"] == 2 and config["pair_input_dim"] == 4
        result = subprocess.run(["bash", "scripts/run_angular_experiment.sh", path.stem, "--num-epochs=7", "--seed", "124", "--train-percentage", "25"],
            cwd=ROOT, env={**os.environ, "AIO_DRY_RUN": "1", "COMMENT": "", "OUTPUT_PATH": str(tmp_path / "unused")}, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        assert "EPOCHS=7" in result.stdout and f"angular_{path.stem}_seed124" in result.stdout
        assert "AngularParT kin --train-percentage 25 --seed 124" in result.stdout
    assert not (tmp_path / "unused").exists()
    plan = subprocess.run([sys.executable, "scripts/run_angular_suite.py"], cwd=ROOT, text=True, capture_output=True)
    assert plan.returncode == 0 and len(plan.stdout.splitlines()) == 8
    assert len(set(plan.stdout.splitlines())) == 8
