"""Physics/operator checks plus masking, optimization and Weaver integration checks."""
from __future__ import annotations

import math
from types import SimpleNamespace
import pytest
import torch
from aio.core.geometry.attention import GeometricAttention
from aio.core.geometry.lorentz import inverse_lorentz, metric_like
from aio.models.jets.region_transformer import RegionTransformer
from aio.models.jets.representation_experiments import experiment_config


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(27)
    torch.set_num_threads(2)


def batch(n=13, f=3, dtype=torch.float64):
    spatial = torch.randn(2, n, 3, dtype=dtype)
    spatial[..., 0] += 2
    energy = spatial.square().sum(-1).add(0.3).sqrt()
    v = torch.cat((spatial, energy[..., None]), -1).transpose(1, 2)
    return torch.randn(2, f, n, dtype=dtype), v, torch.ones(2, 1, n, dtype=torch.bool)


def small(k=4, tensors=1, dtype=torch.float64):
    return RegionTransformer(3, 5, num_regions=k, tensors=tensors, hidden_dim=16,
                             geometry_dim=12, num_heads=2, ffn_dim=24, num_layers=2,
                             local_neighbors=4).to(dtype)


def boost_rotation(dtype=torch.float64):
    beta = 0.41
    gamma = (1 - beta**2)**-0.5
    boost = torch.eye(4, dtype=dtype)
    boost[0, 0] = boost[1, 1] = gamma
    boost[0, 1] = boost[1, 0] = -gamma * beta
    rotation = torch.eye(4, dtype=dtype)
    angle = 0.73
    rotation[1:3, 1:3] = torch.tensor([[math.cos(angle), -math.sin(angle)],
                                     [math.sin(angle), math.cos(angle)]], dtype=dtype)
    return boost @ rotation


@pytest.mark.parametrize("k,tensors", [(0, 0), (4, 0), (4, 1), (1, 1)])
def test_full_model_lorentz_covariance_with_transformed_references(k, tensors):
    model = small(k, tensors).eval()
    x, v, mask = batch()
    references = torch.tensor([[1., 0., 0., 0.], [0., 0., 0., 1.]], dtype=v.dtype)[None].expand(2, -1, -1)
    transform = boost_rotation()
    p = v[:, [3, 0, 1, 2]].transpose(1, 2)
    boosted = (p @ transform.T)[..., [1, 2, 3, 0]].transpose(1, 2)
    with torch.no_grad():
        y, g = model(x, v, mask, references=references, return_aux=True)
        y2, g2 = model(x, boosted, mask, references=references @ transform.T, return_aux=True)
    assert g["frame_valid"].all()
    torch.testing.assert_close(g["frames"], g2["frames"] @ transform, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(y, y2, atol=1e-8, rtol=1e-8)
    if k:
        torch.testing.assert_close(g["assignments"], g2["assignments"], atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(g["regularization"], g2["regularization"], atol=1e-9, rtol=1e-9)


def test_permutation_and_padding_do_not_change_prediction_or_regions():
    model = small().eval()
    x, v, mask = batch()
    with torch.no_grad():
        y, g = model(x, v, mask, return_aux=True)
        perm = torch.randperm(v.shape[-1])
        yp, gp = model(x[..., perm], v[..., perm], mask[..., perm], return_aux=True)
        xp = torch.cat((x, torch.full((2, 3, 7), float("nan"), dtype=x.dtype)), -1)
        vp = torch.cat((v, torch.full((2, 4, 7), float("nan"), dtype=v.dtype)), -1)
        mp = torch.cat((mask, torch.zeros(2, 1, 7, dtype=torch.bool)), -1)
        yz, gz = model(xp, vp, mp, return_aux=True)
    torch.testing.assert_close(y, yp)
    torch.testing.assert_close(y, yz)
    torch.testing.assert_close(g["assignments"][:, perm], gp["assignments"])
    torch.testing.assert_close(g["assignments"], gz["assignments"][:, :13])
    assert not gz["assignments"][:, 13:].any()


def test_rank2_transport_and_contraction_against_explicit_pairwise_operator():
    model = small().eval()
    x, v, mask = batch(n=6)
    _, g = model(x, v, mask, return_aux=True)
    frames, valid = g["frames"], g["frame_valid"]
    attention = GeometricAttention(7, 2, 2, 1, 1).double()
    h = torch.randn(2, 6, 7, dtype=torch.float64)
    actual = attention(h, frames, valid, mask[:, 0])
    q, key, value = attention.qkv(h).reshape(2, 6, 3, 2, 22).unbind(2)
    # Slow reference: transport EACH sender's K,V into EACH receiver's frame.
    outputs = []
    for i in range(6):
        pair_frame = frames[:, i:i+1] @ inverse_lorentz(frames)
        keys_i = attention.transport(key, pair_frame, valid)
        values_i = attention.transport(value, pair_frame, valid)
        logits = ((q[:, i:i+1] * attention.metric) * keys_i).sum(-1) / math.sqrt(22)
        weights = logits.softmax(1)
        outputs.append((weights[..., None] * values_i).sum(1))
    expected = attention.out(torch.stack(outputs, 1).flatten(-2))
    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)


def test_identity_regions_equal_full_attention_and_k1_is_not_full():
    x, v, mask = batch(n=6)
    _, g = small()(x, v, mask, return_aux=True)
    attention = GeometricAttention(7, 2, 2, 1, 1).double()
    h = torch.randn(2, 6, 7, dtype=x.dtype)
    args = (h, g["frames"], g["frame_valid"], mask[:, 0])
    full = attention(*args)
    identical = attention(*args, assignments=torch.eye(6, dtype=x.dtype)[None].expand(2, -1, -1),
                          mass=torch.ones(2, 6, dtype=x.dtype), region_mask=mask[:, 0])
    coarse = attention(*args, assignments=torch.ones(2, 6, 1, dtype=x.dtype),
                       mass=torch.full((2, 1), 6., dtype=x.dtype), region_mask=torch.ones(2, 1, dtype=torch.bool))
    torch.testing.assert_close(full, identical)
    assert (coarse - full).abs().max() > 1e-4


def test_learned_regions_and_tensor_parameters_receive_gradients():
    model = small(dtype=torch.float32)
    x, v, mask = batch(dtype=torch.float32)
    logits, g = model(x, v, mask, return_aux=True)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1, 3])) + g["regularization"]
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name  # Required by DDP find_unused_parameters=False.
        assert torch.isfinite(parameter.grad).all(), name
    assert model.geometry.assignment.weight.grad.abs().sum() > 0
    grad = model.blocks[0].global_attention.qkv.weight.grad.reshape(3, 2, 32, 16)
    assert grad[:, :, 16:].abs().sum() > 0


def test_shared_geometry_frozen_checkpoint_preserves_assignments(tmp_path):
    source, target = small(tensors=0), small(tensors=1)
    source.save_geometry(tmp_path / "geometry.pt")
    target.load_geometry(tmp_path / "geometry.pt")
    x, v, mask = batch()
    _, gs = source(x, v, mask, return_aux=True)
    _, gt = target(x, v, mask, return_aux=True)
    torch.testing.assert_close(gs["assignments"], gt["assignments"], rtol=0, atol=0)
    torch.testing.assert_close(gs["frames"], gt["frames"], rtol=0, atol=0)
    assert all(not p.requires_grad for p in target.geometry.parameters())
    with pytest.raises(ValueError, match="configuration mismatch"):
        small(k=8).load_geometry(tmp_path / "geometry.pt")


def test_tensor_gate_has_query_and_value_effect_and_finite_utility():
    model = small()
    x, v, mask = batch()
    gate = torch.ones(2, 4, dtype=x.dtype, requires_grad=True)
    on = model(x, v, mask, tensor_gate=gate)
    on.square().sum().backward()
    assert gate.grad is not None and torch.isfinite(gate.grad).all()
    assert gate.grad.abs().sum() > 0
    off = model(x, v, mask, tensor_gate=torch.zeros_like(gate))
    assert (on - off).abs().max() > 1e-6


def test_short_and_degenerate_jets_have_finite_outputs_and_gradients():
    model = small(dtype=torch.float32)
    x, v, mask = batch(n=1, dtype=torch.float32)
    v[:, :3] = 0
    v[:, 2] = 1
    v[:, 3] = 1  # exactly on the beam: no equivariant transverse frame exists
    y, g = model(x, v, mask, return_aux=True)
    (y.square().sum() + g["regularization"]).backward()
    assert torch.isfinite(y).all()
    assert not g["frame_valid"].any()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(ValueError, match="at least one"):
        model(x, v, torch.zeros_like(mask))


def test_parameter_matched_wide_control():
    tensor = RegionTransformer(7, 188, **experiment_config("regions8_rank2"))
    wide = RegionTransformer(7, 188, **experiment_config("regions8_wide"))
    assert sum(p.numel() for p in tensor.parameters()) == sum(p.numel() for p in wide.parameters())


def test_frames_are_proper_lorentz_and_local_graph_does_not_depend_on_k():
    x, v, mask = batch()
    _, g4 = small(k=4)(x, v, mask, return_aux=True)
    _, g8 = small(k=8)(x, v, mask, return_aux=True)
    frames = g4["frames"]
    metric = torch.diag(metric_like(frames))
    expected = metric.expand_as(frames)
    torch.testing.assert_close(frames @ metric @ frames.transpose(-1, -2), expected,
                               rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(torch.linalg.det(frames), torch.ones_like(frames[..., 0, 0]))
    assert torch.equal(g4["indices"], g8["indices"])


def test_unequal_region_masses_preserve_uniform_particle_attention():
    attention = GeometricAttention(1, num_heads=1, scalars=1, vectors=0).double()
    with torch.no_grad():
        attention.qkv.weight.copy_(torch.tensor([[0.], [0.], [1.]], dtype=torch.float64))
        attention.qkv.bias.zero_()
        attention.out.weight.fill_(1.)
        attention.out.bias.zero_()
    h = torch.tensor([[[2.], [4.], [8.], [10.]]], dtype=torch.float64)
    valid = torch.ones(1, 4, dtype=torch.bool)
    frames = torch.eye(4, dtype=h.dtype)[None, None].expand(1, 4, -1, -1)
    assignment = torch.tensor([[[1., 0.], [0., 1.], [0., 1.], [0., 1.]]], dtype=h.dtype)
    out = attention(h, frames, valid, valid, assignments=assignment, mass=assignment.sum(1),
                    region_mask=torch.ones(1, 2, dtype=torch.bool))
    torch.testing.assert_close(out, h.mean(1, keepdim=True).expand_as(h))


def test_weaver_factory_injected_amp_options_and_loss(monkeypatch):
    pytest.importorskip("weaver")
    from integrations.weaver import RegionLLoCa as adapter
    config = SimpleNamespace(input_dicts={"pf_features": ["a", "b", "c"]},
                             input_names=["pf_points", "pf_features", "pf_vectors", "pf_mask"],
                             input_shapes={"pf_points": (1, 2, 13), "pf_features": (1, 3, 13),
                                           "pf_vectors": (1, 4, 13), "pf_mask": (1, 1, 13)})
    model, info = adapter.get_model(config, preset="regions8_rank2", num_classes=5, use_amp=True,
                                   hidden_dim=16, num_layers=1, ffn_dim=24, fc_params=[])
    x, v, mask = batch(dtype=torch.float32)
    target = torch.tensor([1, 2])
    captured = {}

    def fake_train(model, loss_func, *args, **kwargs):
        logits = model(torch.zeros(2, 2, 13), x, v, mask)
        actual = loss_func(logits, target)
        expected = torch.nn.functional.cross_entropy(logits, target) + model.mod.last_aux_loss
        torch.testing.assert_close(actual, expected)
        actual.backward()
        captured["called"] = True

    monkeypatch.setattr(adapter, "train_classification", fake_train)
    adapter.get_train_fn(config)(model, adapter.get_loss(config))
    assert captured["called"] and info["input_names"] == config.input_names


def test_cpu_bfloat16_autocast_backward():
    model = small(dtype=torch.float32)
    x, v, mask = batch(dtype=torch.float32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(x, v, mask)
        loss = logits.float().square().mean() + model.last_aux_loss
    loss.backward()
    assert torch.isfinite(logits).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
