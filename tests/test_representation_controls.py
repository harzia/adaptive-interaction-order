"""Controls must preserve masking/symmetry and actually intervene on the operator."""
from types import SimpleNamespace
import pytest
import torch
from aio.core.geometry.pair_bias import LocalFramePairBias, pair_features
from aio.core.geometry.attention import GeometricAttention
from aio.core.geometry.diagnostics import assignment_statistics, tensor_interventions
from aio.models.jets.region_transformer import RegionTransformer
from aio.models.jets.representation_experiments import build_experiment
from aio.models.jets.particle_transformer import pairwise_lv_fts_pp
from test_representation_order import batch, boost_rotation, seed, small


@pytest.mark.parametrize("pair,readout", [(True, "mean"), (False, "class_attention"), (True, "class_attention")])
def test_control_covariance_masking_permutation_and_gradients(pair, readout):
    model = RegionTransformer(3, 5, num_regions=0, hidden_dim=16, geometry_dim=12,
                             num_heads=2, num_layers=2, ffn_dim=24, local_neighbors=4,
                             pair_bias=pair, readout=readout).double().eval()
    x, v, mask = batch()
    refs = v.new_tensor([[1., 0., 0., 0.], [0., 0., 0., 1.]])[None].expand(2, -1, -1)
    transform = boost_rotation()
    boosted = (v[:, [3, 0, 1, 2]].transpose(1, 2) @ transform.T)[..., [1, 2, 3, 0]].transpose(1, 2)
    logits = model(x, v, mask, references=refs)
    torch.testing.assert_close(logits, model(x, boosted, mask, references=refs @ transform.T), atol=1e-8, rtol=1e-8)
    perm = torch.randperm(x.shape[-1])
    torch.testing.assert_close(logits, model(x[..., perm], v[..., perm], mask[..., perm]))
    xp = torch.cat((x, x.new_full((2, 3, 2), float("nan"))), -1)
    vp = torch.cat((v, v.new_full((2, 4, 2), float("nan"))), -1)
    mp = torch.cat((mask, torch.zeros(2, 1, 2, dtype=torch.bool)), -1)
    torch.testing.assert_close(logits, model(xp, vp, mp))
    logits.square().mean().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


def test_pair_features_match_part_in_nondegenerate_case():
    _, v, _ = batch()
    expected = pairwise_lv_fts_pp(v[:, :, :-1], v[:, :, 1:]).transpose(1, 2)
    p = v[:, [3, 0, 1, 2]].transpose(1, 2)
    torch.testing.assert_close(pair_features(p[:, :-1], p[:, 1:]), expected, rtol=1e-9, atol=1e-9)


def test_pair_bias_gradients_at_self_zero_and_beam_pairs():
    p = torch.tensor([[[1., 0., 0., 1.], [1., 0., 0., -1.], [0., 0., 0., 0.]]], requires_grad=True)
    features = pair_features(p[:, :, None], p[:, None, :])
    features.square().sum().backward()
    assert torch.isfinite(features).all() and torch.isfinite(p.grad).all()
    module = LocalFramePairBias(2)
    frames = torch.eye(4)[None, None].expand(1, 3, -1, -1)
    mask = torch.tensor([[True, True, False]])
    disabled = module(p.detach(), frames, torch.zeros_like(mask), mask)
    assert not disabled.any()


def test_pair_bias_changes_dense_and_neighbor_attention_consistently():
    attention = GeometricAttention(1, 1, 1, 0).double()
    with torch.no_grad():
        attention.qkv.weight.copy_(torch.tensor([[0.], [0.], [1.]], dtype=torch.float64))
        attention.qkv.bias.zero_()
        attention.out.weight.fill_(1.)
        attention.out.bias.zero_()
    h = torch.tensor([[[1.], [3.], [8.]]], dtype=torch.float64)
    mask = torch.ones(1, 3, dtype=torch.bool)
    frames = torch.eye(4, dtype=h.dtype)[None, None].expand(1, 3, -1, -1)
    bias = torch.tensor([[[[0., 1., 2.]]]], dtype=h.dtype).expand(1, 1, 3, 3)
    actual = attention(h, frames, mask, mask, pair_bias=bias)
    expected = (bias.softmax(-1)[0, 0, 0] * h[0, :, 0]).sum().expand_as(h)
    torch.testing.assert_close(actual, expected)
    indices = torch.arange(3)[None, None].expand(1, 3, -1)
    local = attention(h, frames, mask, mask, pair_bias=bias, indices=indices, neighbor_mask=mask[:, :, None].expand(1, 3, 3))
    torch.testing.assert_close(actual, local)
    no_bias = attention(h, frames, mask, mask)
    zero_bias = attention(h, frames, mask, mask, pair_bias=torch.zeros_like(bias))
    torch.testing.assert_close(no_bias, zero_bias, rtol=0, atol=0)
    assert (no_bias - actual).abs().max() > 0.1


def test_reject_undefined_coarse_pair_bias():
    with pytest.raises(ValueError, match="full particle"):
        RegionTransformer(3, 5, num_regions=8, pair_bias=True)


def test_assignment_diagnostics_disambiguate_uniform_and_disjoint_regions():
    hard = torch.tensor([[[1., 0.], [1., 0.], [0., 1.], [0., 1.]]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    p = torch.tensor([[[1., 1., 0., 0.], [1., 1., 0., 0.], [1., -1., 0., 0.], [1., -1., 0., 0.]]])
    a = assignment_statistics(hard, mask, p)
    b = assignment_statistics(torch.full_like(hard, .5), mask, p)
    torch.testing.assert_close(a["effective_occupied_regions"], b["effective_occupied_regions"])
    assert a["mean_region_overlap"].item() == 0
    assert b["mean_region_overlap"].item() == pytest.approx(1.)
    assert a["normalized_assignment_entropy"].item() == 0
    assert b["normalized_assignment_entropy"].item() == pytest.approx(1.)
    assert not a["angular_dispersion"].any()
    torch.testing.assert_close(b["angular_dispersion"], torch.ones(1, 2))


def test_tensor_interventions_sign_and_single_region_agreement():
    model = small(k=1).eval().requires_grad_(False)
    x, v, mask = batch()
    labels = torch.tensor([1, 2])
    stats = tensor_interventions(model, x, v, mask, labels)
    torch.testing.assert_close(stats["region_off_delta_ce"][:, 0], stats["all_tensors_off_delta_ce"])
    eps = 1e-5
    gate = torch.full((2, 1), 1. - eps, dtype=x.dtype)
    changed = torch.nn.functional.cross_entropy(model(x, v, mask, tensor_gate=gate), labels, reduction="none")
    torch.testing.assert_close((changed - stats["base_ce"]) / eps, stats["first_order_region_delta_ce"][:, 0], atol=1e-5, rtol=1e-3)


def test_small_part_factory_preserves_pair_and_class_attention():
    pytest.importorskip("weaver")
    from integrations.weaver import ParT
    cfg = SimpleNamespace(input_dicts={"pf_features": ["a", "b", "c"]},
                          input_names=["pf_points", "pf_features", "pf_vectors", "pf_mask"],
                          input_shapes={"pf_points": (1, 2, 13), "pf_features": (1, 3, 13),
                                        "pf_vectors": (1, 4, 13), "pf_mask": (1, 1, 13)})
    model, _ = ParT.get_model(cfg, preset="part_small", num_classes=5, trim=False)
    assert len(model.mod.blocks) == 4 and len(model.mod.cls_blocks) == 2
    assert model.mod.pair_embed is not None
    model.eval()
    x, v, mask = batch(dtype=torch.float32)
    logits = model(torch.zeros(2, 2, 13), x, v, mask)
    assert logits.shape == (2, 5) and torch.isfinite(logits).all()
    logits.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_new_controls_bfloat16_training():
    model = build_experiment("full_sv_pair_cls", input_dim=3, num_classes=5, hidden_dim=16,
                             num_layers=1, num_heads=2, ffn_dim=24, fc_params=[])
    x, v, mask = batch(dtype=torch.float32)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(x, v, mask)
        loss = logits.float().square().mean()
    loss.backward()
    assert torch.isfinite(logits).all()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name


def test_export_and_diagnose_cli_with_weaver_preprocessing(tmp_path):
    """Round trip actual Parquet/YAML preprocessing and prefixed checkpoints."""
    pytest.importorskip("weaver")
    ak = pytest.importorskip("awkward")
    pytest.importorskip("pyarrow")
    import json
    import subprocess
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    n = 128
    table = dict(jet_pt=[500.] * n, jet_energy=[600.] * n, jet_sdmass=[80.] * n,
                 jet_eta=[0.] * n, jet_phi=[0.] * n, jet_nparticles=[3] * n,
                 jet_label=[0] * n, **{f"jet_tau{i}": [.2] * n for i in range(1, 5)},
                 part_px=[[2., 3., 1.]] * n, part_py=[[.5, -.5, 1.]] * n,
                 part_pz=[[.4, .2, -.3]] * n, part_energy=[[2.2, 3.2, 1.5]] * n,
                 part_deta=[[.1, -.1, .2]] * n, part_dphi=[[.05, -.1, .15]] * n)
    source = tmp_path / "fixture.parquet"
    ak.to_parquet(ak.Array(table), source)
    inputs = tmp_path / "inputs.pt"
    command = [sys.executable, "scripts/export_representation_inputs.py", "--files", str(source),
               "--output", str(inputs), "--jets-per-file", "3", "--batch-size", "2"]
    subprocess.run(command, cwd=repo, check=True, capture_output=True, text=True)
    data = torch.load(inputs, weights_only=True)
    assert data["features"].shape == (3, 7, 128)
    assert data["mask"].sum().item() == 9
    torch.testing.assert_close(data["vectors"][0, 0, :3], torch.tensor([2., 3., 1.]))
    model = build_experiment("regions8_rank2")
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": {"module.mod." + k: v for k, v in model.state_dict().items()}}, checkpoint)
    report = tmp_path / "diagnostics.json"
    command = [sys.executable, "scripts/diagnose_representation_order.py", "--preset", "regions8_rank2",
               "--checkpoint", str(checkpoint), "--inputs", str(inputs), "--output", str(report),
               "--batch-size", "2", "--tensor-ablations"]
    subprocess.run(command, cwd=repo, check=True, capture_output=True, text=True)
    metadata = json.loads(report.read_text())
    result = torch.load(report.with_suffix(".pt"), weights_only=True)
    assert metadata["num_jets"] == 3 and metadata["num_regions"] == 8
    assert result["region_off_delta_ce"].shape == (3, 8)
    assert torch.isfinite(result["region_off_delta_ce"]).all()
    assert result["source_file_index"].tolist() == [0, 0, 0]
