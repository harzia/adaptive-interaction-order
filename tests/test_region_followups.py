"""Hierarchy partitions, sparse physics bias, diagnostic semantics, and CLI paths."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
import pytest
import torch
from aio.core.geometry.lund import ca_tree_regions, ca_assignments
from aio.core.geometry.diagnostics import assignment_statistics, communication_interventions
from aio.core.geometry.pair_bias import LocalFramePairBias
from aio.models.jets.region_transformer import RegionTransformer
from aio.models.jets.representation_experiments import build_experiment
from test_representation_order import batch, seed, small


def test_ca_matches_independent_sequential_angular_merging():
    pytest.importorskip("fastjet")
    _, v, mask = batch(n=9)
    p = v[:, [3, 0, 1, 2]].transpose(1, 2)
    actual = ca_tree_regions(p, mask[:, 0], 3)["assignments"]
    for jet in range(2):
        nodes = [(p[jet, i].clone(), {i}) for i in range(9)]
        while len(nodes) > 3:
            best = None
            for i in range(len(nodes)):
                for j in range(i):
                    left, right = nodes[i][0], nodes[j][0]
                    y1 = .5 * torch.log((left[0] + left[3]) / (left[0] - left[3]))
                    y2 = .5 * torch.log((right[0] + right[3]) / (right[0] - right[3]))
                    dp = (torch.atan2(left[2], left[1]) - torch.atan2(right[2], right[1]) + torch.pi) % (2 * torch.pi) - torch.pi
                    candidate = (float((y1-y2)**2 + dp**2), i, j)
                    if best is None or candidate < best:
                        best = candidate
            _, i, j = best
            merged = (nodes[i][0] + nodes[j][0], nodes[i][1] | nodes[j][1])
            nodes.pop(i)
            nodes.pop(j)
            nodes.append(merged)
        expected = {frozenset(members) for _, members in nodes}
        observed = {frozenset((actual[jet, :, region] > 0).nonzero(as_tuple=True)[0].tolist()) for region in range(3)}
        assert observed == expected


def test_ca_partition_nested_cuts_masking_and_short_jets():
    pytest.importorskip("fastjet")
    _, v, mask = batch(n=7)
    mask[1, 0, 3:] = False
    p = v[:, [3, 0, 1, 2]].transpose(1, 2)
    p[1, 3:] = float("nan")
    cuts = {k: ca_tree_regions(p, mask[:, 0], k) for k in [1, 3, 8]}
    for k, tree in cuts.items():
        a = tree["assignments"]
        torch.testing.assert_close(a.sum(-1), mask[:, 0].double())
        assert (a.sum(1) > 0).sum(1).tolist() == [min(k, 7), min(k, 3)]
        assert all(torch.isfinite(value).all() for value in tree.values())
    coarse, fine = cuts[3]["assignments"], cuts[8]["assignments"]
    for jet in range(2):
        for region in range(8):
            ids = fine[jet, :, region].bool()
            if ids.any():
                assert coarse[jet, ids].argmax(-1).unique().numel() == 1
    assert not cuts[8]["lund_split_valid"].any()
    assert cuts[1]["lund_split_valid"].all()


def test_batched_ca_matches_reference_with_padding_holes_and_short_jets():
    pytest.importorskip("fastjet")
    _, v, mask = batch(n=15)
    mask[0, 0, [1, 4, 5, 8]] = False
    mask[1, 0] = False
    mask[1, 0, [2, 9, 14]] = True
    p = v[:, [3, 0, 1, 2]].transpose(1, 2)
    p = p.masked_fill(~mask[:, 0, :, None], float("nan"))
    for k in [1, 4, 16]:
        batched = ca_assignments(p, mask[:, 0], k)
        reference = ca_tree_regions(p, mask[:, 0], k)["assignments"]
        # Coassignment comparison is independent of equal-pt slot permutations.
        torch.testing.assert_close(batched @ batched.transpose(1, 2), reference @ reference.transpose(1, 2))


def test_ca_rotation_longitudinal_boost_and_scale():
    pytest.importorskip("fastjet")
    _, v, mask = batch(n=11)
    p = v[:, [3, 0, 1, 2]].transpose(1, 2)
    original = ca_tree_regions(p, mask[:, 0], 3)
    moved = p.clone()
    c, s = torch.cos(torch.tensor(.8)), torch.sin(torch.tensor(.8))
    moved[..., 1] = c * p[..., 1] - s * p[..., 2]
    moved[..., 2] = s * p[..., 1] + c * p[..., 2]
    beta = .4
    gamma = (1-beta**2)**-.5
    moved[..., 0] = gamma * (p[..., 0] - beta * p[..., 3])
    moved[..., 3] = gamma * (p[..., 3] - beta * p[..., 0])
    transformed = ca_tree_regions(moved * 3, mask[:, 0], 3)
    torch.testing.assert_close(original["assignments"], transformed["assignments"])
    torch.testing.assert_close(original["lund_delta"], transformed["lund_delta"], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(original["lund_kt_fraction"], transformed["lund_kt_fraction"], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(original["lund_kt"] * 3, transformed["lund_kt"], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("local_pair", [False, True])
def test_ca_model_permutation_padding_and_gradients(local_pair):
    pytest.importorskip("fastjet")
    model = RegionTransformer(3, 5, num_regions=4, region_source="ca", tensors=1,
                              hidden_dim=16, geometry_dim=12, num_heads=2, ffn_dim=24,
                              num_layers=2, local_neighbors=4, pair_bias=local_pair,
                              pair_bias_scope="local").double().eval()
    x, v, mask = batch()
    y = model(x, v, mask)
    perm = torch.randperm(v.shape[-1])
    torch.testing.assert_close(y, model(x[..., perm], v[..., perm], mask[..., perm]))
    xp = torch.cat((x, x.new_full((2, 3, 2), float("nan"))), -1)
    vp = torch.cat((v, v.new_full((2, 4, 2), float("nan"))), -1)
    mp = torch.cat((mask, torch.zeros(2, 1, 2, dtype=torch.bool)), -1)
    torch.testing.assert_close(y, model(xp, vp, mp))
    (y.square().mean() + model.last_aux_loss).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.geometry.assignment is None and model.last_aux_loss == 0


def test_sparse_local_bias_matches_dense_gather_and_attention():
    model = RegionTransformer(3, 5, hidden_dim=16, geometry_dim=12, num_heads=2,
                              ffn_dim=24, num_layers=1, pair_bias=True).double().eval()
    x, v, mask = batch()
    _, g = model(x, v, mask, return_aux=True)
    p = v[:, [3, 0, 1, 2]].transpose(1, 2)
    module = model.pair_bias
    dense = module(p, g["frames"], g["frame_valid"], mask[:, 0])
    sparse = module(p, g["frames"], g["frame_valid"], mask[:, 0],
                    indices=g["indices"], neighbor_mask=g["neighbor_mask"])
    expected = dense.gather(3, g["indices"][:, None].expand(-1, 2, -1, -1))
    torch.testing.assert_close(sparse, expected, rtol=1e-9, atol=1e-9)
    attention = model.blocks[0].local_attention
    h = torch.randn(2, 13, 16, dtype=torch.float64)
    args = (h, g["frames"], g["frame_valid"], mask[:, 0])
    kwargs = dict(indices=g["indices"], neighbor_mask=g["neighbor_mask"])
    torch.testing.assert_close(attention(*args, pair_bias=dense, **kwargs),
                               attention(*args, neighbor_pair_bias=sparse, **kwargs), rtol=1e-9, atol=1e-9)


def test_communication_gate_removes_output_projection_bias_too():
    model = small(tensors=0).eval()
    x, v, mask = batch()
    off = model(x, v, mask, global_gate=0.)
    with torch.no_grad():
        for block in model.blocks:
            block.global_attention.out.weight.zero_()
            block.global_attention.out.bias.zero_()
    torch.testing.assert_close(off, model(x, v, mask), rtol=0, atol=0)
    result = communication_interventions(model, x, v, mask, torch.tensor([1, 2]))
    torch.testing.assert_close(result["global_off_delta_ce"], torch.zeros(2, dtype=x.dtype))


def test_shape_statistics_and_empty_regions():
    spatial = torch.cat((torch.eye(3), -torch.eye(3)), 0)[None]
    p = torch.cat((torch.ones(1, 6, 1), spatial), -1)
    mask = torch.ones(1, 6, dtype=torch.bool)
    a = torch.zeros(1, 6, 2)
    a[..., 0] = 1
    stats = assignment_statistics(a, mask, p)
    torch.testing.assert_close(stats["angular_covariance_eigenvalues"][0, 0], torch.full((3,), 1/3))
    assert stats["angular_anisotropy"][0, 0] < 1e-6
    assert not stats["angular_shape_valid"][0, 1]
    p[..., 1:] = torch.tensor([1., 0., 0.])
    stats = assignment_statistics(a, mask, p)
    assert not stats["angular_shape_valid"].any()


def test_ca_geometry_export_and_parameter_matched_controls(tmp_path):
    pytest.importorskip("fastjet")
    rank2 = build_experiment("regions8_ca_rank2")
    wide = build_experiment("regions8_ca_wide")
    assert sum(p.numel() for p in rank2.parameters()) == sum(p.numel() for p in wide.parameters())
    path = tmp_path / "geometry.pt"
    rank2.save_geometry(path)
    wide.load_geometry(path)
    assert not any(p.requires_grad for p in wide.geometry.parameters())
    with pytest.raises(ValueError, match="mismatch"):
        build_experiment("regions8_wide").load_geometry(path)


def test_diagnostics_and_plot_cli_on_shared_events(tmp_path):
    pytest.importorskip("fastjet")
    pytest.importorskip("matplotlib")
    root = Path(__file__).resolve().parents[1]
    x, v, mask = batch(n=6, f=7, dtype=torch.float32)
    inputs = tmp_path / "inputs.pt"
    torch.save(dict(features=x, vectors=v, mask=mask, labels=torch.tensor([77, 87]),
                    observer__jet_pt=torch.tensor([400., 500.]),
                    observer__jet_sdmass=torch.tensor([80., 90.])), inputs)
    reports = []
    for preset in ["regions8_rank2", "regions8_wide", "regions8_ca_rank2", "full_sv_pair"]:
        checkpoint = tmp_path / (preset + "_weights.pt")
        model = build_experiment(preset)
        torch.save({"state_dict": {"module.mod."+k: v for k,v in model.state_dict().items()}}, checkpoint)
        report = tmp_path / (preset + ".json")
        command = [sys.executable, "scripts/diagnose_representation_order.py", "--preset", preset,
                   "--checkpoint", str(checkpoint), "--inputs", str(inputs), "--output", str(report),
                   "--batch-size", "1", "--communication-ablations"]
        if "rank2" in preset:
            command.append("--tensor-ablations")
        subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
        info = json.loads(report.read_text())
        assert set(info["group_summaries"]) == {"X_YY_gggg", "X_YY_ggg"}
        assert "global_off_delta_ce" in info["summary"]
        assert info["input_sha256"]
        reports.append(report)
    output = tmp_path / "plots"
    subprocess.run([sys.executable, "scripts/plot_region_diagnostics.py", "--reports", *map(str, reports),
                    "--output-dir", str(output), "--examples", "1", "--pt-range", "300", "600"],
                   cwd=root, check=True, capture_output=True, text=True)
    assert (output / "utility_bins.csv").is_file()
    assert "paired_ce_difference" in (output / "group_summary.csv").read_text()
    assert (output / "2_regions8_ca_rank2_lund_cut.png").is_file()


def test_label_capped_export_preserves_observers_and_event_ids(tmp_path):
    pytest.importorskip("weaver")
    ak = pytest.importorskip("awkward")
    pytest.importorskip("pyarrow")
    root = Path(__file__).resolve().parents[1]
    n = 128
    table = dict(jet_pt=[500.] * n, jet_energy=[600.] * n, jet_sdmass=[80.] * n,
                 jet_eta=[0.] * n, jet_phi=[0.] * n, jet_nparticles=[3] * n,
                 jet_label=[77, 87] * (n//2), **{f"jet_tau{i}": [.2] * n for i in range(1,5)},
                 part_px=[[2., 3., 1.]] * n, part_py=[[.5, -.5, 1.]] * n,
                 part_pz=[[.4, .2, -.3]] * n, part_energy=[[2.2, 3.2, 1.5]] * n,
                 part_deta=[[.1, -.1, .2]] * n, part_dphi=[[.05, -.1, .15]] * n)
    source = tmp_path / "fixture.parquet"
    ak.to_parquet(ak.Array(table), source)
    output = tmp_path / "export.pt"
    command = [sys.executable, "scripts/export_representation_inputs.py", "--files", str(source),
               "--output", str(output), "--label-ids", "77", "87", "--jets-per-label", "2",
               "--jets-per-file", "100", "--batch-size", "3"]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    data = torch.load(output, weights_only=True)
    assert data["labels"].tolist() == [77, 87, 77, 87]
    assert data["source_selected_index"].tolist() == [0, 1, 2, 3]
    assert data["observer__jet_pt"].tolist() == [500.] * 4
    assert data["metadata"]["unfilled_label_counts"] == {}
