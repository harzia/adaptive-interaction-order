"""Leakage guards, a learnable complementarity control, and actual sparse dispatch."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn

from aio.core.routing.frozen_experts import (
    FEATURE_NAMES, ConstantGate, ExpertCache, InputGate, file_sha256, hard_dispatch,
    input_identities, jet_gate_features, validate_splits,
)
from aio.core.routing.experiment import evaluate_gates, fit_gate, export_jetclass2_metrics
from aio.models.jets.representation_experiments import build_experiment, experiment_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def inputs(n=8, seed=19):
    rng = torch.Generator().manual_seed(seed)
    spatial = torch.randn(n, 3, 6, generator=rng)
    spatial[:, 0] += 4
    energy = (spatial.square().sum(1, keepdim=True) + .2).sqrt()
    return dict(features=torch.randn(n, 3, 6, generator=rng),
                vectors=torch.cat((spatial, energy), 1), mask=torch.ones(n, 1, 6, dtype=torch.bool),
                labels=torch.randint(0, 3, (n,), generator=rng),
                source_file_index=torch.zeros(n, dtype=torch.long), source_selected_index=torch.arange(n),
                metadata=dict(files=[f"/synthetic/source-{seed}.root"], data_config_sha256="test-config"))


def make_cache(path, n=128, seed=1):
    rng = np.random.default_rng(seed)
    x = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
    x[:, 0] = rng.permutation(np.resize([-1., 1.], n))
    y = rng.integers(0, 2, n, dtype=np.int64)
    probability_true = np.where(x[:, :1] > 0, [.95, .05], [.05, .95])
    p = np.empty((n, 2, 2), dtype=np.float32)
    for i in range(2):
        p[np.arange(n), i, y] = probability_true[:, i]
        p[np.arange(n), i, 1-y] = 1-probability_true[:, i]
    path.mkdir()
    arrays = dict(features=x, probabilities=p, labels=y,
                  event_ids=np.array([hashlib.sha256(f"event:{seed}:{i}".encode()).hexdigest() for i in range(n)], dtype="S64"),
                  content_ids=np.array([hashlib.sha256(f"content:{seed}:{i}".encode()).hexdigest() for i in range(n)], dtype="S64"))
    for name, value in arrays.items():
        np.save(path / f"{name}.npy", value)
    metadata = dict(format="aio-frozen-experts-v1", class_names=["a", "b"], feature_names=list(FEATURE_NAMES),
                    preprocessing_sha256="synthetic", input_dim=3,
                    experts=[dict(name=str(i), family="region", model={}, checkpoint_sha256=str(i)) for i in range(2)])
    (path / "manifest.json").write_text(json.dumps(metadata))
    return ExpertCache(path)


def cli(script, *args, success=True):
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)],
                            cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT)},
                            text=True, capture_output=True)
    if success:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0
    return result


def test_gate_features_mask_permutation_scale_and_beam_symmetry():
    data = inputs()
    p, mask = data["vectors"].double(), data["mask"]
    original = jet_gate_features(p, mask)
    order = torch.tensor([4, 2, 5, 0, 1, 3])
    torch.testing.assert_close(original, jet_gate_features(p[..., order] * 17, mask[..., order]))
    padded = torch.cat((p, p.new_full((len(p), 4, 2), float("nan"))), -1)
    padded_mask = torch.cat((mask, torch.zeros(len(p), 1, 2, dtype=torch.bool)), -1)
    torch.testing.assert_close(original, jet_gate_features(padded, padded_mask))
    angle = .7
    moved = p.clone()
    moved[:, 0] = np.cos(angle) * p[:, 0] - np.sin(angle) * p[:, 1]
    moved[:, 1] = np.sin(angle) * p[:, 0] + np.cos(angle) * p[:, 1]
    moved[:, 2] = np.cosh(.4) * p[:, 2] + np.sinh(.4) * p[:, 3]
    moved[:, 3] = np.cosh(.4) * p[:, 3] + np.sinh(.4) * p[:, 2]
    torch.testing.assert_close(original, jet_gate_features(moved, mask))
    with pytest.raises(ValueError, match="Nonempty"):
        jet_gate_features(p, torch.zeros_like(mask))


def test_identities_ignore_labels_padding_and_constituent_order():
    data = inputs()
    ids, hashes = input_identities(data)
    changed = {**data, "labels": 2-data["labels"]}
    for key in ("features", "vectors", "mask"):
        changed[key] = data[key].flip(-1)
    other_ids, other_hashes = input_identities(changed)
    np.testing.assert_array_equal(ids, other_ids)
    np.testing.assert_array_equal(hashes, other_hashes)
    changed["metadata"] = {**data["metadata"], "files": ["/renamed/copy.root"]}
    other_ids, other_hashes = input_identities(changed)
    assert (ids != other_ids).all()
    np.testing.assert_array_equal(hashes, other_hashes)


@pytest.mark.parametrize("field", ["event_ids", "content_ids"])
def test_rejects_overlap_and_duplicate_rows(tmp_path, field):
    fit = make_cache(tmp_path / "fit", seed=1)
    val = make_cache(tmp_path / "val", seed=2)
    validate_splits(fit, val)
    values = np.array(getattr(val, field), copy=True)
    values[0] = getattr(fit, field)[0]
    np.save(val.path / f"{field}.npy", values)
    with pytest.raises(ValueError, match="Overlapping"):
        validate_splits(fit, ExpertCache(val.path))
    values[1] = values[0]
    np.save(val.path / f"{field}.npy", values)
    with pytest.raises(ValueError, match="Duplicate"):
        ExpertCache(val.path)


def test_rejects_expert_order_class_order_and_bad_probabilities(tmp_path):
    fit, val = make_cache(tmp_path / "fit"), make_cache(tmp_path / "val", seed=2)
    for key in ("experts", "class_names"):
        original = val.metadata[key]
        val.metadata[key] = original[::-1]
        with pytest.raises(ValueError, match="Caches differ"):
            validate_splits(fit, val)
        val.metadata[key] = original
    bad = np.array(val.probabilities, copy=True)
    bad[0, 0] = [-.1, 1.1]
    np.save(val.path / "probabilities.npy", bad)
    with pytest.raises(ValueError, match="probabilities"):
        ExpertCache(val.path)


def test_gate_learns_input_predictable_complementarity_on_unseen_events(tmp_path):
    fit, validation, test = (make_cache(tmp_path / name, n=512, seed=seed)
                             for name, seed in (("fit", 1), ("val", 2), ("test", 3)))
    validate_splits(fit, validation, test)
    gates = {}
    for kind in ("constant", "soft", "hard"):
        gates[kind], info = fit_gate(fit, validation, kind=kind, hidden=0, epochs=50,
                                     patience=12, batch_size=128, learning_rate=.05)
        assert np.isfinite(info["validation_ce"])
    report, losses, _, names = evaluate_gates(test, gates)
    assert report["methods"]["constant"]["ce"] > .6
    assert report["methods"]["soft"]["ce"] < .2
    assert report["methods"]["hard"]["ce"] < .06
    oracle = report["diagnostic_only_oracle"]["ce"]
    assert oracle <= min(report["methods"]["expert:0"]["ce"], report["methods"]["expert:1"]["ce"])
    assert report["hard_route_fraction"] == [.5, .5]
    assert len(names) == losses.shape[1] == 6
    torch.testing.assert_close(gates["hard"].mean, torch.from_numpy(fit.features.mean(0)))
    # Class labels do not enter the gate: held-out predictions depend only on x.
    weights = gates["hard"](torch.from_numpy(test.features.copy()))
    assert weights.shape == (len(test), 2)


def test_uninformative_inputs_cannot_reach_oracle_with_constant_gate(tmp_path):
    fit, val = make_cache(tmp_path / "fit"), make_cache(tmp_path / "val", seed=2)
    for cache in (fit, val):
        np.save(cache.path / "features.npy", np.zeros_like(cache.features))
    fit, val = ExpertCache(fit.path), ExpertCache(val.path)
    gate, info = fit_gate(fit, val, kind="hard", epochs=5, hidden=0)
    assert info["validation_ce"] > 1
    assert gate(torch.from_numpy(val.features.copy())).argmax(-1).unique().numel() == 1


def test_dispatch_skips_unused_experts_and_restores_event_order():
    class Gate(nn.Module):
        def forward(self, x):
            choice = (x[:, 0] > 1.5).long()
            return torch.nn.functional.one_hot(choice, 3).float()
    class Expert(nn.Module):
        def __init__(self, offset):
            super().__init__()
            self.count, self.offset = 0, offset
        def forward(self, x, v, mask):
            self.count += len(x)
            return torch.stack((x[:, 0, 0] + self.offset, -x[:, 0, 0]), -1)
    data = inputs(n=5)
    data["mask"][[1, 3], :, 2:] = False
    experts = [Expert(i).eval() for i in range(3)]
    output, choices = hard_dispatch(experts, Gate().eval(), data["features"], data["vectors"], data["mask"])
    assert choices.tolist() == [1, 0, 1, 0, 1]
    assert [e.count for e in experts] == [2, 3, 0]
    expected = torch.stack((data["features"][:, 0, 0] + choices, -data["features"][:, 0, 0]), -1).softmax(-1)
    torch.testing.assert_close(output, expected)


def test_cli_checkpoint_cache_fit_dispatch_roundtrip(tmp_path):
    overrides = dict(hidden_dim=16, num_layers=1, num_heads=2, ffn_dim=24,
                     geometry_dim=8, local_neighbors=2, fc_params=[])
    experts = []
    for representation in ("sv", "rank2"):
        preset = f"regions4_ca_{representation}"
        model = build_experiment(preset, input_dim=3, num_classes=3, **overrides)
        checkpoint = tmp_path / f"{representation}.pt"
        torch.save({"state_dict": {"module.mod." + k: v for k, v in model.state_dict().items()}}, checkpoint)
        experts.append(dict(name=representation, preset=preset, checkpoint=str(checkpoint), model_overrides=overrides))
    manifest = tmp_path / "experts.json"
    manifest.write_text(json.dumps(dict(class_names=["a", "b", "c"], experts=experts)))
    for name, seed in (("fit", 31), ("val", 32), ("test", 33)):
        path = tmp_path / f"{name}.pt"
        torch.save(inputs(seed=seed), path)
        cli("cache_representation_experts.py", "--experts", manifest, "--inputs", path,
            "--output", tmp_path / name, "--batch-size", 3)
    cli("run_frozen_expert_routing.py", "diagnose", "--cache", tmp_path / "val", "--output", tmp_path / "diagnostic.json")
    diagnostic = json.loads((tmp_path / "diagnostic.json").read_text())
    assert set(diagnostic["methods"]) == {"expert:sv", "expert:rank2", "uniform"}
    cli("run_frozen_expert_routing.py", "fit", "--fit", tmp_path / "fit", "--validation", tmp_path / "val",
        "--test", tmp_path / "test", "--output", tmp_path / "run", "--epochs", 3)
    cli("predict_routed_representation.py", "--router", tmp_path / "run/router.pt", "--inputs", tmp_path / "test.pt",
        "--output", tmp_path / "prediction.pt", "--batch-size", 3)
    prediction = torch.load(tmp_path / "prediction.pt", weights_only=True)
    cache = ExpertCache(tmp_path / "test")
    choices = prediction["expert_choice"].numpy()
    np.testing.assert_allclose(prediction["probabilities"].numpy(), cache.probabilities[np.arange(len(cache)), choices], atol=2e-6)
    assert json.loads((tmp_path / "run/report.json").read_text())["best_single_selected_on_validation"].startswith("expert:")
    cli("run_frozen_expert_routing.py", "fit", "--fit", tmp_path / "fit", "--validation", tmp_path / "val",
        "--test", tmp_path / "val", "--output", tmp_path / "bad", success=False)
    assert not (tmp_path / "bad").exists()
    # Replacing a checkpoint after gate fitting must not silently change routing.
    torch.save({}, experts[0]["checkpoint"])
    cli("predict_routed_representation.py", "--router", tmp_path / "run/router.pt", "--inputs", tmp_path / "test.pt",
        "--output", tmp_path / "bad-prediction.pt", success=False)


def test_jetclass2_export_has_aligned_rows_and_no_oracle(tmp_path, monkeypatch):
    from integrations.weaver.eval import jetclass2_eval as evaluator
    cache = make_cache(tmp_path / "cache", n=8)
    p = np.zeros((8, 2, 188), dtype=np.float32)
    p[:, :, :2] = cache.probabilities
    np.save(cache.path / "probabilities.npy", p)
    metadata = cache.metadata
    metadata["class_names"] = evaluator.CLASS_NAMES
    (cache.path / "manifest.json").write_text(json.dumps(metadata))
    cache = ExpertCache(cache.path)
    calls = []
    def evaluate(*, prediction_files, args):
        chunks = list(evaluator._iter_prediction_chunks(prediction_files))
        labels = np.concatenate([c[0] for c in chunks])
        probabilities = np.concatenate([c[1] for c in chunks])
        np.testing.assert_array_equal(labels, cache.labels)
        np.testing.assert_allclose(probabilities.sum(1), 1)
        calls.extend(prediction_files)
    monkeypatch.setattr(evaluator, "evaluate_prediction_files", evaluate)
    gates = {k: ConstantGate(2).eval() for k in ("constant", "soft", "hard")}
    folder = tmp_path / "outputs"
    folder.mkdir()
    export_jetclass2_metrics(cache, gates, folder, batch_size=3)
    assert calls == ["expert_0", "expert_1", "uniform", "constant", "soft", "hard"]


def test_matched_granularity_configs_and_sweep_launch(tmp_path):
    for k in (4, 8, 16, 32):
        for source in ("", "_ca"):
            for representation in ("sv", "rank2", "wide"):
                for pair in ("", "_pair_local"):
                    config = experiment_config(f"regions{k}{source}_{representation}{pair}")
                    expected = {**experiment_config(f"regions8{source}_{representation}{pair}"), "num_regions": k}
                    assert config == expected
    plan = cli("run_region_sweep.py", "--regions", 4, 16, 32, "--sources", "ca", "learned", "--seeds", 124,
               "--pair-bias", "local", "--", "--num-epochs", 2)
    assert len(plan.stdout.splitlines()) == 18
    assert len(set(plan.stdout.splitlines())) == 18
    assert "regions32_ca_rank2_pair_local" in plan.stdout
    assert not (tmp_path / "output").exists()
    result = subprocess.run(["bash", "scripts/run_representation_experiment.sh", "regions32_ca_rank2_pair_local", "--seed", "124"],
                            cwd=ROOT, env={**os.environ, "AIO_DRY_RUN": "1"}, text=True, capture_output=True)
    assert result.returncode == 0 and "regions32_ca_rank2_pair_local" in result.stdout
    cli("run_region_sweep.py", "--regions", 4, 4, success=False)
