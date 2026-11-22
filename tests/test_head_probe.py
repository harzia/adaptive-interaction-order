"""Frozen-embedding head test: reweighting, scaling, evaluation and an end-to-end smoke run."""
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import yaml

if "uproot" not in sys.modules and importlib.util.find_spec("uproot") is None:   # imported by the evaluator, never used here
    sys.modules["uproot"] = types.ModuleType("uproot")

from aio.analysis import head_probe as hp
from aio.analysis import jet_observables as jo

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs/datasets/jetclass2/JetClassII_kin.yaml").read_text())


def test_class_membership_covers_every_label_once():
    labels = np.arange(188)
    index = hp.class_membership(labels, CONFIG)
    classes = CONFIG["weights"]["reweight_classes"]
    assert index.min() == 0 and index.max() == len(classes) - 1
    assert classes[index[9]] == "label_X_gg" and classes[index[77]] == "label_X_YY_gggg"
    assert set(index[161:]) == {classes.index("label_QCD")}


def test_flat_reweighting_flattens_spectra_and_sets_class_balance():
    rng = np.random.default_rng(0)
    n = 400_000
    labels = rng.choice([3, 9, 170], size=n, p=[0.5, 0.2, 0.3])           # X_QQ, X_gg, QCD
    pt = 200 + 2300 * rng.beta(1.0, 4.0, n)                                # steeply falling
    mass = 20 + 480 * rng.beta(2.0, 3.0, n)
    index = hp.class_membership(labels, CONFIG)
    hists = hp.flat_reweight_hists(index, pt, mass, CONFIG)
    w = hp.event_weights(index, pt, mass, hists, CONFIG)
    assert 0 < w.max() <= 0.9 + 1e-6
    classes, class_weights = CONFIG["weights"]["reweight_classes"], CONFIG["weights"]["class_weights"]
    totals = {classes[c]: w[index == c].sum() for c in np.unique(index)}
    expected = {name: class_weights[classes.index(name)] for name in totals}
    ratio = totals["label_X_QQ"] / totals["label_QCD"]
    assert ratio == pytest.approx(expected["label_X_QQ"] / expected["label_QCD"], rel=0.02)
    # Within a class the weighted pT spectrum is far flatter than the raw one.
    member = index == classes.index("label_X_QQ")
    bins = CONFIG["weights"]["reweight_vars"]["jet_pt"]
    raw = np.histogram(pt[member], bins)[0][:8]
    flat = np.histogram(pt[member], bins, weights=w[member])[0][:8]
    assert raw.max() / raw.min() > 1.3 and flat.max() / flat.min() < 1.05
    assert 0 < hp.effective_sample_size(w) < n


def test_event_weights_follow_weaver_for_overflow_and_auto_yaml(tmp_path):
    classes = CONFIG["weights"]["reweight_classes"]
    pt_bins, mass_bins = CONFIG["weights"]["reweight_vars"].values()
    hists = {name: np.full((len(pt_bins) - 1, len(mass_bins) - 1), 0.5, dtype=np.float32) for name in classes}
    hists["label_QCD"][0, 0] = 0.25
    index = hp.class_membership(np.array([3, 170, 170, 3, 3]), CONFIG)
    w = hp.event_weights(index, np.array([300.0, 210.0, 2600.0, 199.0, 2500.0]),
                         np.array([100.0, 25.0, 100.0, 100.0, 500.0]), hists, CONFIG)
    assert w.tolist() == [0.5, 0.25, 0.0, 0.0, 0.5]        # outside the binning: discarded, as in Weaver

    config_path = tmp_path / "data.yaml"
    config_path.write_text("weights: {}\n")
    assert hp.weaver_auto_yaml(config_path) is None
    import hashlib
    auto = tmp_path / f"data.{hashlib.md5(config_path.read_bytes()).hexdigest()}.auto.yaml"
    auto.write_text(yaml.safe_dump({"weights": {"reweight_hists": {k: v.tolist() for k, v in hists.items()}}}))
    assert hp.weaver_auto_yaml(config_path) == auto
    loaded = hp.load_reweight_hists(auto)
    assert set(loaded) == set(classes) and loaded["label_QCD"][0, 0] == pytest.approx(0.25)
    with pytest.raises(ValueError):
        hp.load_reweight_hists(config_path)

    logits = np.log(np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]]))
    loss = hp.weighted_cross_entropy(logits + 3.0, np.array([0, 1]), np.array([1.0, 3.0]))
    assert loss == pytest.approx((-np.log(0.7) - 3 * np.log(0.1)) / 4)


def test_feature_scaler_logs_standardises_and_handles_zeros():
    rng = np.random.default_rng(1)
    x = np.stack((np.exp(rng.normal(0, 3, 5000)), rng.normal(5, 2, 5000)), 1)
    x[:50, 0] = 0.0
    scaler = hp.FeatureScaler([True, False]).fit(x)
    z = scaler.transform(x)
    assert z.dtype == np.float32 and np.isfinite(z).all() and np.abs(z).max() <= hp.CLIP
    assert abs(z[:, 1].mean()) < 1e-3 and z[:, 1].std() == pytest.approx(1, abs=1e-3)
    assert (z[:50, 0] < z[50:, 0].min() + 1e-6).all()          # zeros sit just below the valid range
    assert hp.select_columns([("a", "x", True), ("b", "y", False)], ("b",)).tolist() == [False, True]
    with pytest.raises(ValueError):
        hp.select_columns([("a", "x", True)], ("missing",))


def synthetic_split(directory, rng, per_class):
    """Jets whose prong count follows the label; the embedding knows the label only weakly."""
    from integrations.weaver.eval import jetclass2_eval as evaluator
    directory.mkdir(parents=True)
    prongs = np.ones(188, dtype=int)
    prongs[:15] = 2
    prongs[evaluator.THREE_PRONG_INDICES] = 3
    prongs[evaluator.FOUR_PRONG_INDICES] = 4
    labels = np.repeat(np.arange(188), per_class)
    rng.shuffle(labels)
    vectors, counts = [], []
    for label in labels:
        k, n = prongs[label], 16
        axes = rng.normal(0, 0.25, (k, 2))
        which = rng.integers(0, k, n)
        y, phi = (axes[which] + rng.normal(0, 0.03, (n, 2))).T
        pt = 500 * rng.dirichlet(np.full(n, 0.7))
        vectors.append(np.stack((pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(y), pt * np.cosh(y)), 1))
        counts.append(n)
    signal = (prongs[labels] - 2.5)[:, None]
    embedding = np.concatenate((0.15 * signal + rng.normal(size=(len(labels), 1)),
                                rng.normal(size=(len(labels), 7))), 1).astype(np.float32)
    np.savez(directory / "Synthetic_0000.npz", label=labels.astype(np.int16), embedding=embedding,
             logits=rng.normal(size=(len(labels), 188)).astype(np.float32),
             jet_pt=rng.uniform(200, 2500, len(labels)).astype(np.float32),
             jet_sdmass=rng.uniform(20, 500, len(labels)).astype(np.float32),
             part_vectors=np.concatenate(vectors).astype(np.float32),
             part_offsets=np.concatenate(([0], np.cumsum(counts))))
    return labels


def test_end_to_end_with_the_repository_evaluator(tmp_path, monkeypatch, capsys):
    from integrations.weaver.eval import jetclass2_eval as evaluator
    from scripts import compute_jet_observables, train_embedding_head

    rng = np.random.default_rng(5)
    shards = tmp_path / "shards"
    for split, per_class in (("train", 10), ("val", 4), ("test", 6)):
        synthetic_split(shards / split, rng, per_class)
    (shards / "train" / "Synthetic_0001.tmp.npz").write_bytes(b"unfinished")     # must be ignored
    monkeypatch.setattr(sys, "argv", ["compute", *(str(shards / s) for s in ("train", "val", "test")),
                                      "--workers", "1"])
    compute_jet_observables.main()
    names = hp.load_observable_names(shards / "train")
    assert [tuple(n) for n in names] == jo.compute_batch(np.zeros((1, 1, 4)), np.zeros((1, 1), bool))[1]
    loaded = hp.load_split(shards / "test")
    assert loaded["obs"].shape == (188 * 6, len(names)) and np.isfinite(loaded["obs"]).all()

    reader = evaluator._iter_prediction_chunks
    monkeypatch.setattr(evaluator, "roc_auc_score", lambda *a, **k: 0.5)   # the slow sampled 188-way AUC
    output = tmp_path / "heads"
    monkeypatch.setattr(sys, "argv", ["train", "--shards", str(shards), "--output", str(output),
                                      "--backend", "sklearn", "--arms", "embedding", "embedding+nsub", "obs_only",
                                      "--seeds", "0", "1", "--epochs", "40", "--hidden", "64",
                                      "--samples-per-epoch", "4000", "--batch-size", "256"])
    train_embedding_head.main()
    assert evaluator._iter_prediction_chunks is reader            # the evaluator is restored
    summary = json.loads((output / "summary.json").read_text())
    assert set(summary["table"]) == {"parT_head", "embedding", "embedding+nsub", "obs_only"}
    assert (output / "embedding+nsub" / "seed1" / "metrics.json").is_file()
    metrics = json.loads((output / "embedding" / "seed0" / "metrics.json").read_text())
    assert metrics["model"]["network_config"] == "head:embedding" and metrics["run"]["seed"] == 0
    accuracy = {arm: row["accuracy_top1"]["mean"] for arm, row in summary["table"].items()}
    # N-subjettiness separates the prong counts that the weak embedding cannot.
    assert accuracy["embedding+nsub"] > 1.5 * accuracy["embedding"]
    assert summary["table"]["embedding+nsub"]["accuracy_top1"]["change"] > 0.5
    assert summary["table"]["embedding"]["accuracy_top1"]["change"] == pytest.approx(0)
    assert "embedding+nsub" in capsys.readouterr().out
