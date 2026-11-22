"""Shared pieces of the frozen-embedding head test (NumPy only).

The test asks whether hand-computed observables add information to a trained
tagger's class-token embedding. A small head is trained twice on held-out jets,
once with the embedding alone and once with the embedding plus observables, and
both are scored with the repository's JetClass-II evaluator.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

LOG_FLOOR_MARGIN = 1.0
CLIP = 8.0
SUMMARY_KEYS = ("rej50_geomean_hadronic", "rej30_geomean_hadronic", "accuracy_top1",
                "sig_eff_at_epsb1e-3_macro_hadronic", "auc_group_vs_qcd_macro_hadronic")
HADRONIC = ("X_QQ", "X_gg", "X_YY_QQQQ", "X_YY_QQgg", "X_YY_gggg",
            "X_YY_QQQ", "X_YY_QQg", "X_YY_Qgg", "X_YY_ggg")
FOUR_PRONG, THREE_PRONG, TWO_PRONG = HADRONIC[2:5], HADRONIC[5:], HADRONIC[:2]
PAIRS = ("X_bb_vs_QCD_bb", "X_qq_vs_X_gg", "X_YY_qqqq_vs_X_YY_gggg", "X_bb_vs_X_cc")


# ----------------------------------------------------------------------------- shards
def shard_paths(directory):
    """Jet shards in a split directory, without observable companions or unfinished writes."""
    return sorted(p for p in Path(directory).glob("*.npz")
                  if not p.name.endswith((".obs.npz", ".tmp.npz")))


def observable_path(shard):
    return Path(shard).with_suffix("").with_suffix(".obs.npz")


def load_split(directory, keys=("label", "embedding", "logits", "jet_pt", "jet_sdmass"), observables=True):
    """Concatenate a split's shards. Returns a dict of arrays, plus ``obs`` if requested."""
    paths = shard_paths(directory)
    if not paths:
        raise FileNotFoundError(f"No shards in {directory}")
    parts = {k: [] for k in keys}
    obs = []
    for path in paths:
        with np.load(path) as data:
            for k in keys:
                parts[k].append(data[k])
        if observables:
            companion = observable_path(path)
            if not companion.is_file():
                raise FileNotFoundError(f"Missing {companion}; run compute_jet_observables.py first")
            with np.load(companion) as data:
                obs.append(data["values"])
            if len(obs[-1]) != len(parts[keys[0]][-1]):
                raise ValueError(f"{companion} does not match its shard")
    out = {k: np.concatenate(v) for k, v in parts.items()}
    if observables:
        out["obs"] = np.concatenate(obs)
    return out


def load_observable_names(directory):
    """List of (group, name, log_scale) written by compute_jet_observables.py."""
    return [tuple(row) for row in json.loads((Path(directory) / "observables.json").read_text())]


# ----------------------------------------------------------------------------- reweighting
def class_membership(labels, config):
    """Index of each jet's reweighting class, from the data config's label expressions."""
    weights = config["weights"]
    index = np.full(len(labels), -1)
    for c, name in enumerate(weights["reweight_classes"]):
        expression = config["new_variables"][name]
        member = eval(compile(expression, name, "eval"), {"__builtins__": {}}, {"jet_label": labels, "np": np})
        index[np.asarray(member, dtype=bool)] = c
    return index


def flat_reweight_hists(class_index, x, y, config, max_weight=0.9):
    """Weaver's ``reweight_method: flat`` histograms, built from the given jets."""
    weights = config["weights"]
    (_, x_bins), (_, y_bins) = weights["reweight_vars"].items()
    threshold = weights.get("reweight_threshold", 10)
    raw, result, events = {}, {}, {}
    for c, (name, class_weight) in enumerate(zip(weights["reweight_classes"], weights["class_weights"])):
        member = class_index == c
        hist = np.histogram2d(x[member], y[member], bins=(x_bins, y_bins))[0].astype(np.float32)
        raw[name] = hist
        if not (hist > 0).any():
            result[name], events[name] = np.zeros_like(hist), None
            continue
        populated = hist[hist > np.median(hist[hist > 0]) * 0.01]
        reference = np.percentile(populated, threshold)
        with np.errstate(divide="ignore"):
            result[name] = np.clip(np.nan_to_num(reference / hist, posinf=0), 0, 1)
        events[name] = float((hist * result[name]).sum()) / class_weight
    present = [v for v in events.values() if v]
    if not present:
        raise ValueError("No jets fall in any reweighting class and bin")
    smallest = min(present) * max_weight
    return {name: result[name] * (smallest / events[name] if events[name] else 0.0) for name in result}


def weaver_auto_yaml(data_config):
    """The ``<name>.<md5>.auto.yaml`` Weaver writes beside a data config when it trains, if present."""
    import hashlib
    data_config = Path(data_config)
    digest = hashlib.md5(data_config.read_bytes()).hexdigest()
    candidate = data_config.with_name(f"{data_config.stem}.{digest}.auto.yaml")
    return candidate if candidate.is_file() else None


def load_reweight_hists(path):
    """Histograms from a Weaver auto-generated YAML, to reproduce a training run exactly."""
    import yaml
    hists = (yaml.safe_load(Path(path).read_text()).get("weights") or {}).get("reweight_hists")
    if not hists:
        raise ValueError(f"{path} has no weights.reweight_hists; pass the training run's *.auto.yaml")
    return {name: np.asarray(h, dtype=np.float32) for name, h in hists.items()}


def event_weights(class_index, x, y, hists, config):
    """Per-jet sampling weight, as Weaver's ``_build_weights`` computes it."""
    weights = config["weights"]
    (_, x_bins), (_, y_bins) = weights["reweight_vars"].items()
    ix = np.clip(np.digitize(x, x_bins) - 1, 0, len(x_bins) - 2)
    iy = np.clip(np.digitize(y, y_bins) - 1, 0, len(y_bins) - 2)
    out = np.zeros(len(x), dtype=np.float64)
    for c, name in enumerate(weights["reweight_classes"]):
        member = class_index == c
        out[member] = np.asarray(hists[name])[ix[member], iy[member]]
    if weights.get("reweight_discard_under_overflow", True):
        out[(x < min(x_bins)) | (x > max(x_bins)) | (y < min(y_bins)) | (y > max(y_bins))] = 0.0
    return out


def effective_sample_size(w):
    total = w.sum()
    return float(total * total / (w * w).sum()) if total > 0 else 0.0


# ----------------------------------------------------------------------------- features
class FeatureScaler:
    """Log the heavy-tailed columns, then standardise and clip. Fitted on the training split.

    Works through the rows in blocks, so it never holds a float64 copy of a whole split.
    """

    def __init__(self, log_columns, block=200_000):
        self.log_columns = np.asarray(log_columns, dtype=bool)
        self.block = block

    def _blocks(self, x):
        for start in range(0, len(x), self.block):
            yield slice(start, start + self.block)

    def _log(self, x):
        x = np.array(x, dtype=np.float64)
        cols = self.log_columns
        positive = x[:, cols] > 0
        x[:, cols] = np.where(positive, np.log(np.where(positive, x[:, cols], 1.0)), self.floor)
        return x

    def fit(self, x, weights=None):
        cols = self.log_columns
        lowest = np.full(int(cols.sum()), np.inf)
        for rows in self._blocks(x):
            part = x[rows][:, cols]
            lowest = np.minimum(lowest, np.where(part > 0, part, np.inf).min(0, initial=np.inf))
        found = np.isfinite(lowest)
        self.floor = np.where(found, np.log(np.where(found, lowest, 1.0)), 0.0) - LOG_FLOOR_MARGIN
        w = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=np.float64)
        total = w.sum()
        self.mean = sum((w[rows, None] * self._log(x[rows])).sum(0) for rows in self._blocks(x)) / total
        variance = sum((w[rows, None] * (self._log(x[rows]) - self.mean) ** 2).sum(0)
                       for rows in self._blocks(x)) / total
        self.scale = np.sqrt(variance)
        self.scale[self.scale < 1e-12] = 1.0
        return self

    def transform(self, x):
        out = np.empty(x.shape, dtype=np.float32)
        for rows in self._blocks(x):
            out[rows] = np.clip((self._log(x[rows]) - self.mean) / self.scale, -CLIP, CLIP)
        return out


def select_columns(names, groups):
    """Boolean column mask for the requested observable groups."""
    known = {g for g, _, _ in names}
    missing = set(groups) - known
    if missing:
        raise ValueError(f"Unknown observable groups {sorted(missing)}; available: {sorted(known)}")
    return np.array([g in groups for g, _, _ in names])


def softmax(logits):
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def weighted_cross_entropy(logits, labels, weights, block=200_000):
    """Weighted mean of -log softmax(logits)[label], as the head trainer's validation loss."""
    total = 0.0
    for start in range(0, len(labels), block):
        z = logits[start:start + block].astype(np.float64)
        z -= z.max(1, keepdims=True)
        picked = z[np.arange(len(z)), labels[start:start + block]]
        total += float((weights[start:start + block] * (np.log(np.exp(z).sum(1)) - picked)).sum())
    return total / float(np.sum(weights))


# ----------------------------------------------------------------------------- evaluation
def evaluate_chunks(chunks, results_dir, data_config=None, network_config=None, seed=None):
    """Score predictions with the repository's JetClass-II evaluator.

    ``chunks`` yields ``(labels, probabilities[n, 188])``. The evaluator streams
    prediction ROOT files; here its file reader is replaced by that iterator, so
    the metric definitions are exactly the ones used for full training runs. It
    writes ``metrics.json`` into ``results_dir`` and returns the metrics dict.
    """
    from integrations.weaver.eval import jetclass2_eval as evaluator

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    def reader(_files):
        for labels, probabilities in chunks:
            yield np.asarray(labels).astype(np.int64), np.asarray(probabilities, dtype=np.float32)

    original = evaluator._iter_prediction_chunks
    evaluator._iter_prediction_chunks = reader
    try:
        args = SimpleNamespace(data_config=data_config, network_config=network_config, seed=seed)
        result = evaluator.evaluate_prediction_files(
            prediction_files={"head": str(results_dir / "pred_head.root")}, args=args)
    finally:
        evaluator._iter_prediction_chunks = original
    return result["metrics"]


def geomean(values):
    values = [v for v in values if v is not None and np.isfinite(v) and v > 0]
    return float(np.exp(np.mean(np.log(values)))) if values else None


def summarise(metrics):
    """The handful of numbers the comparison is read from."""
    out = {k: metrics.get(k) for k in SUMMARY_KEYS}
    group = lambda names: geomean([metrics.get(f"rej50_per_group/{g}") for g in names])
    out.update({"rej50_4prong": group(FOUR_PRONG), "rej50_3prong": group(THREE_PRONG),
                "rej50_2prong": group(TWO_PRONG)})
    out.update({f"rej50/{g}": metrics.get(f"rej50_per_group/{g}") for g in HADRONIC})
    out.update({f"pair/{p}": metrics.get(f"rej50_pairs/{p}") for p in PAIRS})
    return out


def compare(arms, reference):
    """Mean over seeds per arm and its relative change against the reference arm.

    ``arms`` maps arm name to a list of ``summarise`` dicts, one per seed, with
    seeds in the same order for every arm so the differences are paired.
    """
    table = {}
    for name, runs in arms.items():
        row = {}
        for key in runs[0]:
            values = np.array([np.nan if r[key] is None else r[key] for r in runs], dtype=float)
            base = np.array([np.nan if r[key] is None else r[key] for r in arms[reference]], dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = values / base - 1.0
            row[key] = dict(mean=float(np.nanmean(values)) if np.isfinite(values).any() else None,
                            change=float(np.nanmean(ratio)) if np.isfinite(ratio).any() else None,
                            change_spread=float(np.nanstd(ratio)) if np.isfinite(ratio).sum() > 1 else None)
        table[name] = row
    return table


def format_table(table, reference, keys=("rej50_geomean_hadronic", "rej50_4prong", "rej50_3prong",
                                         "rej50_2prong", "pair/X_bb_vs_QCD_bb", "pair/X_bb_vs_X_cc",
                                         "accuracy_top1", "val_loss")):
    """Plain-text table: value for the reference arm, relative change for the others."""
    width = max(len(n) for n in table) + 2
    lines = ["arm".ljust(width) + "".join(k.split("/")[-1][:18].rjust(20) for k in keys)]
    for name, row in table.items():
        cells = []
        for k in keys:
            cell = row.get(k, {})
            if name == reference:
                cells.append("-" if cell.get("mean") is None else f"{cell['mean']:.4g}")
            elif cell.get("change") is None:
                cells.append("-")
            else:
                spread = "" if cell.get("change_spread") is None else f" ±{100 * cell['change_spread']:.1f}"
                cells.append(f"{100 * cell['change']:+.1f}%{spread}")
        lines.append(name.ljust(width) + "".join(c.rjust(20) for c in cells))
    return "\n".join(lines)
