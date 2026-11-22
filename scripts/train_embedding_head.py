#!/usr/bin/env python3
"""Does a frozen tagger embedding miss information that hand-made observables carry?

Trains a small head on held-out jets for several arms and scores each on the
test shards with the repository's JetClass-II evaluator:

  parT_head             the checkpoint's own head (no training; the reference level)
  embedding             a new head on the frozen class-token embedding
  embedding+<group>     the same head, also given one group of observables
  embedding+all         the same head, given every observable
  obs_only              observables alone, as a check that they carry signal

By default every embedding arm is residual: its logits are the checkpoint's own
logits plus a correction that starts at zero. Each arm therefore starts exactly
at parT_head, and only a real gain can move it.

Training jets are drawn with Weaver's flat pT x soft-drop-mass reweighting, so
the head sees the distribution the tagger was trained on. Without this the head
could gain from mass and pT priors the tagger was deliberately kept blind to.
The weights come from --reweight-yaml, else from the auto.yaml Weaver left beside
the data config, else they are rebuilt from the train shards.
"""
from __future__ import annotations

import os

# Only the sampled 188-way AUC uses this; the rejections do not.
os.environ.setdefault("JETCLASS2_EVAL_AUC_SAMPLES_PER_CLASS", "200")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
from aio.analysis import head_probe as hp  # noqa: E402
from aio.analysis.jet_observables import GROUPS  # noqa: E402

REFERENCE = "embedding"


def parse_arm(name, groups):
    """Return (uses_embedding, observable_groups)."""
    if name == "obs_only":
        return False, tuple(groups)
    head, *extra = name.split("+")
    if head != "embedding":
        raise ValueError(f"Unknown arm {name!r}")
    if extra == ["all"]:
        return True, tuple(groups)
    unknown = set(extra) - set(groups)
    if unknown:
        raise ValueError(f"Arm {name!r} names unknown groups {sorted(unknown)}; available: {list(groups)}")
    return True, tuple(extra)


# ----------------------------------------------------------------------------- torch backend
def fit_torch(train, val, test, args, seed, residual):
    import torch
    from torch.nn import functional as F

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device(args.device)
    classes = args.num_classes

    class Head(torch.nn.Module):
        def __init__(self, width):
            super().__init__()
            layers = []
            for hidden in args.hidden:
                layers += [torch.nn.Linear(width, hidden), torch.nn.GELU(), torch.nn.Dropout(args.dropout)]
                width = hidden
            self.body = torch.nn.Sequential(*layers)
            self.out = torch.nn.Linear(width, classes)
            if residual:   # start exactly at the checkpoint's own logits
                torch.nn.init.zeros_(self.out.weight)
                torch.nn.init.zeros_(self.out.bias)

        def forward(self, x, base=None):
            y = self.out(self.body(x))
            return y if base is None else y + base

    def tensors(split):
        return (torch.from_numpy(split.x), torch.from_numpy(split.y.astype(np.int64)),
                torch.from_numpy(split.base) if residual else None)

    xtr, ytr, btr = tensors(train)
    xva, yva, bva = tensors(val)
    xte, _, bte = tensors(test)
    wva = torch.from_numpy(val.w.astype(np.float32))
    model = Head(xtr.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    batch = min(args.batch_size, len(ytr))
    steps = max(1, math.ceil(train.samples_per_epoch / batch))
    total_steps = args.epochs * steps
    warmup = max(1, total_steps // 10)

    def learning_rate_factor(step):   # linear warm-up, then cosine decay
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1.0 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    probability = train.w / train.w.sum()

    def forward(x, base, index):
        b = None if base is None else base[index].to(device, non_blocking=True)
        return model(x[index].to(device, non_blocking=True), b)

    @torch.no_grad()
    def validation_loss():
        model.eval()
        total, weight = 0.0, 0.0
        for start in range(0, len(yva), args.eval_batch):
            index = slice(start, start + args.eval_batch)
            loss = F.cross_entropy(forward(xva, bva, index), yva[index].to(device), reduction="none")
            w = wva[index].to(device)
            total, weight = total + float((loss * w).sum()), weight + float(w.sum())
        return total / max(weight, 1e-12)

    best_loss, best_state, best_epoch = validation_loss(), None, -1
    initial = best_loss
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for epoch in range(args.epochs):
        model.train()
        order = torch.from_numpy(rng.choice(len(probability), size=steps * batch, p=probability))
        for step in range(steps):
            index = order[step * batch:(step + 1) * batch]
            loss = F.cross_entropy(forward(xtr, btr, index), ytr[index].to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
        current = validation_loss()
        if current < best_loss:
            best_loss, best_epoch = current, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()

    def predict():
        for start in range(0, len(test.y), args.eval_batch):
            index = slice(start, start + args.eval_batch)
            with torch.no_grad():
                probabilities = forward(xte, bte, index).float().softmax(1).cpu().numpy()
            yield test.y[index], probabilities

    return predict(), dict(val_loss_initial=initial, val_loss_best=best_loss, best_epoch=best_epoch)


# ----------------------------------------------------------------------------- sklearn backend
def fit_sklearn(train, val, test, args, seed, residual):
    """CPU smoke-test backend. It ignores the residual option and is not for results."""
    from sklearn.neural_network import MLPClassifier

    rng = np.random.default_rng(seed)
    size = int(min(max(train.samples_per_epoch, 1), args.sklearn_samples))
    index = rng.choice(len(train.y), size=size, p=train.w / train.w.sum())
    model = MLPClassifier(hidden_layer_sizes=tuple(args.hidden), max_iter=args.epochs,
                          batch_size=min(args.batch_size, size), learning_rate_init=args.lr, random_state=seed)
    model.fit(train.x[index], train.y[index])

    def predict():
        for start in range(0, len(test.y), args.eval_batch):
            index = slice(start, start + args.eval_batch)
            probabilities = np.zeros((len(test.y[index]), args.num_classes), dtype=np.float32)
            probabilities[:, model.classes_] = model.predict_proba(test.x[index])
            yield test.y[index], probabilities

    return predict(), dict(val_loss_initial=None, val_loss_best=None, best_epoch=None)


# ----------------------------------------------------------------------------- driver
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shards", type=Path, required=True, help="Directory with train/, val/ and test/ shards")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-config", type=Path, default=ROOT / "configs/datasets/jetclass2/JetClassII_kin.yaml")
    p.add_argument("--reweight-yaml", type=Path,
                   help="Weaver *.auto.yaml from the training run (default: the one beside the data config, if any)")
    p.add_argument("--rebuild-reweight", action="store_true",
                   help="Ignore any auto.yaml and rebuild the flat weights from the train shards")
    p.add_argument("--arms", nargs="+", help="Default: embedding, one arm per observable group, embedding+all, obs_only")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--no-residual", dest="residual", action="store_false",
                   help="Train heads from scratch instead of as a correction to the checkpoint's logits")
    p.add_argument("--hidden", type=int, nargs="*", default=[1024, 1024])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--samples-per-epoch", type=int, help="Default: the effective sample size of the weighted train split")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--eval-batch", type=int, default=65536)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--num-classes", type=int, default=188)
    p.add_argument("--backend", choices=("torch", "sklearn"), default="torch")
    p.add_argument("--sklearn-samples", type=int, default=200_000)
    p.add_argument("--device", default=None, help="Default: cuda if available")
    args = p.parse_args()
    if args.backend == "torch":
        import torch
        args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if len(set(args.seeds)) != len(args.seeds) or min(args.epochs, args.batch_size, args.eval_batch) < 1:
        p.error("Expected distinct seeds and positive epochs and batch sizes")

    config = yaml.safe_load(args.data_config.read_text())
    names = hp.load_observable_names(args.shards / "train")
    for split in ("val", "test"):
        if hp.load_observable_names(args.shards / split) != names:
            p.error(f"{split} observables do not match train; rerun compute_jet_observables.py on all splits")
    groups = [g for g in GROUPS if any(g == group for group, _, _ in names)]
    arm_names = args.arms or (["embedding"] + [f"embedding+{g}" for g in groups] + ["embedding+all", "obs_only"])
    if REFERENCE not in arm_names:
        arm_names = [REFERENCE] + arm_names
    arms = {name: parse_arm(name, groups) for name in arm_names}

    data = {split: hp.load_split(args.shards / split) for split in ("train", "val", "test")}
    for split, d in data.items():
        print(f"{split}: {len(d['label'])} jets, embedding width {d['embedding'].shape[1]}, "
              f"{d['obs'].shape[1]} observables", flush=True)
    membership = {s: hp.class_membership(data[s]["label"].astype(np.int64), config) for s in ("train", "val")}
    source = None if args.rebuild_reweight else (args.reweight_yaml or hp.weaver_auto_yaml(args.data_config))
    try:
        hists = hp.load_reweight_hists(source) if source else None
    except ValueError:
        if args.reweight_yaml:
            raise
        hists = None          # an auto.yaml without histograms: fall back to rebuilding
    if hists is None:
        source = "rebuilt from the train shards"
        hists = hp.flat_reweight_hists(membership["train"], data["train"]["jet_pt"], data["train"]["jet_sdmass"], config)
    args.reweight_source = str(source)
    print(f"flat reweighting: {source}", flush=True)
    weights = {s: hp.event_weights(membership[s], data[s]["jet_pt"], data[s]["jet_sdmass"], hists, config)
               for s in ("train", "val")}
    ess = hp.effective_sample_size(weights["train"])
    samples_per_epoch = int(args.samples_per_epoch or max(ess, 1))
    print(f"train weights: mean {weights['train'].mean():.4f}, effective sample size {ess:.0f}; "
          f"{samples_per_epoch} samples per epoch", flush=True)
    if min(weights["train"].sum(), weights["val"].sum()) <= 0:
        p.error("All sampling weights are zero; check the reweighting inputs")

    args.output.mkdir(parents=True, exist_ok=True)
    kwargs = dict(data_config=str(args.data_config))
    labels = {s: data[s]["label"].astype(np.int64) for s in data}

    def chunks_from(probabilities):
        for start in range(0, len(labels["test"]), args.eval_batch):
            yield labels["test"][start:start + args.eval_batch], probabilities(slice(start, start + args.eval_batch))

    start = time.time()
    reference_metrics = hp.evaluate_chunks(chunks_from(lambda index: hp.softmax(data["test"]["logits"][index])),
                                           args.output / "parT_head", network_config="parT_head", **kwargs)
    results = {"parT_head": [hp.summarise(reference_metrics)]}
    # Weighted validation cross-entropy of the checkpoint's own logits: where every residual head starts.
    results["parT_head"][0]["val_loss"] = hp.weighted_cross_entropy(data["val"]["logits"], labels["val"], weights["val"])
    print(f"parT_head: hadronic rej50 {results['parT_head'][0]['rej50_geomean_hadronic']:.3f} "
          f"({time.time() - start:.0f}s)", flush=True)

    embedding_scaler = hp.FeatureScaler(np.zeros(data["train"]["embedding"].shape[1], dtype=bool))
    embedding_scaler.fit(data["train"]["embedding"], weights["train"])
    observable_scaler = hp.FeatureScaler([log for _, _, log in names]).fit(data["train"]["obs"], weights["train"])
    scaled, bases = {}, {}
    for s, d in data.items():   # keep one scaled copy and the logits; drop the raw features
        scaled[s] = (embedding_scaler.transform(d.pop("embedding")), observable_scaler.transform(d.pop("obs")))
        bases[s] = np.ascontiguousarray(d.pop("logits"), dtype=np.float32)
    fit = fit_torch if args.backend == "torch" else fit_sklearn
    training = {}
    for name, (uses_embedding, arm_groups) in arms.items():
        columns = hp.select_columns(names, arm_groups) if arm_groups else np.zeros(len(names), dtype=bool)
        residual = args.residual and uses_embedding

        def split(s):
            parts = ([scaled[s][0]] if uses_embedding else []) + ([scaled[s][1][:, columns]] if columns.any() else [])
            return SimpleNamespace(x=np.ascontiguousarray(np.concatenate(parts, 1)), y=labels[s],
                                   base=bases[s], w=weights.get(s), samples_per_epoch=samples_per_epoch)

        train, val, test = split("train"), split("val"), split("test")
        results[name], training[name] = [], []
        for seed in args.seeds:
            start = time.time()
            predictions, info = fit(train, val, test, args, seed, residual)
            metrics = hp.evaluate_chunks(predictions, args.output / name / f"seed{seed}",
                                         network_config=f"head:{name}", seed=seed, **kwargs)
            results[name].append(dict(hp.summarise(metrics), val_loss=info["val_loss_best"]))
            training[name].append(info)
            progress = "" if info["val_loss_best"] is None else (
                f", val loss {info['val_loss_initial']:.4f} -> {info['val_loss_best']:.4f}"
                + (" (never improved)" if info["best_epoch"] < 0 else f" at epoch {info['best_epoch'] + 1}"))
            print(f"{name} seed {seed}: hadronic rej50 {results[name][-1]['rej50_geomean_hadronic']:.3f}, "
                  f"{train.x.shape[1]} inputs{progress} ({time.time() - start:.0f}s)", flush=True)
        del predictions
        train = val = test = None      # release this arm's arrays before building the next

    table = hp.compare(results, REFERENCE)
    summary = dict(reference=REFERENCE, table=table, runs=results, training=training,
                   settings={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                   sample=dict(train=len(labels["train"]), val=len(labels["val"]), test=len(labels["test"]),
                               train_effective_sample_size=ess, observables=[list(n) for n in names]))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=1))
    print("\nReference row shows values; other rows show the relative change against it, "
          "with the spread across seeds.\nval_loss is the weighted validation cross-entropy (lower is better).\n")
    print(hp.format_table(table, REFERENCE))
    print(f"\nSaved {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
