"""Fit on fit data, choose epochs on validation, then report held-out outcomes."""
from __future__ import annotations

import copy
import numpy as np
import torch

from .frozen_experts import ConstantGate, InputGate, routed_probabilities, true_class_losses


def feature_scaler(cache):
    total = np.zeros(cache.features.shape[1], dtype=np.float64)
    squares = total.copy()
    for start in range(0, len(cache), 4096):
        x = np.asarray(cache.features[start:start+4096], dtype=np.float64)
        total += x.sum(0)
        squares += (x * x).sum(0)
    mean = total / len(cache)
    scale = np.maximum(squares / len(cache) - mean * mean, 1e-12) ** .5
    return mean.astype(np.float32), scale.astype(np.float32)


@torch.no_grad()
def gate_validation_loss(gate, cache, *, hard, batch_size, device):
    gate.eval()
    total = 0.
    for x, p, y in cache.batches(batch_size, device=device):
        output = routed_probabilities(p, gate(x), hard=hard)
        loss = -output.gather(1, y[:, None]).clamp_min(1e-12).log().sum()
        total += float(loss)
    return total / len(cache)


def fit_gate(fit, validation, *, kind, hidden=16, epochs=40, batch_size=1024,
             learning_rate=.003, seed=123, device="cpu", patience=8):
    """hard minimizes expected expert CE, soft/constant minimize mixture CE.

    Hard routing selects its checkpoint by actual argmax-dispatch validation CE.
    There is intentionally no test-cache argument to this function.
    """
    if kind not in ("constant", "soft", "hard"):
        raise ValueError("Unknown gate kind")
    if min(epochs, batch_size, patience) < 1 or hidden < 0 or learning_rate <= 0:
        raise ValueError("Invalid fitting options")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mean, scale = feature_scaler(fit)
    gate = (ConstantGate(fit.probabilities.shape[1]) if kind == "constant"
            else InputGate(mean, scale, fit.probabilities.shape[1], hidden)).to(device)
    optimizer = torch.optim.Adam(gate.parameters(), lr=learning_rate)
    best_loss = gate_validation_loss(gate, validation, hard=kind == "hard", batch_size=batch_size, device=device)
    best_state, best_epoch, stale = copy.deepcopy(gate.state_dict()), 0, 0
    history = [dict(epoch=0, validation_ce=best_loss)]
    for epoch in range(1, epochs + 1):
        gate.train()
        total = 0.
        for x, p, y in fit.batches(batch_size, order=rng.permutation(len(fit)), device=device):
            optimizer.zero_grad(set_to_none=True)
            weights = gate(x)
            if kind == "hard":
                loss = (weights * true_class_losses(p, y)).sum(-1).mean()
            else:
                probabilities = routed_probabilities(p, weights)
                loss = -probabilities.gather(1, y[:, None]).clamp_min(1e-12).log().mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite gate objective")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(x)
        validation_loss = gate_validation_loss(gate, validation, hard=kind == "hard", batch_size=batch_size, device=device)
        history.append(dict(epoch=epoch, train_objective=total / len(fit), validation_ce=validation_loss))
        if validation_loss < best_loss - 1e-7:
            best_state, best_loss, best_epoch, stale = copy.deepcopy(gate.state_dict()), validation_loss, epoch, 0
        else:
            stale += 1
        if stale >= patience:
            break
    gate.load_state_dict(best_state)
    gate.eval()
    return gate, dict(kind=kind, best_epoch=best_epoch, validation_ce=best_loss, history=history,
                      parameters=sum(p.numel() for p in gate.parameters()))


def paired_difference(first, second):
    """Conditional-on-checkpoints paired CE difference, not training-seed uncertainty."""
    delta = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    mean = float(delta.mean())
    se = float(delta.std(ddof=1) / np.sqrt(len(delta))) if len(delta) > 1 else None
    return dict(mean_ce_delta=mean, standard_error=se,
                normal_95_interval=[mean - 1.96 * se, mean + 1.96 * se] if se is not None else None,
                first_lower_ce_fraction=float((delta < -1e-7).mean()),
                second_lower_ce_fraction=float((delta > 1e-7).mean()),
                tie_fraction=float((abs(delta) <= 1e-7).mean()))


@torch.no_grad()
def evaluate_gates(cache, gates, *, batch_size=1024, device="cpu"):
    names = ["expert:" + e["name"] for e in cache.metadata["experts"]] + ["uniform", "constant", "soft", "hard"]
    e, n = len(cache.metadata["experts"]), len(cache)
    losses = np.empty((n, len(names)), dtype=np.float32)
    correct = np.empty((n, len(names)), dtype=bool)
    choices = np.empty(n, dtype=np.int64)
    soft_weights = np.zeros(e, dtype=np.float64)
    disagreement = np.zeros((e, e), dtype=np.int64)
    offset = 0
    for x, p, y in cache.batches(batch_size, device=device):
        weights = {name: gate(x) for name, gate in gates.items()}
        outputs = [p[:, index] for index in range(e)] + [p.mean(1)] + [
            routed_probabilities(p, weights[name], hard=name == "hard") for name in ("constant", "soft", "hard")]
        end = offset + len(x)
        for index, output in enumerate(outputs):
            losses[offset:end, index] = -output.gather(1, y[:, None]).squeeze(1).clamp_min(1e-12).log().cpu().numpy()
            correct[offset:end, index] = (output.argmax(-1) == y).cpu().numpy()
        predictions = p.argmax(-1)
        disagreement += (predictions[:, :, None] != predictions[:, None, :]).sum(0).cpu().numpy()
        choices[offset:end] = weights["hard"].argmax(-1).cpu().numpy()
        soft_weights += weights["soft"].sum(0).cpu().numpy()
        offset = end
    expert_losses = losses[:, :e]
    oracle_loss = expert_losses.min(1)
    # Ties are shared instead of incorrectly assigning all ties to expert zero.
    oracle_ties = np.isclose(expert_losses, oracle_loss[:, None], atol=1e-7, rtol=0)
    oracle_fraction = (oracle_ties / oracle_ties.sum(1, keepdims=True)).mean(0)
    report = dict(num_events=n, methods={name: dict(ce=float(losses[:, i].mean(dtype=np.float64)),
                accuracy=float(correct[:, i].mean())) for i, name in enumerate(names)},
                diagnostic_only_oracle=dict(ce=float(oracle_loss.mean(dtype=np.float64)),
                    expert_fraction=oracle_fraction.tolist(),
                    note="True-label selection: optimistic CE bound for selecting ONE frozen expert. No ROC/accuracy claim; not a bound for mixtures."),
                expert_disagreement_fraction=(disagreement / n).tolist(),
                expert_pair_differences={f"{names[i]} minus {names[j]}": paired_difference(losses[:, i], losses[:, j])
                    for i in range(e) for j in range(i+1, e)},
                hard_route_fraction=(np.bincount(choices, minlength=e) / n).tolist(),
                mean_soft_weights=(soft_weights / n).tolist(),
                constant_weights=gates["constant"].logits.softmax(-1).cpu().tolist(),
                comparisons={name + " minus constant": paired_difference(losses[:, names.index(name)], losses[:, names.index("constant")])
                    for name in ("uniform", "soft", "hard")},
                uncertainty_note="Paired normal CE intervals condition on these checkpoints and approximately independent events; no training-seed variance or multiplicity correction.")
    groups = {name: [i] for i, name in enumerate(cache.metadata["class_names"])}
    if len(groups) == 188:
        from integrations.weaver.eval.jetclass2_eval import CLASS_NAMES, SIGNAL_GROUPS, QCD_INDICES
        if cache.metadata["class_names"] == CLASS_NAMES:
            groups.update({"group:" + k: v for k, v in SIGNAL_GROUPS.items()})
            groups["group:QCD"] = QCD_INDICES.tolist()
    report["groups"] = {}
    for name, labels in groups.items():
        selected = np.isin(cache.labels, labels)
        count = int(selected.sum())
        if not count:
            continue
        report["groups"][name] = dict(num_events=count,
            ce={method: float(losses[selected, i].mean(dtype=np.float64)) for i, method in enumerate(names)},
            hard_route_fraction=(np.bincount(choices[selected], minlength=e) / count).tolist(),
            oracle_ce_diagnostic_only=float(oracle_loss[selected].mean(dtype=np.float64)))
    return report, losses, choices, names


def export_jetclass2_metrics(cache, gates, output, *, batch_size=1024, device="cpu", seed=123):
    """Same ROOT evaluator as the training runs; intentionally NO oracle export."""
    import uproot
    from types import SimpleNamespace
    from integrations.weaver.eval.jetclass2_eval import CLASS_NAMES, evaluate_prediction_files
    if cache.metadata["class_names"] != CLASS_NAMES:
        raise ValueError("JetClass-II metrics require its exact 188-class output order")
    methods = [(f"expert_{i}", i) for i in range(len(cache.metadata["experts"]))] + [(k, None) for k in ("uniform", "constant", "soft", "hard")]
    for name, expert_index in methods:
        folder = output / name
        folder.mkdir()
        path = folder / "predictions.root"
        with uproot.recreate(path) as root:
            tree = root.mktree("Events", {"truth_label": "int64", "output": np.dtype((np.float32, (188,)))})
            with torch.no_grad():
                for x, p, y in cache.batches(batch_size, device=device):
                    if expert_index is not None:
                        probabilities = p[:, expert_index]
                    elif name == "uniform":
                        probabilities = p.mean(1)
                    else:
                        probabilities = routed_probabilities(p, gates[name](x), hard=name == "hard")
                    tree.extend(dict(truth_label=y.cpu().numpy(), output=probabilities.cpu().numpy()))
        evaluate_prediction_files(prediction_files={name: str(path)}, args=SimpleNamespace(seed=seed))
