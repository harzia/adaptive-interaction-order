from __future__ import annotations

import time
from collections import Counter, defaultdict

import awkward as ak
import matplotlib as mpl
import numpy as np
import sklearn.metrics as sk_metrics
import torch
import tqdm

mpl.use("Agg")
import matplotlib.pyplot as plt

from weaver.utils.data.tools import _concat
from weaver.utils.logger import _logger
from weaver.utils.nn.metrics import evaluate_metrics
from weaver.utils.nn.tools import (
    AllGather,
    _flatten_label,
    _flatten_preds,
    get_autocast_config,
    train_classification,
)

# Actual model implementation from this repository.
from aio.models.jets.particle_transformer import ParticleTransformer


class ParticleTransformerSophonWrapper(torch.nn.Module):
    """Particle Transformer wrapper for JetClass-II / Sophon-style training."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.export_embed = kwargs.pop("export_embed", False)
        self.mod = ParticleTransformer(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"mod.cls_token"}

    def forward(self, points, features, lorentz_vectors, mask):
        return self.mod(features, v=lorentz_vectors, mask=mask)


def get_model(data_config, **kwargs):
    cfg = dict(
        input_dim=len(data_config.input_dicts["pf_features"]),
        num_classes=None,
        # network configuration
        use_pre_activation_pair=True,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={
            "dropout": 0,
            "attn_dropout": 0,
            "activation_dropout": 0,
        },
        fc_params=[],
        activation="gelu",

        # misc
        trim=True,
        for_inference=False,
    )

    # Weaver network options, e.g.:
    #   -o num_classes 188
    cfg.update(**kwargs)

    if cfg["num_classes"] is None:
        raise ValueError(
            "num_classes must be provided, e.g. "
            "`-o num_classes 188` for JetClass-II."
        )

    _logger.info("Model config: %s", str(cfg))

    model = ParticleTransformerSophonWrapper(**cfg)

    model_info = {
        "input_names": list(data_config.input_names),
        "input_shapes": {
            k: ((1,) + s[1:])
            for k, s in data_config.input_shapes.items()
        },
        "output_names": ["softmax"],
        "dynamic_axes": {
            **{
                k: {0: "N", 2: "n_" + k.split("_")[0]}
                for k in data_config.input_names
            },
            "softmax": {0: "N"},
        },
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    del data_config, kwargs
    return torch.nn.CrossEntropyLoss()


def get_train_fn(data_config, **kwargs):
    """Use Weaver's current default classification training loop."""
    del data_config, kwargs
    return train_classification


def get_evaluate_fn(data_config, **kwargs):
    """Use current Weaver evaluation plus Sophon ROC monitoring."""
    del data_config, kwargs
    return evaluate_classification_sophon


def evaluate_classification_sophon(
    model,
    test_loader,
    dev,
    epoch,
    for_training=True,
    loss_func=None,
    steps_per_epoch=None,
    eval_metrics=[
        "roc_auc_score",
        "roc_auc_score_matrix",
        "confusion_matrix",
    ],
    tb_helper=None,
    extra_args=None,
):
    """
    Current Weaver classification evaluation with the original Sophon
    Xbb/Xcc/QCD ROC and background-rejection monitoring added.
    """
    model.eval()

    data_config = test_loader.dataset.config

    label_counter = Counter()
    total_loss = 0
    num_batches = 0
    total_correct = 0
    entry_count = 0
    count = 0
    scores = []
    labels = defaultdict(list)
    labels_counts = []
    observers = defaultdict(list)

    enable_autocast, autocast_dtype = get_autocast_config(
        extra_args["args"]
    )

    start_time = time.time()

    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y, Z in tq:
                # X, y: torch.Tensor; Z: ak.Array
                inputs = [
                    X[k].to(dev)
                    for k in data_config.input_names
                ]

                # Match current Weaver DDP evaluation behavior.
                y = {
                    k: AllGather.apply(v.to(dev))
                    for k, v in y.items()
                }

                label = y[
                    data_config.label_names[0]
                ].long().to(dev)
                entry_count += label.shape[0]

                try:
                    mask = y[
                        data_config.label_names[0] + "_mask"
                    ].bool().to(dev)
                except KeyError:
                    mask = None

                with torch.autocast(
                    "cuda",
                    enabled=enable_autocast,
                    dtype=autocast_dtype,
                ):
                    model_output = AllGather.apply(
                        model(*inputs)
                    )

                logits, label, mask = _flatten_preds(
                    model_output,
                    label=label,
                    mask=mask,
                )

                scores.append(
                    torch.softmax(
                        logits.float(),
                        dim=1,
                    ).numpy(force=True)
                )

                if mask is not None:
                    mask = mask.cpu()

                for k, v in y.items():
                    labels[k].append(
                        _flatten_label(
                            v,
                            mask,
                        ).numpy(force=True)
                    )

                if not for_training:
                    for k, v in Z.items():
                        observers[k].append(v)

                num_examples = label.shape[0]
                label_counter.update(
                    label.numpy(force=True)
                )

                if (
                    not for_training
                    and mask is not None
                ):
                    labels_counts.append(
                        np.squeeze(
                            mask.numpy(force=True).sum(
                                axis=-1
                            )
                        )
                    )

                _, preds = logits.max(1)

                loss = (
                    0
                    if loss_func is None
                    else loss_func(logits, label).item()
                )

                num_batches += 1
                count += num_examples
                correct = (
                    preds == label
                ).sum().item()
                total_loss += loss * num_examples
                total_correct += correct

                tq.set_postfix(
                    {
                        "Loss": "%.5f" % loss,
                        "AvgLoss": "%.5f" % (
                            total_loss / count
                        ),
                        "Acc": "%.5f" % (
                            correct / num_examples
                        ),
                        "AvgAcc": "%.5f" % (
                            total_correct / count
                        ),
                    }
                )

                if tb_helper:
                    if tb_helper.custom_fn:
                        with torch.no_grad():
                            tb_helper.custom_fn(
                                model_output=model_output,
                                model=model,
                                epoch=epoch,
                                i_batch=num_batches,
                                mode=(
                                    "eval"
                                    if for_training
                                    else "test"
                                ),
                            )

                if (
                    steps_per_epoch is not None
                    and num_batches >= steps_per_epoch
                ):
                    break

    time_diff = time.time() - start_time

    _logger.info(
        "Processed %d entries in total "
        "(avg. speed %.1f entries/s)",
        entry_count,
        entry_count / time_diff,
    )
    _logger.info(
        "Eval AvgLoss: %.5f, AvgAcc: %.5f",
        total_loss / count,
        total_correct / count,
    )
    _logger.info(
        "Evaluation class distribution: \n    %s",
        str(sorted(label_counter.items())),
    )

    if tb_helper:
        tb_mode = (
            "eval"
            if for_training
            else "test"
        )

        tb_helper.write_scalars(
            [
                (
                    "Loss/%s (epoch)" % tb_mode,
                    total_loss / count,
                    epoch,
                ),
                (
                    "Acc/%s (epoch)" % tb_mode,
                    total_correct / count,
                    epoch,
                ),
            ]
        )

        if tb_helper.custom_fn:
            with torch.no_grad():
                tb_helper.custom_fn(
                    model_output=model_output,
                    model=model,
                    epoch=epoch,
                    i_batch=-1,
                    mode=tb_mode,
                )

    scores = np.concatenate(scores)
    labels = {
        k: _concat(v)
        for k, v in labels.items()
    }

    metric_results = evaluate_metrics(
        labels[data_config.label_names[0]],
        scores,
        eval_metrics=eval_metrics,
    )

    _logger.info(
        "Evaluation metrics: \n%s",
        "\n".join(
            [
                "    - %s: \n%s" % (k, str(v))
                for k, v in metric_results.items()
            ]
        ),
    )

    # Sophon-specific monitoring:
    #   Xbb = label 0
    #   Xcc = label 1
    #   QCD = labels 161..187, with the QCD probability formed by summing
    #         output nodes 161:188.
    if tb_helper:
        _write_sophon_roc_monitoring(
            scores=scores,
            truth_labels=labels[
                data_config.label_names[0]
            ],
            tb_helper=tb_helper,
            epoch=epoch,
            mode=(
                "eval"
                if for_training
                else "test"
            ),
        )

    if for_training:
        return total_correct / count

    # Match current Weaver behavior for possible 2-D outputs.
    if len(scores) != entry_count:
        if len(labels_counts):
            labels_counts = np.concatenate(
                labels_counts
            )
            scores = ak.unflatten(
                scores,
                labels_counts,
            )
            for k, v in labels.items():
                labels[k] = ak.unflatten(
                    v,
                    labels_counts,
                )
        else:
            assert count % entry_count == 0
            scores = scores.reshape(
                (
                    entry_count,
                    int(count / entry_count),
                    -1,
                )
            ).transpose((1, 2))

            for k, v in labels.items():
                labels[k] = v.reshape(
                    (entry_count, -1)
                )

    observers = {
        k: _concat(v)
        for k, v in observers.items()
    }

    return (
        total_correct / count,
        scores,
        labels,
        observers,
    )


def _write_sophon_roc_monitoring(
    *,
    scores,
    truth_labels,
    tb_helper,
    epoch,
    mode,
):
    """
    Reproduce Sophon's lightweight Xbb/Xcc/QCD TensorBoard diagnostics.

    Background rejection is measured at 30% signal efficiency.
    """
    truth_labels = np.asarray(truth_labels)

    scores_dict = {
        "Xbb": scores[:, 0],
        "Xcc": scores[:, 1],
        "QCD": np.sum(
            scores[:, 161:188],
            axis=1,
        ),
    }

    flag_dict = {
        "Xbb": truth_labels == 0,
        "Xcc": truth_labels == 1,
        "QCD": (
            (truth_labels >= 161)
            & (truth_labels < 188)
        ),
    }

    comparisons = [
        ("Xbb", "QCD"),
        ("Xcc", "QCD"),
        ("Xcc", "Xbb"),
    ]

    background_rejections = {}

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.plot(
        np.linspace(0, 1, 1000),
        np.linspace(0, 1, 1000),
        linestyle="--",
        color="gray",
        label="Random guess",
    )

    for signal_name, background_name in comparisons:
        denominator = (
            scores_dict[signal_name]
            + scores_dict[background_name]
        )

        discriminant = np.divide(
            scores_dict[signal_name],
            denominator,
            out=np.zeros_like(
                denominator,
                dtype=np.float64,
            ),
            where=denominator != 0,
        )

        signal_scores = discriminant[
            flag_dict[signal_name]
        ]
        background_scores = discriminant[
            flag_dict[background_name]
        ]

        binary_labels = np.concatenate(
            [
                np.ones(
                    signal_scores.shape[0],
                    dtype=np.int8,
                ),
                np.zeros(
                    background_scores.shape[0],
                    dtype=np.int8,
                ),
            ]
        )
        binary_scores = np.concatenate(
            [
                signal_scores,
                background_scores,
            ]
        )

        fpr, tpr, _ = sk_metrics.roc_curve(
            binary_labels,
            binary_scores,
        )
        auc = sk_metrics.auc(fpr, tpr)

        ax.plot(
            tpr,
            fpr,
            label=(
                f"{signal_name} vs {background_name} "
                f"(AUC={auc:.4f})"
            ),
        )

        rejection = 1.0 / np.maximum(
            fpr,
            1e-10,
        )

        background_rejections[
            (signal_name, background_name)
        ] = float(
            np.interp(
                0.30,
                tpr,
                rejection,
            )
        )

    ax.legend()
    ax.set_xlabel(
        "True positive rate (signal eff.)",
        ha="right",
        x=1.0,
    )
    ax.set_ylabel(
        "False positive rate (BKG eff.)",
        ha="right",
        y=1.0,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(1e-4, 1)
    ax.set_yscale("log")

    tb_helper.writer.add_figure(
        "ROC/%s/epoch%s"
        % (
            mode,
            str(epoch).zfill(4),
        ),
        fig,
    )
    plt.close(fig)

    for signal_name, background_name in comparisons:
        tb_helper.write_scalars(
            [
                (
                    (
                        "BkgRej_%s_vs_%s/%s (epoch)"
                        % (
                            signal_name,
                            background_name,
                            mode,
                        )
                    ),
                    background_rejections[
                        (signal_name, background_name)
                    ],
                    epoch,
                )
            ]
        )