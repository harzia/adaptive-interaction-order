from __future__ import annotations

import time
from collections import Counter, defaultdict

import awkward as ak
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

mpl.use("Agg")

from weaver.utils.data.tools import _concat
from weaver.utils.logger import _logger
from weaver.utils.nn.tools import (
    AllGather,
    _flatten_label,
    _flatten_preds,
    get_autocast_config,
    train_classification,
)


from aio.models.jets.particle_transformer import ParticleTransformer


class ParticleTransformerWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.export_embed = kwargs.pop("export_embed", False)
        self.mod = ParticleTransformer(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"mod.cls_token"}

    def forward(self, points, features, lorentz_vectors, mask):
        del points

        if not self.export_embed:
            return self.mod(
                features,
                v=lorentz_vectors,
                mask=mask,
            )

        x, padding_mask = self.mod._forward_encoder(
            features,
            v=lorentz_vectors,
            mask=mask,
        )
        x_cls = self.mod._forward_aggregator(
            x,
            padding_mask,
        )

        if self.mod.fc is None:
            return x_cls

        output = self.mod.fc(x_cls)

        if self.mod.for_inference:
            output = torch.softmax(output, dim=1)

        return torch.cat([output, x_cls], dim=1)


def get_model(data_config, **kwargs):
    cfg = dict(
        input_dim=len(
            data_config.input_dicts["pf_features"]
        ),
        num_classes=None,

        # Sophon-aligned ParticleTransformer configuration.
        pair_input_dim=4,
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

        # Misc.
        trim=True,
        for_inference=False,
    )

    # Weaver network options, e.g.
    #   -o num_classes 188
    #   -o fc_params "[(512,0.1)]"
    cfg.update(**kwargs)

    if cfg["num_classes"] is None:
        raise ValueError(
            "num_classes must be provided, e.g. "
            "`-o num_classes 188` for JetClass-II."
        )

    _logger.info(
        "Model config: %s",
        str(cfg),
    )

    model = ParticleTransformerWrapper(**cfg)

    model_info = {
        "input_names": list(
            data_config.input_names
        ),
        "input_shapes": {
            k: ((1,) + s[1:])
            for k, s
            in data_config.input_shapes.items()
        },
        "output_names": ["softmax"],
        "dynamic_axes": {
            **{
                k: {
                    0: "N",
                    2: "n_" + k.split("_")[0],
                }
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
    del data_config, kwargs
    return train_classification


def get_evaluate_fn(data_config, **kwargs):
    del data_config, kwargs
    return evaluate_classification_lean


def _log_validation_health(
    confusion,
    epoch,
    tb_helper=None,
):
    if confusion is None:
        return

    num_classes = confusion.shape[0]

    truth_counts = confusion.sum(dim=1)
    pred_counts = confusion.sum(dim=0)

    present_truth = truth_counts > 0
    present_pred = pred_counts > 0

    num_truth_classes = int(
        present_truth.sum().item()
    )
    num_predicted_classes = int(
        present_pred.sum().item()
    )

    row_denom = truth_counts.clamp_min(1)
    per_class_recall = (
        confusion.diag().float()
        / row_denom.float()
    )

    present_recall = per_class_recall[
        present_truth
    ]

    if len(present_recall):
        min_recall = float(
            present_recall.min().item()
        )
        mean_recall = float(
            present_recall.mean().item()
        )
        max_recall = float(
            present_recall.max().item()
        )
    else:
        min_recall = float("nan")
        mean_recall = float("nan")
        max_recall = float("nan")

    dominant_fraction = float(
        pred_counts.max().item()
        / max(1, pred_counts.sum().item())
    )

    _logger.info(
        "Validation health: truth classes=%d/%d, "
        "predicted classes=%d/%d, "
        "per-class recall min/mean/max="
        "%.5f/%.5f/%.5f, "
        "largest predicted-class fraction=%.5f",
        num_truth_classes,
        num_classes,
        num_predicted_classes,
        num_classes,
        min_recall,
        mean_recall,
        max_recall,
        dominant_fraction,
    )

    if num_predicted_classes <= 1:
        _logger.warning(
            "Model-collapse warning: validation predictions contain "
            "only %d predicted class.",
            num_predicted_classes,
        )
    elif dominant_fraction > 0.95:
        _logger.warning(
            "Model-collapse warning: %.2f%% of validation predictions "
            "belong to one class.",
            100.0 * dominant_fraction,
        )

    if tb_helper:
        tb_helper.write_scalars(
            [
                (
                    "Health/eval_predicted_classes",
                    num_predicted_classes,
                    epoch,
                ),
                (
                    "Health/eval_truth_classes",
                    num_truth_classes,
                    epoch,
                ),
                (
                    "Health/eval_recall_min",
                    min_recall,
                    epoch,
                ),
                (
                    "Health/eval_recall_mean",
                    mean_recall,
                    epoch,
                ),
                (
                    "Health/eval_recall_max",
                    max_recall,
                    epoch,
                ),
                (
                    "Health/eval_dominant_pred_fraction",
                    dominant_fraction,
                    epoch,
                ),
            ]
        )

        normalized = (
            confusion.float()
            / row_denom[:, None].float()
        ).numpy()

        fig, ax = plt.subplots(
            figsize=(10, 9)
        )
        image = ax.imshow(
            normalized,
            aspect="auto",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(
            "Validation confusion matrix "
            "(row-normalized)"
        )
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Truth class")
        fig.colorbar(
            image,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )
        fig.tight_layout()

        tb_helper.writer.add_figure(
            "Health/eval_confusion_matrix",
            fig,
            global_step=epoch,
        )
        plt.close(fig)


def evaluate_classification_lean(
    model,
    test_loader,
    dev,
    epoch,
    for_training=True,
    loss_func=None,
    steps_per_epoch=None,
    eval_metrics=None,
    tb_helper=None,
    extra_args=None,
):
    del eval_metrics

    model.eval()
    data_config = test_loader.dataset.config

    label_counter = Counter()
    total_loss = 0.0
    num_batches = 0
    total_correct = 0
    entry_count = 0
    count = 0
    confusion = None
    num_classes = None

    scores = []
    labels = defaultdict(list)
    labels_counts = []
    observers = defaultdict(list)

    enable_autocast, autocast_dtype = (
        get_autocast_config(
            extra_args["args"]
        )
    )

    start_time = time.time()

    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y, Z in tq:
                inputs = [
                    X[k].to(dev)
                    for k in data_config.input_names
                ]

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
                        data_config.label_names[0]
                        + "_mask"
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

                logits, label, mask = (
                    _flatten_preds(
                        model_output,
                        label=label,
                        mask=mask,
                    )
                )

                if not torch.isfinite(
                    logits
                ).all():
                    num_nan = int(
                        torch.isnan(
                            logits
                        ).sum().item()
                    )
                    num_inf = int(
                        torch.isinf(
                            logits
                        ).sum().item()
                    )
                    raise RuntimeError(
                        "Non-finite logits detected during "
                        f"{'validation' if for_training else 'testing'} "
                        f"at epoch {epoch}, batch {num_batches + 1}: "
                        f"{num_nan} NaN, {num_inf} Inf values."
                    )

                if num_classes is None:
                    num_classes = int(
                        logits.shape[1]
                    )
                    confusion = torch.zeros(
                        (
                            num_classes,
                            num_classes,
                        ),
                        dtype=torch.int64,
                    )
                elif logits.shape[1] != num_classes:
                    raise RuntimeError(
                        "Classifier output dimension changed during "
                        f"evaluation: expected {num_classes}, "
                        f"got {logits.shape[1]}."
                    )

                _, preds = logits.max(1)

                indices = (
                    label
                    * num_classes
                    + preds
                )

                confusion += (
                    torch.bincount(
                        indices,
                        minlength=(
                            num_classes
                            * num_classes
                        ),
                    )
                    .reshape(
                        num_classes,
                        num_classes,
                    )
                    .cpu()
                )

                if not for_training:
                    probs = torch.softmax(
                        logits.float(),
                        dim=1,
                    )

                    if not torch.isfinite(
                        probs
                    ).all():
                        raise RuntimeError(
                            "Non-finite probabilities detected during "
                            f"testing at epoch {epoch}, "
                            f"batch {num_batches + 1}."
                        )

                    scores.append(
                        probs.numpy(force=True)
                    )

                    mask_cpu = (
                        mask.cpu()
                        if mask is not None
                        else None
                    )

                    for k, v in y.items():
                        labels[k].append(
                            _flatten_label(
                                v,
                                mask_cpu,
                            ).numpy(force=True)
                        )

                    for k, v in Z.items():
                        observers[k].append(v)

                    if mask_cpu is not None:
                        labels_counts.append(
                            np.squeeze(
                                mask_cpu.numpy(
                                    force=True
                                ).sum(axis=-1)
                            )
                        )

                num_examples = label.shape[0]

                label_counter.update(
                    label.numpy(force=True)
                )

                if loss_func is None:
                    loss = 0.0
                else:
                    loss_tensor = loss_func(
                        logits,
                        label,
                    )

                    if not torch.isfinite(
                        loss_tensor
                    ).all():
                        raise RuntimeError(
                            "Non-finite evaluation loss detected during "
                            f"{'validation' if for_training else 'testing'} "
                            f"at epoch {epoch}, batch {num_batches + 1}: "
                            f"{loss_tensor.detach().cpu()}."
                        )

                    loss = loss_tensor.item()

                correct = (
                    preds == label
                ).sum().item()

                num_batches += 1
                count += num_examples
                total_loss += (
                    loss * num_examples
                )
                total_correct += correct

                tq.set_postfix(
                    {
                        "Loss": "%.5f" % loss,
                        "AvgLoss": "%.5f"
                        % (total_loss / count),
                        "Acc": "%.5f"
                        % (
                            correct
                            / num_examples
                        ),
                        "AvgAcc": "%.5f"
                        % (
                            total_correct
                            / count
                        ),
                    }
                )

                if (
                    tb_helper
                    and tb_helper.custom_fn
                ):
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
                    and num_batches
                    >= steps_per_epoch
                ):
                    break

    if count == 0:
        raise RuntimeError(
            "Evaluation processed zero examples."
        )

    time_diff = time.time() - start_time

    _logger.info(
        "Processed %d entries in total "
        "(avg. speed %.1f entries/s)",
        entry_count,
        entry_count / time_diff,
    )

    _logger.info(
        "Eval AvgLoss: %.5f, "
        "AvgAcc: %.5f",
        total_loss / count,
        total_correct / count,
    )

    _logger.info(
        "Evaluation class distribution: "
        "\n    %s",
        str(
            sorted(
                label_counter.items()
            )
        ),
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
                    "Loss/%s (epoch)"
                    % tb_mode,
                    total_loss / count,
                    epoch,
                ),
                (
                    "Acc/%s (epoch)"
                    % tb_mode,
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

    accuracy = total_correct / count

    if for_training:
        _log_validation_health(
            confusion=confusion,
            epoch=epoch,
            tb_helper=tb_helper,
        )
        return accuracy

    scores = np.concatenate(scores)

    labels = {
        k: _concat(v)
        for k, v in labels.items()
    }

    if len(scores) != entry_count:
        if len(labels_counts):
            labels_counts = (
                np.concatenate(
                    labels_counts
                )
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
                    int(
                        count
                        / entry_count
                    ),
                    -1,
                )
            ).transpose((1, 2))

            for k, v in labels.items():
                labels[k] = v.reshape(
                    (
                        entry_count,
                        -1,
                    )
                )

    observers = {
        k: _concat(v)
        for k, v in observers.items()
    }

    return (
        accuracy,
        scores,
        labels,
        observers,
    )
