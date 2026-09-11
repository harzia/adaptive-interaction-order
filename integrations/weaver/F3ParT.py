from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from functools import partial
from pathlib import Path

import awkward as ak
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch.profiler import ProfilerActivity, profile, record_function

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


from aio.models.jets.particle_transformer_aio import AIOParticleTransformer


def _profiler_event_time_us(event, preferred: str, fallback: str) -> float:
    value = getattr(event, preferred, None)
    if value is None:
        value = getattr(event, fallback, 0.0)
    return float(value)


def _install_aio_profile_ranges(base_model):
    """
    Add record_function ranges at runtime, so profiling needs no edits to AIO
    source files.  The motif _chunk wrapper is also seen during activation-
    checkpoint recomputation in backward.
    """
    aio = base_model.mod
    originals = []

    def wrap(obj, attr: str, label: str):
        original = getattr(obj, attr)

        def wrapped(*args, **kwargs):
            with record_function(label):
                return original(*args, **kwargs)

        setattr(obj, attr, wrapped)
        originals.append((obj, attr, original))

    wrap(aio, "_prelude", "aio/prelude")
    wrap(aio, "_forward_encoder", "aio/encoder")
    wrap(aio, "_forward_aggregator", "aio/aggregator")

    if aio.fc is not None:
        wrap(aio.fc, "forward", "aio/classifier")

    for index, block in enumerate(aio.blocks, start=1):
        wrap(block, "forward", f"aio/block_{index}")

    wrap(aio.hob, "forward", "aio/f3_hob")
    wrap(aio.hob.factors, "forward", "aio/f3/factors")
    wrap(aio.hob.factors, "rbar_at", "aio/f3/rbar")
    wrap(aio.hob.motif, "forward", "aio/f3/motif")
    wrap(aio.hob.motif, "pools", "aio/f3/motif_pools")
    wrap(aio.hob.motif, "_chunk", "aio/f3/motif_chunk")
    wrap(aio.hob.motif, "contract", "aio/f3/motif_contract")
    wrap(aio.inj, "bias", "aio/f3/inject_bias")
    wrap(aio.inj, "message", "aio/f3/inject_message")

    return originals


def _restore_aio_profile_ranges(originals):
    for obj, attr, original in reversed(originals):
        setattr(obj, attr, original)


def _profile_f3_training_steps(
    *,
    base_model,
    model,
    loss_func,
    opt,
    scheduler,
    train_loader,
    dev,
    grad_scaler,
    extra_args,
    num_steps: int,
):
    """
    Profile a few real Weaver training steps.

    Outputs:
      f3_profile_rankN.json             Chrome/PyTorch trace
      f3_profile_rankN_summary.json     named-range totals + throughput/memory
      f3_profile_rankN_operators.txt    top low-level profiler operators
    """
    model.train()
    data_config = train_loader.dataset.config
    clip_grad_norm = getattr(opt, "_clip_grad_norm", float("inf"))
    enable_autocast, autocast_dtype = get_autocast_config(
        extra_args["args"]
    )

    distributed = (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )
    rank = torch.distributed.get_rank() if distributed else 0

    output_dir = Path(
        os.environ.get("AIO_PROFILE_DIR", "profiles/f3")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
        torch.cuda.reset_peak_memory_stats(dev)

    originals = _install_aio_profile_ranges(base_model)
    iterator = iter(train_loader)
    processed_entries = 0
    completed_steps = 0

    _logger.info(
        "AIO profiler: profiling %d real training steps on rank %d; "
        "output directory: %s",
        num_steps,
        rank,
        output_dir,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize(dev)
    wall_start = time.perf_counter()

    try:
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
            with_stack=False,
        ) as prof:
            for _ in range(num_steps):
                with record_function("weaver/data_wait"):
                    try:
                        X, y, _ = next(iterator)
                    except StopIteration:
                        break

                with record_function("weaver/h2d"):
                    inputs = [
                        X[name].to(dev)
                        for name in data_config.input_names
                    ]
                    label = y[
                        data_config.label_names[0]
                    ].long().to(dev)
                    try:
                        label_mask = y[
                            data_config.label_names[0] + "_mask"
                        ].bool().to(dev)
                    except KeyError:
                        label_mask = None

                processed_entries += int(label.shape[0])

                with record_function("weaver/zero_grad"):
                    opt.zero_grad()

                with record_function("weaver/forward_loss"):
                    with torch.autocast(
                        "cuda",
                        enabled=enable_autocast,
                        dtype=autocast_dtype,
                    ):
                        model_output = model(*inputs)
                        logits, flat_label, _ = _flatten_preds(
                            model_output,
                            label=label,
                            mask=label_mask,
                        )
                        loss = loss_func(logits, flat_label)

                if grad_scaler is None:
                    with record_function("weaver/backward"):
                        loss.backward()

                    with record_function("weaver/grad_clip"):
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm=clip_grad_norm,
                        )

                    with record_function("weaver/optimizer_step"):
                        opt.step()
                else:
                    with record_function("weaver/backward"):
                        grad_scaler.scale(loss).backward()

                    with record_function("weaver/grad_clip"):
                        grad_scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm=clip_grad_norm,
                        )

                    with record_function("weaver/optimizer_step"):
                        grad_scaler.step(opt)
                        grad_scaler.update()

                if (
                    scheduler
                    and getattr(
                        scheduler,
                        "_update_per_step",
                        False,
                    )
                ):
                    with record_function("weaver/scheduler_step"):
                        scheduler.step()

                completed_steps += 1
                prof.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize(dev)
        wall_seconds = time.perf_counter() - wall_start

        trace_path = (
            output_dir / f"f3_profile_rank{rank}.json"
        )
        prof.export_chrome_trace(str(trace_path))

        averages = prof.key_averages(
            group_by_input_shape=False
        )

        ranges = []
        for event in averages:
            if not (
                event.key.startswith("aio/")
                or event.key.startswith("weaver/")
            ):
                continue

            ranges.append(
                {
                    "name": event.key,
                    "calls": int(event.count),
                    "cpu_total_ms": (
                        float(event.cpu_time_total) / 1000.0
                    ),
                    "cpu_self_ms": (
                        float(event.self_cpu_time_total) / 1000.0
                    ),
                    "device_total_ms": (
                        _profiler_event_time_us(
                            event,
                            "device_time_total",
                            "cuda_time_total",
                        )
                        / 1000.0
                    ),
                    "device_self_ms": (
                        _profiler_event_time_us(
                            event,
                            "self_device_time_total",
                            "self_cuda_time_total",
                        )
                        / 1000.0
                    ),
                }
            )

        ranges.sort(
            key=lambda row: row["device_total_ms"],
            reverse=True,
        )

        peak_cuda_memory_mb = None
        if torch.cuda.is_available():
            peak_cuda_memory_mb = float(
                torch.cuda.max_memory_allocated(dev)
                / 1024.0**2
            )

        summary = {
            "rank": int(rank),
            "requested_steps": int(num_steps),
            "completed_steps": int(completed_steps),
            "processed_entries": int(processed_entries),
            "wall_seconds": float(wall_seconds),
            "seconds_per_step": (
                float(wall_seconds / completed_steps)
                if completed_steps
                else None
            ),
            "entries_per_second": (
                float(processed_entries / wall_seconds)
                if wall_seconds > 0
                else None
            ),
            "peak_cuda_memory_mb": peak_cuda_memory_mb,
            "note": (
                "Named ranges are nested totals; do not sum them. "
                "Profiler instrumentation adds overhead, so use these "
                "primarily for relative attribution."
            ),
            "ranges": ranges,
        }

        summary_path = (
            output_dir
            / f"f3_profile_rank{rank}_summary.json"
        )
        summary_path.write_text(
            json.dumps(summary, indent=2)
        )

        sort_key = (
            "self_cuda_time_total"
            if torch.cuda.is_available()
            else "self_cpu_time_total"
        )
        operator_table = averages.table(
            sort_by=sort_key,
            row_limit=80,
        )

        operators_path = (
            output_dir
            / f"f3_profile_rank{rank}_operators.txt"
        )
        operators_path.write_text(operator_table)

        _logger.info(
            "AIO profiler top operators:\n%s",
            operator_table,
        )
        _logger.info(
            "AIO profiler named ranges "
            "(nested totals; do not sum):\n%s",
            "\n".join(
                (
                    f"  {row['name']:<30} "
                    f"calls={row['calls']:<5d} "
                    f"device_total="
                    f"{row['device_total_ms']:.3f} ms "
                    f"cpu_total="
                    f"{row['cpu_total_ms']:.3f} ms"
                )
                for row in ranges
            ),
        )
        _logger.info(
            "AIO profiler: %d steps, %.2f s wall, "
            "%.3f s/step, %.2f entries/s, "
            "peak CUDA memory=%s MB",
            completed_steps,
            wall_seconds,
            (
                wall_seconds / completed_steps
                if completed_steps
                else float("nan")
            ),
            (
                processed_entries / wall_seconds
                if wall_seconds > 0
                else float("nan")
            ),
            (
                f"{peak_cuda_memory_mb:.1f}"
                if peak_cuda_memory_mb is not None
                else "n/a"
            ),
        )
        _logger.info(
            "AIO profiler trace: %s",
            trace_path,
        )
        _logger.info(
            "AIO profiler summary: %s",
            summary_path,
        )
        _logger.info(
            "AIO profiler operators: %s",
            operators_path,
        )

    finally:
        _restore_aio_profile_ranges(originals)




class AIOParticleTransformerWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.export_embed = kwargs.pop("export_embed", False)
        self.mod = AIOParticleTransformer(**kwargs)

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

    @torch.no_grad()
    def calibrate(self, points, features, lorentz_vectors, mask, target=1.0, force=False):
        del points

        return self.mod.calibrate(
            features,
            v=lorentz_vectors,
            mask=mask,
            target=target,
            force=force,
        )


def get_model(data_config, **kwargs):
    kwargs.pop("auto_calibrate", None)
    kwargs.pop("calibration_target", None)
    
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

        f3_after=4,
        spec="triangle",
        D=32, H3=2, d_r=32,
        anchor_chunk=256,
        dense_matmul=False,
        use_router=False,
        diagnostics=False,

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

    model = AIOParticleTransformerWrapper(**cfg)

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
    del data_config
    auto_calibrate = bool(
        kwargs.get(
            "auto_calibrate",
            True,
        )
    )
    calibration_target = float(
        kwargs.get(
            "calibration_target",
            1.0,
        )
    )
    return partial(
        train_classification_with_f3_calibration,
        auto_calibrate=auto_calibrate,
        calibration_target=calibration_target,
    )


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

class _ReplayFirstBatchLoader:
    """
    Loader facade that replays the batch used for calibration and then
    continues from the already-created DataLoader iterator.

    This prevents calibration from consuming an extra training batch.
    """

    def __init__(
        self,
        first_batch,
        remaining_iterator,
        original_loader,
    ):
        self.first_batch = first_batch
        self.remaining_iterator = remaining_iterator
        self.original_loader = original_loader

        # Weaver's stock training function reads this.
        self.dataset = original_loader.dataset

    def __iter__(self):
        yield self.first_batch
        yield from self.remaining_iterator

    def __len__(self):
        return len(self.original_loader)


def _unwrap_model(model):
    while True:
        if hasattr(model, "module"):
            model = model.module
            continue

        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
            continue

        return model


def _format_calibration_report(report):
    formatted = {}

    for key, value in report.items():
        if torch.is_tensor(value):
            value = value.detach().cpu()

            if value.numel() == 1:
                formatted[key] = float(value.item())
            else:
                formatted[key] = value.tolist()
        else:
            formatted[key] = value

    return formatted


def train_classification_with_f3_calibration(
    model,
    loss_func,
    opt,
    scheduler,
    train_loader,
    dev,
    epoch,
    steps_per_epoch=None,
    grad_scaler=None,
    tb_helper=None,
    extra_args=None,
    *,
    auto_calibrate=True,
    calibration_target=1.0,
):
    """
    Weaver classification loop with a one-time F3 initialization
    calibration before the first optimizer step.

    After calibration, the stock Weaver train_classification function
    is used unchanged.
    """

    base_model = _unwrap_model(model)

    if not isinstance(
        base_model,
        AIOParticleTransformerWrapper,
    ):
        raise TypeError(
            "F3 calibration train hook expected "
            "AIOParticleTransformerWrapper, got "
            f"{type(base_model).__name__}."
        )

    hob = base_model.mod.hob

    if not auto_calibrate:
        return train_classification(
            model,
            loss_func,
            opt,
            scheduler,
            train_loader,
            dev,
            epoch,
            steps_per_epoch=steps_per_epoch,
            grad_scaler=grad_scaler,
            tb_helper=tb_helper,
            extra_args=extra_args,
        )

    distributed = (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )

    rank = (
        torch.distributed.get_rank()
        if distributed
        else 0
    )

    calibrated = bool(
        hob.calibrated.item()
    )

    if distributed:
        flag = torch.tensor(
            int(calibrated),
            device=dev,
            dtype=torch.int32,
        )

        torch.distributed.broadcast(
            flag,
            src=0,
        )

        calibrated = bool(flag.item())

    if calibrated:
        if rank == 0 and epoch == 0:
            _logger.info(
                "F3 calibration already present in checkpoint; "
                "skipping initialization calibration."
            )

        return train_classification(
            model,
            loss_func,
            opt,
            scheduler,
            train_loader,
            dev,
            epoch,
            steps_per_epoch=steps_per_epoch,
            grad_scaler=grad_scaler,
            tb_helper=tb_helper,
            extra_args=extra_args,
        )

    args = (
        extra_args.get("args")
        if extra_args is not None
        else None
    )

    load_epoch = (
        getattr(args, "load_epoch", None)
        if args is not None
        else None
    )

    if load_epoch is not None:
        raise RuntimeError(
            "Resumed F3 checkpoint has hob.calibrated=False. "
            "Refusing to recalibrate a resumed trajectory. "
            "Use a properly calibrated checkpoint or explicitly "
            "handle this legacy checkpoint."
        )

    if epoch != 0:
        raise RuntimeError(
            "Reached a later epoch with an uncalibrated F3 model. "
            "Calibration must occur before optimizer step 1."
        )

    train_iterator = iter(train_loader)

    try:
        first_batch = next(train_iterator)
    except StopIteration as exc:
        raise RuntimeError(
            "Cannot calibrate F3: training loader is empty."
        ) from exc

    X, _, _ = first_batch

    data_config = train_loader.dataset.config

    if rank == 0:
        _logger.info(
            "Running one-time F3 initialization calibration "
            "on the first real Weaver training batch."
        )

        calibration_inputs = [
            X[name].to(
                dev,
                non_blocking=True,
            )
            for name in data_config.input_names
        ]

        report = base_model.calibrate(
            *calibration_inputs,
            target=calibration_target,
        )

        del calibration_inputs

        _logger.info(
            "F3 calibration report: %s",
            _format_calibration_report(report),
        )

    if distributed:
        for parameter in hob.parameters():
            torch.distributed.broadcast(
                parameter.data,
                src=0,
            )

        for buffer in hob.buffers():
            torch.distributed.broadcast(
                buffer.data,
                src=0,
            )

        torch.distributed.barrier()

    if (
        hasattr(opt, "reset")
        and hasattr(opt, "step_counter")
        and hasattr(opt, "k")
        and hasattr(opt, "optimizer")
    ):
        opt.reset()
        if rank == 0:
            _logger.info("Reset Ranger/Lookahead cache after F3 calibration.")

    if not bool(hob.calibrated.item()):
        raise RuntimeError(
            "F3 calibration synchronization failed: "
            "hob.calibrated is still False."
        )

    replay_loader = _ReplayFirstBatchLoader(
        first_batch,
        train_iterator,
        train_loader,
    )

    profile_steps = int(
        os.environ.get("AIO_PROFILE_STEPS", "0")
    )
    if profile_steps > 0:
        if epoch != 0:
            raise RuntimeError(
                "AIO_PROFILE_STEPS is intended for an epoch-0 "
                "throwaway profiling run."
            )

        _profile_f3_training_steps(
            base_model=base_model,
            model=model,
            loss_func=loss_func,
            opt=opt,
            scheduler=scheduler,
            train_loader=replay_loader,
            dev=dev,
            grad_scaler=grad_scaler,
            extra_args=extra_args,
            num_steps=profile_steps,
        )

        # Avoid immediately entering the normal multi-hour validation pass.
        # SystemExit(0) makes the profiling job terminate successfully.
        raise SystemExit(0)

    return train_classification(
        model,
        loss_func,
        opt,
        scheduler,
        replay_loader,
        dev,
        epoch,
        steps_per_epoch=steps_per_epoch,
        grad_scaler=grad_scaler,
        tb_helper=tb_helper,
        extra_args=extra_args,
    )