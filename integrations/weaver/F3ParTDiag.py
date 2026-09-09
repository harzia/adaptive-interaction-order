from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from weaver.utils.logger import _logger

from integrations.weaver.F3ParT import (
    get_model,
    get_loss,
    get_evaluate_fn,
)


def _unwrap_model(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def _to_cpu_tensor(value):
    if torch.is_tensor(value):
        return (
            value.detach()
            .float()
            .cpu()
        )

    return torch.as_tensor(
        value,
        dtype=torch.float32,
    )


def _json_value(value):
    if torch.is_tensor(value):
        value = value.detach().cpu()

        if value.ndim == 0:
            return value.item()

        return value.tolist()

    return value


def _aggregate_reports(reports):
    """
    Elementwise aggregation across Weaver batches.

    Vector-valued quantities such as per-head logit std
    remain vectors in the JSON.
    """
    result = {}

    keys = reports[0].keys()

    for key in keys:
        values = [
            _to_cpu_tensor(report[key])
            for report in reports
        ]

        try:
            stacked = torch.stack(
                values,
                dim=0,
            )
        except RuntimeError:
            # Defensive fallback for any future
            # variable-length diagnostic.
            continue

        result[key] = {
            "mean": _json_value(
                stacked.mean(dim=0)
            ),
            "std": _json_value(
                stacked.std(
                    dim=0,
                    unbiased=False,
                )
            ),
            "min": _json_value(
                stacked.min(dim=0).values
            ),
            "max": _json_value(
                stacked.max(dim=0).values
            ),
        }

    return result


def diagnose_f3_calibration(
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
):
    """
    Weaver training hook used only for Phase-0 diagnostics.

    It consumes real batches from Weaver's train_loader,
    performs no optimizer step, writes JSON, and exits 0.
    """
    del (
        loss_func,
        opt,
        scheduler,
        epoch,
        steps_per_epoch,
        grad_scaler,
        tb_helper,
    )

    if (
        dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() != 1
    ):
        raise RuntimeError(
            "F3 calibration diagnostics should be "
            "run with one process/GPU."
        )

    num_batches = int(
        os.environ.get(
            "F3_DIAG_BATCHES",
            "5",
        )
    )

    if num_batches < 1:
        raise ValueError(
            "F3_DIAG_BATCHES must be >= 1"
        )

    output_path = Path(
        os.environ.get(
            "F3_DIAG_OUTPUT",
            "f3_calibration_diagnostics.json",
        )
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    wrapped = _unwrap_model(model)

    if not hasattr(
        wrapped,
        "calibration_diagnostics",
    ):
        raise RuntimeError(
            "Model does not expose "
            "calibration_diagnostics()."
        )

    data_config = (
        train_loader.dataset.config
    )

    reports = []

    _logger.info(
        "Starting F3 calibration diagnostic "
        "on %d real Weaver training batches.",
        num_batches,
    )

    for batch_idx, (X, _, _) in enumerate(
        train_loader
    ):
        if batch_idx >= num_batches:
            break

        inputs = [
            X[name].to(dev)
            for name
            in data_config.input_names
        ]

        report = (
            wrapped
            .calibration_diagnostics(
                *inputs
            )
        )

        report_cpu = {
            key: (
                value.detach()
                .float()
                .cpu()
                if torch.is_tensor(value)
                else value
            )
            for key, value
            in report.items()
        }

        reports.append(
            report_cpu
        )

        _logger.info(
            "F3 diagnostic batch %d/%d: %s",
            batch_idx + 1,
            num_batches,
            {
                key: _json_value(value)
                for key, value
                in report_cpu.items()
                if (
                    "per_channel"
                    not in key
                )
            },
        )

    if not reports:
        raise RuntimeError(
            "Diagnostic processed zero batches."
        )

    args = (
        extra_args.get("args")
        if extra_args is not None
        else None
    )

    seed = getattr(
        args,
        "seed",
        None,
    )

    payload = {
        "mode": "f3_calibration_diagnostic",
        "seed": seed,
        "num_batches": len(reports),
        "input_names": list(
            data_config.input_names
        ),

        "calibrated_before": bool(
            wrapped.mod.hob.calibrated.item()
        ),

        "per_batch": [
            {
                key: _json_value(value)
                for key, value
                in report.items()
            }
            for report in reports
        ],

        "aggregate": (
            _aggregate_reports(
                reports
            )
        ),
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
        )

    agg = payload["aggregate"]

    _logger.info(
        "============================================================"
    )
    _logger.info(
        "F3 CALIBRATION DIAGNOSTIC COMPLETE"
    )
    _logger.info(
        "Seed: %s",
        seed,
    )
    _logger.info(
        "Batches: %d",
        len(reports),
    )

    for key in (
        "g_std_mean",
        "n_std_mean",
        "term_std",
        "logit_std_mean_pool0",
        "logit_std_median_pool0",
        "logit_std_p10_pool0",
        "logit_std_p90_pool0",
        "entropy_mean_pool0",
    ):
        if key in agg:
            _logger.info(
                "%s: mean=%s std=%s",
                key,
                agg[key]["mean"],
                agg[key]["std"],
            )

    _logger.info(
        "Output: %s",
        output_path,
    )
    _logger.info(
        "No optimizer step was performed."
    )
    _logger.info(
        "============================================================"
    )

    # Exit successfully before Weaver proceeds with training.
    raise SystemExit(0)


def get_train_fn(
    data_config,
    **kwargs,
):
    del data_config, kwargs
    return diagnose_f3_calibration