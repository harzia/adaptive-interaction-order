#!/usr/bin/env python3

from __future__ import annotations

import argparse

import torch

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from weaver.utils.data.config import DataConfig
from weaver.utils.nn.optimizer.ranger import Ranger
from integrations.weaver.ParT import get_model as get_part_model
from integrations.weaver.F3ParT import get_model as get_f3_model

from aio.models.jets.particle_transformer_aio import parity_check


def make_batch(
    *,
    batch_size: int,
    seq_len: int,
    feature_dim: int,
    device: torch.device,
):
    """
    Synthetic but physically sensible ParT input.

    Returns the underlying ParticleTransformer forward tuple:
        (features, lorentz_vectors, mask)

    features:         [B, C, N]
    lorentz_vectors:  [B, 4, N]
    mask:             [B, 1, N]
    """

    torch.manual_seed(12345)

    B = batch_size
    N = seq_len

    # --------------------------------------------------------------
    # Variable jet multiplicities.
    # Keep comfortably > 2 so F3 has environment particles k.
    # --------------------------------------------------------------

    min_len = max(8, N // 2)

    lengths = torch.linspace(
        min_len,
        N,
        steps=B,
        device=device,
    ).long()

    positions = torch.arange(
        N,
        device=device,
    )[None, :]

    mask = (
        positions < lengths[:, None]
    )[:, None, :]

    # --------------------------------------------------------------
    # Preprocessed constituent features.
    # These just need realistic O(1) scale for parity testing.
    # --------------------------------------------------------------

    features = torch.randn(
        B,
        feature_dim,
        N,
        device=device,
        dtype=torch.float32,
    )

    # --------------------------------------------------------------
    # Physically sensible four-vectors:
    # E^2 = p^2 + m^2
    # --------------------------------------------------------------

    momentum = (
        torch.randn(
            B,
            3,
            N,
            device=device,
            dtype=torch.float32,
        )
        * 20.0
    )

    mass = (
        0.5
        + torch.rand(
            B,
            1,
            N,
            device=device,
            dtype=torch.float32,
        )
        * 4.0
    )

    energy = torch.sqrt(
        momentum.square().sum(dim=1, keepdim=True)
        + mass.square()
    )

    vectors = torch.cat(
        [momentum, energy],
        dim=1,
    )

    # Zero padded entries.
    features = features.masked_fill(
        ~mask.expand_as(features),
        0.0,
    )

    vectors = vectors.masked_fill(
        ~mask.expand_as(vectors),
        0.0,
    )

    return features, vectors, mask


def changed_state_keys(before, after):
    changed = []

    for key in before:
        if key not in after:
            continue

        if not torch.equal(
            before[key],
            after[key],
        ):
            changed.append(key)

    return changed


def assert_only_calibration_state_changed(changed):
    """
    Calibration may only alter the F3 factor/query scales and flag.
    It must not touch the ParT backbone, injection projections, U3,
    BatchNorm buffers, etc.
    """

    allowed_prefixes = (
        "hob.factors.g.",
        "hob.factors.n.",
        "hob.motif.Wq.",
    )

    allowed_exact = {
        "hob.calibrated",
    }

    bad = [
        key
        for key in changed
        if key not in allowed_exact
        and not key.startswith(allowed_prefixes)
    ]

    if bad:
        raise AssertionError(
            "Calibration unexpectedly changed non-calibration state:\n"
            + "\n".join(
                f"  {key}"
                for key in bad
            )
        )


def count_stale_lookahead_cache(opt):
    stale = 0
    total = 0

    for group in opt.optimizer.param_groups:
        for parameter in group["params"]:
            total += 1

            cached = (
                opt.state[parameter][
                    "cached_params"
                ]
            )

            if not torch.equal(
                cached,
                parameter.data,
            ):
                stale += 1

    return stale, total


def print_parity(title, report):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)

    for key, value in report.items():
        print(f"{key}: {value}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-config",
        default=(
            "configs/datasets/jetclass2/"
            "JetClassII_kin.yaml"
        ),
    )

    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--seq-len",
        type=int,
        default=64,
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    if (
        device.type == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but CUDA is unavailable."
        )

    print("=" * 72)
    print("F3 / ParT EQUIVALENCE TEST")
    print("=" * 72)
    print("device:", device)
    print("data config:", args.data_config)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(device),
        )

    # --------------------------------------------------------------
    # Load the actual Weaver data configuration so feature dimension
    # is never hard-coded.
    # --------------------------------------------------------------

    data_config = DataConfig.load(
        args.data_config,
        load_observers=False,
        load_reweight_info=False,
    )

    feature_dim = len(
        data_config.input_dicts[
            "pf_features"
        ]
    )

    print("feature_dim:", feature_dim)

    common_options = dict(
        num_classes=188,
        fc_params=[(512, 0.1)],

        # Disable SequenceTrimmer for this synthetic structural test.
        # Trimmer behavior should be tested separately on real Weaver
        # batches.
        trim=False,
    )

    # --------------------------------------------------------------
    # Build baseline FIRST.
    # --------------------------------------------------------------

    torch.manual_seed(123)

    base_wrapper, _ = get_part_model(
        data_config,
        **common_options,
    )

    # F3-specific initialization may consume arbitrary RNG afterward.
    aio_wrapper, _ = get_f3_model(
        data_config,
        **common_options,
    )

    base = base_wrapper.mod.to(device)
    aio = aio_wrapper.mod.to(device)

    # --------------------------------------------------------------
    # Force exactly identical shared ParT initialization.
    # --------------------------------------------------------------

    missing = aio.load_baseline(
        base.state_dict()
    )

    print()
    print("New AIO-only parameters:")

    for key in missing:
        print("  ", key)

    # --------------------------------------------------------------
    # Verify every baseline state tensor is exactly identical.
    # --------------------------------------------------------------

    base_state = base.state_dict()
    aio_state = aio.state_dict()

    shared_mismatches = []

    for key, value in base_state.items():
        if key not in aio_state:
            shared_mismatches.append(
                f"{key}: absent from AIO"
            )
            continue

        if not torch.equal(
            value,
            aio_state[key],
        ):
            shared_mismatches.append(key)

    if shared_mismatches:
        raise AssertionError(
            "Initial shared weights differ:\n"
            + "\n".join(shared_mismatches)
        )

    print()
    print("Shared ParT state: BITWISE IDENTICAL")

    # --------------------------------------------------------------
    # U3 must start exactly zero.
    # --------------------------------------------------------------

    u3_max = float(
        aio.hob.motif.U.weight
        .detach()
        .abs()
        .max()
        .cpu()
    )

    print("U3 max abs:", u3_max)

    if u3_max != 0.0:
        raise AssertionError(
            "U3 is not exactly zero at initialization."
        )

    # --------------------------------------------------------------
    # Synthetic batch.
    # --------------------------------------------------------------

    batch = make_batch(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        feature_dim=feature_dim,
        device=device,
    )

    # --------------------------------------------------------------
    # PRE-CALIBRATION parity.
    #
    # Explicit-vs-SDPA deployment comparison legitimately differs at
    # floating-point roundoff, so use the tolerance that previously
    # passed your Phase-0 test.
    # --------------------------------------------------------------

    pre_report = parity_check(
        aio,
        base,
        batch,
        rtol=1e-5,
        atol=1e-5,
    )

    print_parity(
        "PRE-CALIBRATION PARITY",
        pre_report,
    )

    if not pre_report["identity_pass"]:
        raise AssertionError(
            "Pre-calibration identity parity failed."
        )

    if not pre_report["adapter_pass"]:
        raise AssertionError(
            "Pre-calibration adapter parity failed."
        )

    if not pre_report["deployment_pass"]:
        raise AssertionError(
            "Pre-calibration deployment parity failed."
        )

    # --------------------------------------------------------------
    # Construct Ranger BEFORE calibration.
    #
    # This deliberately reproduces Weaver's ordering:
    #
    #     optimizer constructed
    #     -> Lookahead caches uncalibrated weights
    #     -> calibration runs
    # --------------------------------------------------------------

    opt = Ranger(
        aio.parameters(),
        lr=5e-4,
    )

    stale_before, total = (
        count_stale_lookahead_cache(opt)
    )

    if stale_before != 0:
        raise AssertionError(
            "Lookahead cache is already stale before calibration."
        )

    print()
    print(
        f"Lookahead cache before calibration: "
        f"{stale_before}/{total} stale"
    )

    # --------------------------------------------------------------
    # Snapshot complete AIO state before calibration.
    # --------------------------------------------------------------

    state_before_calibration = {
        key: value.detach().clone()
        for key, value
        in aio.state_dict().items()
    }

    # Also snapshot logits.
    aio.eval()

    with torch.no_grad():
        logits_before_calibration = (
            aio(*batch)
            .detach()
            .clone()
        )

    # --------------------------------------------------------------
    # CALIBRATE.
    # --------------------------------------------------------------

    features, vectors, mask = batch

    report = aio.calibrate(
        features,
        v=vectors,
        mask=mask,
        target=1.0,
    )

    print()
    print("=" * 72)
    print("CALIBRATION REPORT")
    print("=" * 72)

    for key, value in report.items():
        if torch.is_tensor(value):
            value = value.detach().cpu()

        print(f"{key}: {value}")

    if not bool(
        aio.hob.calibrated.item()
    ):
        raise AssertionError(
            "hob.calibrated was not set."
        )

    # --------------------------------------------------------------
    # Verify calibration modified only allowed F3 initialization state.
    # --------------------------------------------------------------

    state_after_calibration = (
        aio.state_dict()
    )

    changed = changed_state_keys(
        state_before_calibration,
        state_after_calibration,
    )

    print()
    print("State changed by calibration:")

    for key in changed:
        print("  ", key)

    assert_only_calibration_state_changed(
        changed
    )

    # U3 MUST still be zero.
    u3_after = float(
        aio.hob.motif.U.weight
        .detach()
        .abs()
        .max()
        .cpu()
    )

    if u3_after != 0.0:
        raise AssertionError(
            "Calibration changed zero-initialized U3."
        )

    # --------------------------------------------------------------
    # Crucial invariant:
    #
    # calibration may drastically change g/n/Wq internally, but because
    # U3 == 0 it must NOT change the actual network function.
    # --------------------------------------------------------------

    aio.eval()

    with torch.no_grad():
        logits_after_calibration = (
            aio(*batch)
            .detach()
            .clone()
        )

    calibration_logit_diff = float(
        (
            logits_after_calibration
            - logits_before_calibration
        )
        .abs()
        .max()
        .cpu()
    )

    print()
    print(
        "AIO logits max diff caused by calibration:",
        calibration_logit_diff,
    )

    if calibration_logit_diff != 0.0:
        raise AssertionError(
            "Calibration changed model logits even though U3 == 0."
        )

    # --------------------------------------------------------------
    # Ranger bug reproduction.
    #
    # Its Lookahead cache was constructed before calibration, so some
    # cached parameters SHOULD now be stale.
    # --------------------------------------------------------------

    stale_after_calibration, total = (
        count_stale_lookahead_cache(opt)
    )

    print(
        "Lookahead cache immediately after calibration:",
        f"{stale_after_calibration}/{total} stale",
    )

    if stale_after_calibration == 0:
        raise AssertionError(
            "Expected calibration to make Ranger's "
            "Lookahead cache stale, but no stale "
            "parameters were found."
        )

    # --------------------------------------------------------------
    # Apply the fix.
    # --------------------------------------------------------------

    opt.reset()

    stale_after_reset, total = (
        count_stale_lookahead_cache(opt)
    )

    print(
        "Lookahead cache after opt.reset():",
        f"{stale_after_reset}/{total} stale",
    )

    if stale_after_reset != 0:
        raise AssertionError(
            "Lookahead cache still contains stale "
            "weights after opt.reset()."
        )

    # --------------------------------------------------------------
    # POST-CALIBRATION parity with untouched baseline.
    #
    # Shared ParT parameters have not changed; calibration changed only
    # dormant F3 internals. Therefore all parity contracts must remain.
    # --------------------------------------------------------------

    post_report = parity_check(
        aio,
        base,
        batch,
        rtol=1e-5,
        atol=1e-5,
    )

    print_parity(
        "POST-CALIBRATION PARITY",
        post_report,
    )

    if not post_report["identity_pass"]:
        raise AssertionError(
            "Post-calibration identity parity failed."
        )

    if not post_report["adapter_pass"]:
        raise AssertionError(
            "Post-calibration adapter parity failed."
        )

    if not post_report["deployment_pass"]:
        raise AssertionError(
            "Post-calibration deployment parity failed."
        )

    print()
    print("=" * 72)
    print("ALL TESTS PASSED")
    print("=" * 72)
    print(
        "✓ shared ParT initialization identical"
    )
    print(
        "✓ U3 exactly zero"
    )
    print(
        "✓ pre-calibration identity parity"
    )
    print(
        "✓ pre-calibration adapter parity"
    )
    print(
        "✓ pre-calibration deployment parity"
    )
    print(
        "✓ calibration only changed F3 scale state"
    )
    print(
        "✓ calibration did not change network logits"
    )
    print(
        "✓ Ranger cache became stale as expected"
    )
    print(
        "✓ opt.reset() refreshed Lookahead cache"
    )
    print(
        "✓ post-calibration identity parity"
    )
    print(
        "✓ post-calibration adapter parity"
    )
    print(
        "✓ post-calibration deployment parity"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 72)
        print("TEST FAILED")
        print("=" * 72)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        raise