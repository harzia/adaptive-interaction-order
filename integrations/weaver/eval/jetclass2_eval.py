from __future__ import annotations

import json
import os
from pathlib import Path

from typing import Any

import numpy as np
import uproot
from sklearn.metrics import accuracy_score, roc_auc_score


# JetClass-II has 188 output classes:
#   0-160   signal classes
#   161-187 QCD subclasses
NUM_CLASSES = 188
RES2P_END = 15
RES34P_START = RES2P_END
RES34P_END = 161
QCD_START = RES34P_END
QCD_INDICES = np.arange(QCD_START, NUM_CLASSES, dtype=np.int64)

# The full class ordering follows JetClassII_full.yaml.
CLASS_NAMES = ['label_X_bb',
 'label_X_cc',
 'label_X_ss',
 'label_X_qq',
 'label_X_bc',
 'label_X_cs',
 'label_X_bq',
 'label_X_cq',
 'label_X_sq',
 'label_X_gg',
 'label_X_ee',
 'label_X_mm',
 'label_X_tauhtaue',
 'label_X_tauhtaum',
 'label_X_tauhtauh',
 'label_X_YY_bbbb',
 'label_X_YY_bbcc',
 'label_X_YY_bbss',
 'label_X_YY_bbqq',
 'label_X_YY_bbgg',
 'label_X_YY_bbee',
 'label_X_YY_bbmm',
 'label_X_YY_bbtauhtaue',
 'label_X_YY_bbtauhtaum',
 'label_X_YY_bbtauhtauh',
 'label_X_YY_bbb',
 'label_X_YY_bbc',
 'label_X_YY_bbs',
 'label_X_YY_bbq',
 'label_X_YY_bbg',
 'label_X_YY_bbe',
 'label_X_YY_bbm',
 'label_X_YY_cccc',
 'label_X_YY_ccss',
 'label_X_YY_ccqq',
 'label_X_YY_ccgg',
 'label_X_YY_ccee',
 'label_X_YY_ccmm',
 'label_X_YY_cctauhtaue',
 'label_X_YY_cctauhtaum',
 'label_X_YY_cctauhtauh',
 'label_X_YY_ccb',
 'label_X_YY_ccc',
 'label_X_YY_ccs',
 'label_X_YY_ccq',
 'label_X_YY_ccg',
 'label_X_YY_cce',
 'label_X_YY_ccm',
 'label_X_YY_ssss',
 'label_X_YY_ssqq',
 'label_X_YY_ssgg',
 'label_X_YY_ssee',
 'label_X_YY_ssmm',
 'label_X_YY_sstauhtaue',
 'label_X_YY_sstauhtaum',
 'label_X_YY_sstauhtauh',
 'label_X_YY_ssb',
 'label_X_YY_ssc',
 'label_X_YY_sss',
 'label_X_YY_ssq',
 'label_X_YY_ssg',
 'label_X_YY_sse',
 'label_X_YY_ssm',
 'label_X_YY_qqqq',
 'label_X_YY_qqgg',
 'label_X_YY_qqee',
 'label_X_YY_qqmm',
 'label_X_YY_qqtauhtaue',
 'label_X_YY_qqtauhtaum',
 'label_X_YY_qqtauhtauh',
 'label_X_YY_qqb',
 'label_X_YY_qqc',
 'label_X_YY_qqs',
 'label_X_YY_qqq',
 'label_X_YY_qqg',
 'label_X_YY_qqe',
 'label_X_YY_qqm',
 'label_X_YY_gggg',
 'label_X_YY_ggee',
 'label_X_YY_ggmm',
 'label_X_YY_ggtauhtaue',
 'label_X_YY_ggtauhtaum',
 'label_X_YY_ggtauhtauh',
 'label_X_YY_ggb',
 'label_X_YY_ggc',
 'label_X_YY_ggs',
 'label_X_YY_ggq',
 'label_X_YY_ggg',
 'label_X_YY_gge',
 'label_X_YY_ggm',
 'label_X_YY_bee',
 'label_X_YY_cee',
 'label_X_YY_see',
 'label_X_YY_qee',
 'label_X_YY_gee',
 'label_X_YY_bmm',
 'label_X_YY_cmm',
 'label_X_YY_smm',
 'label_X_YY_qmm',
 'label_X_YY_gmm',
 'label_X_YY_btauhtaue',
 'label_X_YY_ctauhtaue',
 'label_X_YY_stauhtaue',
 'label_X_YY_qtauhtaue',
 'label_X_YY_gtauhtaue',
 'label_X_YY_btauhtaum',
 'label_X_YY_ctauhtaum',
 'label_X_YY_stauhtaum',
 'label_X_YY_qtauhtaum',
 'label_X_YY_gtauhtaum',
 'label_X_YY_btauhtauh',
 'label_X_YY_ctauhtauh',
 'label_X_YY_stauhtauh',
 'label_X_YY_qtauhtauh',
 'label_X_YY_gtauhtauh',
 'label_X_YY_qqqb',
 'label_X_YY_qqqc',
 'label_X_YY_qqqs',
 'label_X_YY_bbcq',
 'label_X_YY_ccbs',
 'label_X_YY_ccbq',
 'label_X_YY_ccsq',
 'label_X_YY_sscq',
 'label_X_YY_qqbc',
 'label_X_YY_qqbs',
 'label_X_YY_qqcs',
 'label_X_YY_bcsq',
 'label_X_YY_bcs',
 'label_X_YY_bcq',
 'label_X_YY_bsq',
 'label_X_YY_csq',
 'label_X_YY_bcev',
 'label_X_YY_csev',
 'label_X_YY_bqev',
 'label_X_YY_cqev',
 'label_X_YY_sqev',
 'label_X_YY_qqev',
 'label_X_YY_bcmv',
 'label_X_YY_csmv',
 'label_X_YY_bqmv',
 'label_X_YY_cqmv',
 'label_X_YY_sqmv',
 'label_X_YY_qqmv',
 'label_X_YY_bctauev',
 'label_X_YY_cstauev',
 'label_X_YY_bqtauev',
 'label_X_YY_cqtauev',
 'label_X_YY_sqtauev',
 'label_X_YY_qqtauev',
 'label_X_YY_bctaumv',
 'label_X_YY_cstaumv',
 'label_X_YY_bqtaumv',
 'label_X_YY_cqtaumv',
 'label_X_YY_sqtaumv',
 'label_X_YY_qqtaumv',
 'label_X_YY_bctauhv',
 'label_X_YY_cstauhv',
 'label_X_YY_bqtauhv',
 'label_X_YY_cqtauhv',
 'label_X_YY_sqtauhv',
 'label_X_YY_qqtauhv',
 'label_QCD_bbccss',
 'label_QCD_bbccs',
 'label_QCD_bbcc',
 'label_QCD_bbcss',
 'label_QCD_bbcs',
 'label_QCD_bbc',
 'label_QCD_bbss',
 'label_QCD_bbs',
 'label_QCD_bb',
 'label_QCD_bccss',
 'label_QCD_bccs',
 'label_QCD_bcc',
 'label_QCD_bcss',
 'label_QCD_bcs',
 'label_QCD_bc',
 'label_QCD_bss',
 'label_QCD_bs',
 'label_QCD_b',
 'label_QCD_ccss',
 'label_QCD_ccs',
 'label_QCD_cc',
 'label_QCD_css',
 'label_QCD_cs',
 'label_QCD_c',
 'label_QCD_ss',
 'label_QCD_s',
 'label_QCD_light']

# For rejection/AUC reporting, use the same 29 broad signal topology groups
# already defined in the JetClass-II data config for reweighting.
# These groups are disjoint and exactly cover labels 0-160.
SIGNAL_GROUPS = {'label_X_QQ': [0, 1, 2, 3, 4, 5, 6, 7, 8],
 'label_X_gg': [9],
 'label_X_ll': [10, 11],
 'label_X_tauhtaul': [12, 13],
 'label_X_tauhtauh': [14],
 'label_X_YY_QQQQ': [15,
                     16,
                     17,
                     18,
                     32,
                     33,
                     34,
                     48,
                     49,
                     63,
                     115,
                     116,
                     117,
                     118,
                     119,
                     120,
                     121,
                     122,
                     123,
                     124,
                     125,
                     126],
 'label_X_YY_QQgg': [19, 35, 50, 64],
 'label_X_YY_gggg': [77],
 'label_X_YY_QQQ': [25,
                    26,
                    27,
                    28,
                    41,
                    42,
                    43,
                    44,
                    56,
                    57,
                    58,
                    59,
                    70,
                    71,
                    72,
                    73,
                    127,
                    128,
                    129,
                    130],
 'label_X_YY_QQg': [29, 45, 60, 74],
 'label_X_YY_Qgg': [83, 84, 85, 86],
 'label_X_YY_ggg': [87],
 'label_X_YY_QQll': [20, 21, 36, 37, 51, 52, 65, 66],
 'label_X_YY_QQl': [30, 31, 46, 47, 61, 62, 75, 76],
 'label_X_YY_Qll': [90, 91, 92, 93, 95, 96, 97, 98],
 'label_X_YY_QQtauhtaul': [22, 23, 38, 39, 53, 54, 67, 68],
 'label_X_YY_QQtauhtauh': [24, 40, 55, 69],
 'label_X_YY_Qtauhtaul': [100, 101, 102, 103, 105, 106, 107, 108],
 'label_X_YY_Qtauhtauh': [110, 111, 112, 113],
 'label_X_YY_ggll': [78, 79],
 'label_X_YY_ggl': [88, 89],
 'label_X_YY_gll': [94, 99],
 'label_X_YY_ggtauhtaul': [80, 81],
 'label_X_YY_ggtauhtauh': [82],
 'label_X_YY_gtauhtaul': [104, 109],
 'label_X_YY_gtauhtauh': [114],
 'label_X_YY_QQlv': [131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142],
 'label_X_YY_QQtaulv': [143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154],
 'label_X_YY_QQtauhv': [155, 156, 157, 158, 159, 160]}

WORKING_POINTS = (0.30, 0.50)

# The full JetClass-II test set is very large, so evaluation is streamed.
#
# Group-vs-QCD ROC quantities are accumulated with fine score histograms.
# The 188-way macro OVO AUC is calculated on a deterministic stratified
# sample to avoid storing the full N x 188 score matrix in memory.
STEP_SIZE = os.environ.get("JETCLASS2_EVAL_STEP_SIZE", "100 MB")
ROC_BINS = int(os.environ.get("JETCLASS2_EVAL_ROC_BINS", "50000"))
AUC_SAMPLES_PER_CLASS = int(
    os.environ.get("JETCLASS2_EVAL_AUC_SAMPLES_PER_CLASS", "1000")
)
AUC_RANDOM_SEED = int(os.environ.get("JETCLASS2_EVAL_SEED", "12345"))


GROUP_NAMES = list(SIGNAL_GROUPS)
GROUP_INDICES = [
    np.asarray(SIGNAL_GROUPS[name], dtype=np.int64)
    for name in GROUP_NAMES
]
NUM_SIGNAL_GROUPS = len(GROUP_NAMES)

CLASS_TO_GROUP = np.full(QCD_START, -1, dtype=np.int64)
for group_index, indices in enumerate(GROUP_INDICES):
    CLASS_TO_GROUP[indices] = group_index

if np.any(CLASS_TO_GROUP < 0):
    raise RuntimeError("JetClass-II signal groups do not cover labels 0-160.")


class _StratifiedScoreSampler:
    """Keep a deterministic uniform sample of score vectors for each class."""

    def __init__(
        self,
        *,
        num_classes: int,
        samples_per_class: int,
        seed: int,
    ) -> None:
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.rng = np.random.default_rng(seed)

        self.keys = [
            np.empty(0, dtype=np.float64)
            for _ in range(num_classes)
        ]
        self.scores = [
            np.empty((0, num_classes), dtype=np.float32)
            for _ in range(num_classes)
        ]

    def update(
        self,
        labels: np.ndarray,
        probabilities: np.ndarray,
    ) -> None:
        if self.samples_per_class <= 0:
            return

        for class_index in np.unique(labels):
            class_index = int(class_index)
            rows = np.flatnonzero(labels == class_index)

            if rows.size == 0:
                continue

            new_keys = self.rng.random(rows.size)
            new_scores = probabilities[rows].astype(
                np.float32,
                copy=False,
            )

            all_keys = np.concatenate(
                [self.keys[class_index], new_keys]
            )
            all_scores = np.concatenate(
                [self.scores[class_index], new_scores],
                axis=0,
            )

            if all_keys.size > self.samples_per_class:
                keep = np.argpartition(
                    all_keys,
                    self.samples_per_class - 1,
                )[: self.samples_per_class]
                all_keys = all_keys[keep]
                all_scores = all_scores[keep]

            self.keys[class_index] = all_keys
            self.scores[class_index] = all_scores

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        label_chunks: list[np.ndarray] = []
        score_chunks: list[np.ndarray] = []

        for class_index, class_scores in enumerate(self.scores):
            if class_scores.shape[0] == 0:
                continue

            label_chunks.append(
                np.full(
                    class_scores.shape[0],
                    class_index,
                    dtype=np.int64,
                )
            )
            score_chunks.append(class_scores)

        if not score_chunks:
            return (
                np.empty(0, dtype=np.int64),
                np.empty((0, self.num_classes), dtype=np.float32),
            )

        return (
            np.concatenate(label_chunks),
            np.concatenate(score_chunks, axis=0),
        )


def evaluate_prediction_files(
    *,
    prediction_files: dict[str, str],
    args: Any,
) -> dict[str, Any]:
    """
    Evaluate JetClass-II Weaver prediction ROOT files.

    Expected Weaver prediction branches
    -----------------------------------
    ``truth_label``:
        Integer JetClass-II target in [0, 187].

    ``output``:
        Length-188 score/probability vector.

    Metrics
    -------
    * ``accuracy_top1``: exact 188-way top-1 accuracy over all events;
    * ``accuracy_macro``: unweighted mean of the 188 per-class accuracies
      (equivalent to multiclass balanced accuracy / macro recall);
    * ``accuracy_2prong``: exact 188-way top-1 accuracy restricted to
      true labels 0-14;
    * ``accuracy_34prong``: exact 188-way top-1 accuracy restricted to
      true labels 15-160;
    * ``accuracy_qcd``: exact 188-way top-1 accuracy restricted to
      true labels 161-187;
    * stratified-sample 188-way macro one-vs-one ROC AUC;
    * signal-topology-vs-QCD AUC for the 29 JetClass-II topology groups;
    * QCD background rejection at 30% and 50% signal efficiency for those groups.

    For a signal topology S, the binary discriminant is

        p(S) / (p(S) + p(QCD)),

    where p(S) is the sum of model probabilities for all 188-way classes
    belonging to S, and p(QCD) is the sum over labels 161-187.
    """

    if not prediction_files:
        raise ValueError("No prediction files were supplied.")

    if ROC_BINS < 100:
        raise ValueError("JETCLASS2_EVAL_ROC_BINS must be at least 100.")

    signal_hist = np.zeros(
        (NUM_SIGNAL_GROUPS, ROC_BINS),
        dtype=np.int64,
    )
    qcd_hist = np.zeros(
        (NUM_SIGNAL_GROUPS, ROC_BINS),
        dtype=np.int64,
    )

    class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    class_correct_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    correct_events = 0
    num_events = 0

    auc_sampler = _StratifiedScoreSampler(
        num_classes=NUM_CLASSES,
        samples_per_class=AUC_SAMPLES_PER_CLASS,
        seed=AUC_RANDOM_SEED,
    )

    for true_labels, probabilities in _iter_prediction_chunks(
        prediction_files
    ):
        if true_labels.size == 0:
            continue

        num_events += int(true_labels.size)

        class_counts += np.bincount(
            true_labels,
            minlength=NUM_CLASSES,
        )[:NUM_CLASSES]

        predicted_labels = np.argmax(
            probabilities,
            axis=1,
        )
        correct_mask = predicted_labels == true_labels
        correct_events += int(np.count_nonzero(correct_mask))

        if np.any(correct_mask):
            class_correct_counts += np.bincount(
                true_labels[correct_mask],
                minlength=NUM_CLASSES,
            )[:NUM_CLASSES]

        auc_sampler.update(
            true_labels,
            probabilities,
        )

        qcd_probabilities = probabilities[
            :,
            QCD_START:NUM_CLASSES,
        ].sum(axis=1)

        group_probabilities = np.empty(
            (probabilities.shape[0], NUM_SIGNAL_GROUPS),
            dtype=np.float32,
        )

        for group_index, indices in enumerate(GROUP_INDICES):
            group_probabilities[:, group_index] = probabilities[
                :,
                indices,
            ].sum(axis=1)

        denominator = (
            group_probabilities
            + qcd_probabilities[:, None]
        )

        discriminants = np.divide(
            group_probabilities,
            denominator,
            out=np.zeros_like(group_probabilities),
            where=denominator > 0,
        )

        signal_mask = true_labels < QCD_START
        if np.any(signal_mask):
            signal_labels = true_labels[signal_mask]
            signal_group_indices = CLASS_TO_GROUP[signal_labels]

            signal_scores = discriminants[
                signal_mask,
                signal_group_indices,
            ]
            signal_bins = _score_bins(signal_scores)

            flat_indices = (
                signal_group_indices * ROC_BINS
                + signal_bins
            )
            signal_hist += np.bincount(
                flat_indices,
                minlength=NUM_SIGNAL_GROUPS * ROC_BINS,
            ).reshape(NUM_SIGNAL_GROUPS, ROC_BINS)

        qcd_mask = true_labels >= QCD_START
        if np.any(qcd_mask):
            background_scores = discriminants[qcd_mask]
            background_bins = _score_bins(background_scores)

            offsets = (
                np.arange(NUM_SIGNAL_GROUPS, dtype=np.int64)
                * ROC_BINS
            )

            flat_indices = (
                background_bins
                + offsets[None, :]
            ).ravel()

            qcd_hist += np.bincount(
                flat_indices,
                minlength=NUM_SIGNAL_GROUPS * ROC_BINS,
            ).reshape(NUM_SIGNAL_GROUPS, ROC_BINS)

    if num_events == 0:
        raise ValueError("Prediction files contained no events.")

    accuracy_top1 = correct_events / num_events

    per_class_accuracy = np.divide(
        class_correct_counts,
        class_counts,
        out=np.full(NUM_CLASSES, np.nan, dtype=np.float64),
        where=class_counts > 0,
    )
    present_classes = class_counts > 0
    accuracy_macro = (
        float(np.mean(per_class_accuracy[present_classes]))
        if np.any(present_classes)
        else None
    )

    def _range_accuracy(start: int, end: int) -> float | None:
        count = int(class_counts[start:end].sum())
        if count == 0:
            return None
        correct = int(class_correct_counts[start:end].sum())
        return float(correct / count)

    accuracy_2prong = _range_accuracy(0, RES2P_END)
    accuracy_34prong = _range_accuracy(RES34P_START, RES34P_END)
    accuracy_qcd = _range_accuracy(QCD_START, NUM_CLASSES)

    metrics: dict[str, float | int | None] = {
        "accuracy_top1": float(accuracy_top1),
        "accuracy_macro": accuracy_macro,
        "accuracy_2prong": accuracy_2prong,
        "accuracy_34prong": accuracy_34prong,
        "accuracy_qcd": accuracy_qcd,
        "num_events": int(num_events),
        "num_2prong_events": int(class_counts[:RES2P_END].sum()),
        "num_34prong_events": int(
            class_counts[RES34P_START:RES34P_END].sum()
        ),
        "num_qcd_events": int(class_counts[QCD_START:].sum()),
    }

    auc_labels, auc_probabilities = auc_sampler.finalize()

    if auc_labels.size > 0:
        sampled_classes = np.unique(auc_labels)

        if sampled_classes.size == NUM_CLASSES:
            overall_auc = roc_auc_score(
                auc_labels,
                auc_probabilities,
                average="macro",
                multi_class="ovo",
                labels=np.arange(NUM_CLASSES),
            )
            metrics["overall_auc_macro_ovo_sampled"] = float(
                overall_auc
            )
            metrics["overall_auc_num_events"] = int(
                auc_labels.size
            )
            metrics["overall_auc_samples_per_class"] = int(
                AUC_SAMPLES_PER_CLASS
            )
        else:
            metrics["overall_auc_macro_ovo_sampled"] = None
            metrics["overall_auc_num_events"] = int(
                auc_labels.size
            )

    rejection_rows: list[list[Any]] = []

    plot_rejections_by_wp: dict[float, tuple[list[str], list[float]]] = {
        working_point: ([], [])
        for working_point in WORKING_POINTS
    }

    plot_auc_groups: list[str] = []
    plot_aucs: list[float] = []

    for group_index, group_name in enumerate(GROUP_NAMES):
        display_name = _display_name(group_name)

        group_auc: float | None = None

        for working_point in WORKING_POINTS:
            (
                selected_signal_efficiency,
                selected_background_efficiency,
                background_rejection,
                threshold,
                this_group_auc,
            ) = _metrics_from_histograms(
                signal_hist[group_index],
                qcd_hist[group_index],
                working_point,
            )

            if group_auc is None:
                group_auc = this_group_auc

            metric_suffix = _working_point_suffix(
                working_point
            )

            metrics[
                f"rejection_{display_name}_at_{metric_suffix}"
            ] = background_rejection

            rejection_rows.append(
                [
                    display_name,
                    int(signal_hist[group_index].sum()),
                    int(qcd_hist[group_index].sum()),
                    float(working_point),
                    selected_signal_efficiency,
                    selected_background_efficiency,
                    background_rejection,
                    threshold,
                    group_auc,
                ]
            )

            if background_rejection is not None:
                (
                    plot_groups_for_wp,
                    plot_values_for_wp,
                ) = plot_rejections_by_wp[
                    working_point
                ]

                plot_groups_for_wp.append(
                    display_name
                )
                plot_values_for_wp.append(
                    float(background_rejection)
                )

        metrics[
            f"auc_{display_name}_vs_QCD"
        ] = group_auc

        if group_auc is not None:
            plot_auc_groups.append(display_name)
            plot_aucs.append(float(group_auc))

    class_count_rows = [
        [
            class_index,
            _display_name(CLASS_NAMES[class_index]),
            int(class_counts[class_index]),
            int(class_correct_counts[class_index]),
            (
                float(per_class_accuracy[class_index])
                if class_counts[class_index] > 0
                else None
            ),
            (
                "QCD"
                if class_index >= QCD_START
                else "3/4-prong"
                if class_index >= RES34P_START
                else "2-prong"
            ),
        ]
        for class_index in range(NUM_CLASSES)
    ]

    plots: dict[str, dict[str, Any]] = {}

    for working_point in WORKING_POINTS:
        (
            plot_groups_for_wp,
            plot_values_for_wp,
        ) = plot_rejections_by_wp[
            working_point
        ]

        if plot_groups_for_wp:
            percentage = int(round(working_point * 100))
            plots[
                f"background_rejection_{percentage}pct"
            ] = {
                "type": "bar",
                "title": (
                    "JetClass-II QCD rejection at "
                    f"{percentage}% signal efficiency"
                ),
                "x": plot_groups_for_wp,
                "y": plot_values_for_wp,
                "x_label": "Signal topology",
                "y_label": "QCD background rejection",
            }

    if plot_auc_groups:
        plots["signal_vs_qcd_auc"] = {
            "type": "bar",
            "title": "JetClass-II signal-topology vs QCD AUC",
            "x": plot_auc_groups,
            "y": plot_aucs,
            "x_label": "Signal topology",
            "y_label": "ROC AUC",
        }

    result = {
        "metrics": metrics,
        "tables": {
            "background_rejection_table": {
                "columns": [
                    "signal_group",
                    "num_signal_events",
                    "num_qcd_events",
                    "target_signal_efficiency",
                    "selected_signal_efficiency",
                    "background_efficiency",
                    "background_rejection",
                    "threshold",
                    "auc_vs_qcd",
                ],
                "data": rejection_rows,
            },
            "class_counts": {
                "columns": [
                    "class_index",
                    "class_name",
                    "num_events",
                    "num_correct",
                    "accuracy",
                    "type",
                ],
                "data": class_count_rows,
            },
        },
        "plots": plots,
    }

    prediction_paths = [
        Path(file_path).resolve()
        for file_path in prediction_files.values()
    ]
    results_dir = Path(
        os.path.commonpath(
            [str(prediction_path.parent) for prediction_path in prediction_paths]
        )
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    network_config = _first_arg_value(
        getattr(args, "network_config", None)
    )
    data_config = _first_arg_value(
        getattr(args, "data_config", None)
    )

    model_name = (
        Path(str(network_config)).stem
        if network_config is not None
        else None
    )

    feature_type = None
    if data_config is not None:
        data_config_stem = Path(str(data_config)).stem
        feature_type = data_config_stem.removeprefix("JetClassII_")

    metrics_summary = {
        "model": {
            "name": model_name,
            "features": feature_type,
            "network_config": (
                str(network_config)
                if network_config is not None
                else None
            ),
            "data_config": (
                str(data_config)
                if data_config is not None
                else None
            ),
        },
        "run": {
            "command": os.environ.get("RUN_COMMAND"),
            "comment": os.environ.get(
                "RUN_COMMENT",
                os.environ.get("COMMENT"),
            ),
            # Custom Weaver argument added in this fork.
            "seed": getattr(args, "seed", None),
        },
        "evaluation": {
            "num_classes": NUM_CLASSES,
            "class_ranges": {
                "2prong": [0, RES2P_END - 1],
                "34prong": [RES34P_START, RES34P_END - 1],
                "qcd": [QCD_START, NUM_CLASSES - 1],
            },
            "num_signal_groups": NUM_SIGNAL_GROUPS,
            "signal_working_points": list(WORKING_POINTS),
            "background_rejection_definition": (
                "1 / QCD efficiency at each selected signal working point"
            ),
            "signal_vs_qcd_score": (
                "p(signal_group) / "
                "(p(signal_group) + p(all_QCD_classes))"
            ),
            "overall_auc": {
                "metric": "macro one-vs-one ROC AUC",
                "sampled": True,
                "samples_per_class": AUC_SAMPLES_PER_CLASS,
                "random_seed": AUC_RANDOM_SEED,
            },
            "roc_histogram_bins": ROC_BINS,
            "root_step_size": STEP_SIZE,
            "prediction_files": {
                group_name: str(Path(file_path).resolve())
                for group_name, file_path
                in sorted(prediction_files.items())
            },
        },
        "metrics": metrics,
    }

    metrics_path = results_dir / "metrics.json"
    with metrics_path.open("w") as metrics_file:
        json.dump(
            metrics_summary,
            metrics_file,
            indent=2,
            sort_keys=True,
        )

    print(f"Saved evaluation metrics to: {metrics_path}")

    return result



def _first_arg_value(value: Any) -> Any:
    """Return the first value from a Weaver argparse option when list-like."""
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _iter_prediction_chunks(
    prediction_files: dict[str, str],
):
    requested_branches = ["truth_label", "output"]

    for group_name, file_path in sorted(prediction_files.items()):
        with uproot.open(file_path) as root_file:
            if "Events" not in root_file:
                raise KeyError(
                    f"Prediction file {file_path!r} for group "
                    f"{group_name!r} does not contain an 'Events' tree."
                )

            tree = root_file["Events"]
            available_branches = set(tree.keys())

            missing_branches = [
                branch
                for branch in requested_branches
                if branch not in available_branches
            ]

            if missing_branches:
                raise KeyError(
                    f"Prediction file {file_path!r} is missing branches: "
                    + ", ".join(missing_branches)
                    + ". JetClass-II custom-label predictions are expected "
                    "to contain 'truth_label' and the 188-vector 'output'."
                )

            for arrays in tree.iterate(
                requested_branches,
                library="np",
                step_size=STEP_SIZE,
            ):
                true_labels = np.asarray(
                    arrays["truth_label"]
                ).reshape(-1)

                probabilities = _as_probability_matrix(
                    arrays["output"]
                )

                if true_labels.shape[0] != probabilities.shape[0]:
                    raise ValueError(
                        f"Prediction file {file_path!r} has mismatched "
                        "'truth_label' and 'output' lengths."
                    )

                if np.any(
                    (true_labels < 0)
                    | (true_labels >= NUM_CLASSES)
                ):
                    raise ValueError(
                        f"Prediction file {file_path!r} contains "
                        "JetClass-II labels outside [0, 187]."
                    )

                yield (
                    true_labels.astype(np.int64, copy=False),
                    probabilities,
                )


def _as_probability_matrix(values: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(values)

    if probabilities.dtype == object:
        probabilities = np.stack(probabilities)

    if probabilities.ndim != 2:
        raise ValueError(
            "Expected Weaver 'output' to be a 2D score array; "
            f"got shape {probabilities.shape}."
        )

    if probabilities.shape[1] != NUM_CLASSES:
        raise ValueError(
            "Expected 188 JetClass-II output scores per event; "
            f"got {probabilities.shape[1]}."
        )

    return probabilities.astype(np.float32, copy=False)


def _score_bins(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores, 0.0, 1.0)

    bins = np.floor(
        clipped * ROC_BINS
    ).astype(np.int64)

    return np.minimum(
        bins,
        ROC_BINS - 1,
    )


def _metrics_from_histograms(
    signal_hist: np.ndarray,
    background_hist: np.ndarray,
    target_efficiency: float,
) -> tuple[
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
]:
    num_signal = int(signal_hist.sum())
    num_background = int(background_hist.sum())

    if num_signal == 0 or num_background == 0:
        return None, None, None, None, None

    signal_tail = np.cumsum(
        signal_hist[::-1],
        dtype=np.int64,
    )[::-1]
    background_tail = np.cumsum(
        background_hist[::-1],
        dtype=np.int64,
    )[::-1]

    signal_efficiency = (
        signal_tail.astype(np.float64)
        / num_signal
    )
    background_efficiency = (
        background_tail.astype(np.float64)
        / num_background
    )

    working_point_index = int(
        np.abs(
            signal_efficiency
            - target_efficiency
        ).argmin()
    )

    selected_signal_efficiency = float(
        signal_efficiency[working_point_index]
    )
    selected_background_efficiency = float(
        background_efficiency[working_point_index]
    )

    threshold = (
        working_point_index
        / ROC_BINS
    )

    background_rejection = (
        1.0 / selected_background_efficiency
        if selected_background_efficiency > 0
        else None
    )

    background_below = (
        np.cumsum(background_hist, dtype=np.int64)
        - background_hist
    )

    auc_numerator = np.sum(
        signal_hist.astype(np.float64)
        * (
            background_below.astype(np.float64)
            + 0.5 * background_hist
        )
    )

    auc = float(
        auc_numerator
        / (num_signal * num_background)
    )

    return (
        selected_signal_efficiency,
        selected_background_efficiency,
        background_rejection,
        float(threshold),
        auc,
    )


def _display_name(name: str) -> str:
    return (
        name.removeprefix("label_")
        .replace("/", "_")
        .replace(" ", "_")
    )


def _working_point_suffix(
    working_point: float,
) -> str:
    percentage = working_point * 100.0

    if percentage.is_integer():
        return f"{int(percentage)}pct"

    return (
        f"{percentage:g}"
        .replace(".", "p")
        + "pct"
    )
