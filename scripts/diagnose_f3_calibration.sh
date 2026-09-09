#!/usr/bin/env bash

set -euo pipefail


FEATURE_TYPE=${1:-kin}
TRAIN_PERCENTAGE=${2:-10}

DIAG_BATCHES=${F3_DIAG_BATCHES:-5}
DIAG_BATCH_SIZE=${F3_DIAG_BATCH_SIZE:-64}

SEED_STRING=${F3_DIAG_SEEDS:-123,124,125}
IFS=',' read -r -a SEEDS <<< "${SEED_STRING}"

OUTPUT_BASE=${OUTPUT_PATH:-/output/runs}

DIAG_DIR=${F3_DIAG_OUTPUT_DIR:-\
${OUTPUT_BASE}/diagnostics/f3_calibration/\
${FEATURE_TYPE}-${TRAIN_PERCENTAGE}pct}

mkdir -p "${DIAG_DIR}"


echo "============================================================"
echo "F3 calibration diagnostics"
echo "============================================================"
echo "Features:           ${FEATURE_TYPE}"
echo "Training percent:   ${TRAIN_PERCENTAGE}%"
echo "Batches / seed:     ${DIAG_BATCHES}"
echo "Diagnostic batch:   ${DIAG_BATCH_SIZE}"
echo "Seeds:              ${SEEDS[*]}"
echo "Output:             ${DIAG_DIR}"
echo "============================================================"


for SEED in "${SEEDS[@]}"; do
    OUT="${DIAG_DIR}/seed-${SEED}.json"

    echo
    echo "------------------------------------------------------------"
    echo "Seed ${SEED}"
    echo "------------------------------------------------------------"

    F3_DIAG_BATCHES="${DIAG_BATCHES}" \
    F3_DIAG_OUTPUT="${OUT}" \
    BATCH_SIZE="${DIAG_BATCH_SIZE}" \
        ./scripts/run_jetclass2.sh \
            F3ParTDiag \
            "${FEATURE_TYPE}" \
            --train-percentage "${TRAIN_PERCENTAGE}" \
            --seed "${SEED}"
done


echo
echo "============================================================"
echo "Diagnostics complete"
echo "============================================================"

for FILE in "${DIAG_DIR}"/seed-*.json; do
    echo "${FILE}"
done