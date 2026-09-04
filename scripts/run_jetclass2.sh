#!/bin/bash
set -euo pipefail


printf -v RUN_COMMAND '%q ' "$0" "$@"
RUN_COMMAND="${RUN_COMMAND% }"
export RUN_COMMAND

if [[ -z "${DATA_PATH:-}" ]]; then
    echo "Error: The DATA_PATH environment variable is not set."
    exit 1
fi

if [[ -z "${OUTPUT_PATH:-}" ]]; then
    echo "Error: The OUTPUT_PATH environment variable is not set."
    exit 1
fi

if [[ -z "${COMMENT:-}" ]]; then
    echo "Error: The COMMENT environment variable is not set."
    exit 1
fi

RUN_COMMENT="${COMMENT}"
export RUN_COMMENT

echo "command: ${RUN_COMMAND}"
echo "comment: ${RUN_COMMENT}"


DATADIR="${DATA_PATH}/JetClass2/Pythia/data"
OUTPUT_VOL_DIR="${OUTPUT_PATH}"


MODEL_NAME=${1:-}
if ! [[ "${MODEL_NAME}" =~ ^(ParT|AdaParT)$ ]]; then
    echo "Invalid model ${MODEL_NAME:-<empty>}! Valid options: ParT, AdaParT."
    exit 1
fi
shift

if [[ -z "${1:-}" ]] || [[ "${1:-}" == --* ]]; then
    echo "Error: The second argument must be the feature type (full, kin, kinpid)."
    exit 1
fi

FEATURE_TYPE=$1
shift

if ! [[ "${FEATURE_TYPE}" =~ ^(full|kin|kinpid)$ ]]; then
    echo "Invalid feature type ${FEATURE_TYPE}!"
    exit 1
fi

# ---------------------------------------------------------------------------
# Script-specific arguments
#
# Normal training:
#   ./run_jetclass2.sh ParT kin --train-percentage 10 [Weaver args...]
#
# Test-only, explicit checkpoint:
#   ./run_jetclass2.sh ParT kin \
#       --test-only \
#       --weights /path/to/net_best_epoch_state.pt \
#       [Weaver args...]
#
# Supplying --weights automatically enables test-only mode:
#   ./run_jetclass2.sh ParT kin \
#       --weights /path/to/net_best_epoch_state.pt \
#       [Weaver args...]
#
# Test-only using the default checkpoint for COMMENT:
#   ./run_jetclass2.sh ParT kin --test-only [Weaver args...]
# ---------------------------------------------------------------------------

TRAIN_PERCENTAGE=100
TEST_ONLY=false
WEIGHTS=""
WEAVER_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --train-percentage)
            if [[ $# -lt 2 ]]; then
                echo "Error: --train-percentage requires a value."
                exit 1
            fi
            TRAIN_PERCENTAGE="$2"
            shift 2
            ;;
        --test-only)
            TEST_ONLY=true
            shift
            ;;
        --weights)
            if [[ $# -lt 2 ]]; then
                echo "Error: --weights requires a checkpoint path."
                exit 1
            fi
            WEIGHTS="$2"
            TEST_ONLY=true
            shift 2
            ;;
        *)
            WEAVER_ARGS+=("$1")
            shift
            ;;
    esac
done

if ! [[ "${TRAIN_PERCENTAGE}" =~ ^[0-9]+$ ]] \
    || (( TRAIN_PERCENTAGE < 1 || TRAIN_PERCENTAGE > 100 )); then
    echo "Error: --train-percentage must be an integer from 1 to 100."
    exit 1
fi


suffix=${COMMENT}
NGPUS=${DDP_NGPUS:-1}

if (( NGPUS > 1 )); then
    CMD=(
        torchrun
        --standalone
        --nnodes=1
        --nproc_per_node="${NGPUS}"
        "$(command -v weaver)"
        --backend nccl
    )
else
    CMD=(weaver)
fi

epochs=${EPOCHS:-80}
samples_per_epoch=$(( TRAIN_PERCENTAGE * 10000 * 1024 / (100 * NGPUS) ))
samples_per_epoch_val=$(( 2500 * 1024 ))

NUM_WORKERS=${NUM_WORKERS:-5}
BATCH_SIZE=${BATCH_SIZE:-512}
NUM_CLASSES=${JETCLASS2_NUM_CLASSES:-188}

DATA_OPTS=(
    --num-workers "${NUM_WORKERS}"
    --fetch-step 1.0
    --data-split-num 200
)

BATCH_OPTS=(
    --batch-size "${BATCH_SIZE}"
    --start-lr 5e-4
)

MODEL_OPTS=(
    "integrations/weaver/${MODEL_NAME}.py"
    --use-amp
    --amp-dtype bf16
    -o num_classes "${NUM_CLASSES}"
    -o fc_params "[(512,0.1)]"
)

DATA_CONFIG="configs/datasets/jetclass2/JetClassII_${FEATURE_TYPE}.yaml"


RUN_DIR="${OUTPUT_VOL_DIR}/${MODEL_NAME}/JetClass2_${FEATURE_TYPE}/${suffix}"
TRAINING_DIR="${RUN_DIR}/training"
LOG_DIR="${RUN_DIR}/logs"
TENSORBOARD_DIR="${RUN_DIR}/tensorboard"
RESULTS_DIR="${RUN_DIR}/results"
WANDB_DIR="${RUN_DIR}/wandb"

mkdir -p \
    "${TRAINING_DIR}" \
    "${LOG_DIR}" \
    "${TENSORBOARD_DIR}" \
    "${RESULTS_DIR}" \
    "${WANDB_DIR}"

count_for_percentage() {
    local total=$1
    echo $(( (total * TRAIN_PERCENTAGE + 99) / 100 ))
}

append_labelled_range() {
    local -n out=$1
    local label=$2
    local prefix=$3
    local start=$4
    local end=$5
    local i

    for (( i=start; i<=end; i++ )); do
        printf -v idx '%04d' "${i}"
        out+=("${label}:${DATADIR}/${prefix}_${idx}.parquet")
    done
}

append_plain_range() {
    local -n out=$1
    local prefix=$2
    local start=$3
    local end=$4
    local i

    for (( i=start; i<=end; i++ )); do
        printf -v idx '%04d' "${i}"
        out+=("${DATADIR}/${prefix}_${idx}.parquet")
    done
}

TEST_FILES=()
append_labelled_range TEST_FILES Res2P  Res2P  250 299
append_labelled_range TEST_FILES Res34P Res34P 1075 1289
append_labelled_range TEST_FILES QCD    QCD    350 419

# ---------------------------------------------------------------------------
# Test-only mode
# ---------------------------------------------------------------------------

if [[ "${TEST_ONLY}" == true ]]; then
    if [[ -z "${WEIGHTS}" ]]; then
        WEIGHTS="${TRAINING_DIR}/net_best_epoch_state.pt"
    fi

    if [[ ! -f "${WEIGHTS}" ]]; then
        echo "Error: Test checkpoint does not exist:"
        echo "  ${WEIGHTS}"
        exit 1
    fi

    echo "------------------------------------------------------------"
    echo "JetClass-II TEST ONLY"
    echo "------------------------------------------------------------"
    echo "Data dir:       ${DATADIR}"
    echo "Data config:    ${DATA_CONFIG}"
    echo "Model:          ${MODEL_NAME}"
    echo "Feature type:   ${FEATURE_TYPE}"
    echo "Run directory:  ${RUN_DIR}"
    echo "Weights:        ${WEIGHTS}"
    echo "Test files:     ${#TEST_FILES[@]}"
    echo "Number classes: ${NUM_CLASSES}"
    echo "Batch size:     ${BATCH_SIZE}"
    echo "GPUs:           ${NGPUS}"
    echo "Results dir:    ${RESULTS_DIR}"
    echo "------------------------------------------------------------"

    "${CMD[@]}" \
        --predict \
        --data-test "${TEST_FILES[@]}" \
        --data-config "${DATA_CONFIG}" \
        --network-config "${MODEL_OPTS[@]}" \
        --model-prefix "${WEIGHTS}" \
        "${DATA_OPTS[@]}" \
        --batch-size "${BATCH_SIZE}" \
        --gpus 0 \
        --log "${LOG_DIR}/test.log" \
        --predict-output "${RESULTS_DIR}/pred.root" \
        --tensorboard "${TENSORBOARD_DIR}" \
        --wandb-dir "${WANDB_DIR}" \
        "${WEAVER_ARGS[@]}"

    exit 0
fi

# ---------------------------------------------------------------------------
# Training mode
# ---------------------------------------------------------------------------

TRAIN_FILES=()
VAL_FILES=()

N_RES2P=$(count_for_percentage 200)
N_RES34P=$(count_for_percentage 860)
N_QCD=$(count_for_percentage 280)

append_labelled_range TRAIN_FILES Res2P  Res2P  0 $((N_RES2P - 1))
append_labelled_range TRAIN_FILES Res34P Res34P 0 $((N_RES34P - 1))
append_labelled_range TRAIN_FILES QCD    QCD    0 $((N_QCD - 1))

append_plain_range VAL_FILES Res2P  200 249
append_plain_range VAL_FILES Res34P 860 1074
append_plain_range VAL_FILES QCD    280 349

echo "------------------------------------------------------------"
echo "JetClass-II TRAIN + TEST"
echo "------------------------------------------------------------"
echo "JetClass-II data dir: ${DATADIR}"
echo "Data config: ${DATA_CONFIG}"
echo "Model: ${MODEL_NAME}"
echo "Feature type: ${FEATURE_TYPE}"
echo "Training percentage: ${TRAIN_PERCENTAGE}%"
echo "Training files: ${#TRAIN_FILES[@]} (Res2P=${N_RES2P}, Res34P=${N_RES34P}, QCD=${N_QCD})"
echo "Validation files: ${#VAL_FILES[@]}"
echo "Test files: ${#TEST_FILES[@]}"
echo "Number of classes: ${NUM_CLASSES}"
echo "Epochs: ${epochs}"
echo "Samples/epoch/GPU-normalized: ${samples_per_epoch}"
echo "Validation samples/epoch: ${samples_per_epoch_val}"
echo "Batch size: ${BATCH_SIZE}"
echo "Start LR: 5e-4"
echo "GPUs: ${NGPUS}"
echo "------------------------------------------------------------"

"${CMD[@]}" \
    --data-train "${TRAIN_FILES[@]}" \
    --data-val "${VAL_FILES[@]}" \
    --data-test "${TEST_FILES[@]}" \
    --data-config "${DATA_CONFIG}" \
    --network-config "${MODEL_OPTS[@]}" \
    --model-prefix "${TRAINING_DIR}/net" \
    "${DATA_OPTS[@]}" \
    "${BATCH_OPTS[@]}" \
    --samples-per-epoch "${samples_per_epoch}" \
    --samples-per-epoch-val "${samples_per_epoch_val}" \
    --num-epochs "${epochs}" \
    --gpus 0 \
    --optimizer ranger \
    --log "${LOG_DIR}/train.log" \
    --predict-output "${RESULTS_DIR}/pred.root" \
    --tensorboard "${TENSORBOARD_DIR}" \
    --wandb-dir "${WANDB_DIR}" \
    "${WEAVER_ARGS[@]}"
