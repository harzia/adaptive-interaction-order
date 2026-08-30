#!/bin/bash
set -euo pipefail

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

# Expected layout:
#   ${DATA_PATH}/JetClassII/Pythia/Res2P_0000.parquet
#   ${DATA_PATH}/JetClassII/Pythia/Res34P_0000.parquet
#   ${DATA_PATH}/JetClassII/Pythia/QCD_0000.parquet
DATADIR="${DATA_PATH}/JetClassII/Pythia"
OUTPUT_VOL_DIR="${OUTPUT_PATH}"

echo "args: $*"

MODEL_NAME=${1:-}
if ! [[ "${MODEL_NAME}" =~ ^(ParT|AdaParT)$ ]]; then
    echo "Invalid model ${MODEL_NAME:-<empty>}! Valid options: ParT, AdaParT."
    exit 1
fi
shift

if [[ -z "${1:-}" ]] || [[ "${1:-}" == --* ]]; then
    echo "Error: The second argument must be the feature type (e.g., full, kin, kinpid)."
    exit 1
fi
FEATURE_TYPE=$1
shift
if ! [[ "${FEATURE_TYPE}" =~ ^(full|kin|kinpid)$ ]]; then
    echo "Invalid feature type ${FEATURE_TYPE}!"
    exit 1
fi

TRAIN_PERCENTAGE=100
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
        *)
            WEAVER_ARGS+=("$1")
            shift
            ;;
    esac
done

if ! [[ "${TRAIN_PERCENTAGE}" =~ ^[0-9]+$ ]] || (( TRAIN_PERCENTAGE < 1 || TRAIN_PERCENTAGE > 100 )); then
    echo "Error: --train-percentage must be an integer from 1 to 100."
    exit 1
fi

suffix=${COMMENT}
NGPUS=${DDP_NGPUS:-1}

if (( NGPUS > 1 )); then
    CMD=(torchrun --standalone --nnodes=1 --nproc_per_node="${NGPUS}" "$(command -v weaver)" --backend nccl)
else
    CMD=(weaver)
fi

# JetClass-II's official Weaver recipe uses more workers and a split data loader.
NUM_WORKERS=${NUM_WORKERS:-5}
if (( TRAIN_PERCENTAGE == 1 )); then
    NUM_WORKERS=1
fi

# Keep the original training length, but use the same full-dataset samples/epoch
# normalization as the JetClass-II reference recipe (10.24M at 100%).
epochs=${EPOCHS:-50}
samples_per_epoch=$(( TRAIN_PERCENTAGE * 1000 * 1024 / (10 * NGPUS) ))
samples_per_epoch_val=$(( 2500 * 1024 ))

DATA_OPTS=(--num-workers "${NUM_WORKERS}" --fetch-step 1.0 --data-split-num 200)
BATCH_OPTS=(--batch-size 512 --start-lr 1e-3)

# JetClass-II has 188 classes. Override with JETCLASS2_NUM_CLASSES if needed.
NUM_CLASSES=${JETCLASS2_NUM_CLASSES:-188}
MODEL_OPTS=("model/${MODEL_NAME}.py" --use-amp --amp-dtype bf16 -o num_classes "${NUM_CLASSES}")

# Allow an explicit config path. Otherwise try the repository layouts commonly
# used by this project and by the public Sophon/JetClass-II reference code.
if [[ -n "${JETCLASS2_CONFIG:-}" ]]; then
    DATA_CONFIG="${JETCLASS2_CONFIG}"
else
    CONFIG_CANDIDATES=(
        "dataset/JetClassII/JetClassII_${FEATURE_TYPE}.yaml"
        "data/JetClassII/JetClassII_${FEATURE_TYPE}.yaml"
    )
    DATA_CONFIG=""
    for cfg in "${CONFIG_CANDIDATES[@]}"; do
        if [[ -f "${cfg}" ]]; then
            DATA_CONFIG="${cfg}"
            break
        fi
    done
    if [[ -z "${DATA_CONFIG}" ]]; then
        echo "Error: Could not find a JetClass-II data config for feature type '${FEATURE_TYPE}'."
        echo "Tried: ${CONFIG_CANDIDATES[*]}"
        echo "Set JETCLASS2_CONFIG=/path/to/JetClassII_${FEATURE_TYPE}.yaml to override."
        exit 1
    fi
fi

RUN_DIR="${OUTPUT_VOL_DIR}/${MODEL_NAME}/JetClassII_${FEATURE_TYPE}/${suffix}"
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

# JetClass-II file ranges (100k jets/file):
#   train: Res2P 0000-0199, Res34P 0000-0859, QCD 0000-0279
#   val:   Res2P 0200-0249, Res34P 0860-1074, QCD 0280-0349
#   test:  Res2P 0250-0299, Res34P 1075-1289, QCD 0350-0419
# For --train-percentage, take the same fraction from each training group.
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

TRAIN_FILES=()
VAL_FILES=()
TEST_FILES=()

N_RES2P=$(count_for_percentage 200)
N_RES34P=$(count_for_percentage 860)
N_QCD=$(count_for_percentage 280)

append_labelled_range TRAIN_FILES Res2P  Res2P  0 $((N_RES2P - 1))
append_labelled_range TRAIN_FILES Res34P Res34P 0 $((N_RES34P - 1))
append_labelled_range TRAIN_FILES QCD    QCD    0 $((N_QCD - 1))

append_plain_range VAL_FILES Res2P  200 249
append_plain_range VAL_FILES Res34P 860 1074
append_plain_range VAL_FILES QCD    280 349

append_labelled_range TEST_FILES Res2P  Res2P  250 299
append_labelled_range TEST_FILES Res34P Res34P 1075 1289
append_labelled_range TEST_FILES QCD    QCD    350 419

echo "JetClass-II data dir: ${DATADIR}"
echo "Data config: ${DATA_CONFIG}"
echo "Training percentage: ${TRAIN_PERCENTAGE}%"
echo "Training files: ${#TRAIN_FILES[@]} (Res2P=${N_RES2P}, Res34P=${N_RES34P}, QCD=${N_QCD})"
echo "Validation files: ${#VAL_FILES[@]}"
echo "Test files: ${#TEST_FILES[@]}"
echo "Number of classes: ${NUM_CLASSES}"

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