#!/bin/bash
# Frozen-embedding head test: does a trained tagger miss information that
# hand-made observables carry?  See docs/embedding_head_test.md.
#
#   CHECKPOINT=/path/net_best_epoch_state.pt ./scripts/run_embedding_head_test.sh
#
# Steps (all resumable; rerun to continue):
#   extract      embeddings, logits and constituents from held-out files (GPU)
#   observables  hand-made observables for every exported jet (CPU, parallel)
#   heads        train and score the heads (GPU)
#
# QUICK=1 runs the whole chain on a handful of files in a few minutes, into
# ${NAME}_quick, to check the setup before the full run. Its numbers mean nothing.
set -euo pipefail

: "${DATA_PATH:?Set DATA_PATH as for run_jetclass2.sh}"
: "${OUTPUT_PATH:?Set OUTPUT_PATH as for run_jetclass2.sh}"
: "${CHECKPOINT:?Set CHECKPOINT to the trained net_best_epoch_state.pt}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/src:${ROOT}:${PYTHONPATH:-}"

PYTHON=${PYTHON:-python}
QUICK=${QUICK:-0}
NAME=${NAME:-embedding_head_test}
if [[ "${QUICK}" == 1 ]]; then
    NAME="${NAME}_quick"
    HEAD_TRAIN_RES2P=${HEAD_TRAIN_RES2P:-100:101}; HEAD_TRAIN_RES34P=${HEAD_TRAIN_RES34P:-430:433}
    HEAD_TRAIN_QCD=${HEAD_TRAIN_QCD:-140:141}
    HEAD_VAL_RES2P=${HEAD_VAL_RES2P:-200:200}; HEAD_VAL_RES34P=${HEAD_VAL_RES34P:-860:861}
    HEAD_VAL_QCD=${HEAD_VAL_QCD:-280:280}
    TEST_RES2P=${TEST_RES2P:-250:251}; TEST_RES34P=${TEST_RES34P:-1075:1078}; TEST_QCD=${TEST_QCD:-350:351}
    TRAIN_KEEP=${TRAIN_KEEP:-0.1}; VAL_KEEP=${VAL_KEEP:-0.1}; TEST_KEEP=${TEST_KEEP:-0.1}
    HEAD_ARGS=${HEAD_ARGS:-"--arms embedding embedding+all obs_only --seeds 0 --epochs 3"}
fi
OUT="${OUTPUT_PATH}/${NAME}"
DATADIR="${DATA_PATH}/JetClass2/Pythia/data"
STEPS=${STEPS:-"extract observables heads"}

# Model. FULL_SIZE=1 is the default ParT (width 128, 8 heads, 8 blocks) trained by
# integrations/weaver/ParT.py; FULL_SIZE=0 uses PRESET as it is.
PRESET=${PRESET:-part_pair_mlp64_d4}
FULL_SIZE=${FULL_SIZE:-1}
DATA_CONFIG=${DATA_CONFIG:-configs/datasets/jetclass2/JetClassII_kin.yaml}

# Files, as inclusive index ranges. The head must be trained on jets the
# checkpoint never saw. A 10% run used Res2P 0-19, Res34P 0-85 and QCD 0-27, so
# the defaults below take the same number of files from the middle of the
# training range. Change them if the checkpoint used more than half the files.
CHECKPOINT_TRAIN_PERCENTAGE=${CHECKPOINT_TRAIN_PERCENTAGE:-10}
HEAD_TRAIN_RES2P=${HEAD_TRAIN_RES2P:-100:119}
HEAD_TRAIN_RES34P=${HEAD_TRAIN_RES34P:-430:515}
HEAD_TRAIN_QCD=${HEAD_TRAIN_QCD:-140:167}
HEAD_VAL_RES2P=${HEAD_VAL_RES2P:-200:204}
HEAD_VAL_RES34P=${HEAD_VAL_RES34P:-860:880}
HEAD_VAL_QCD=${HEAD_VAL_QCD:-280:286}
TEST_RES2P=${TEST_RES2P:-250:299}
TEST_RES34P=${TEST_RES34P:-1075:1289}
TEST_QCD=${TEST_QCD:-350:419}

# Random fraction of each file to export (about 2M / 0.4M / 3M jets by default).
TRAIN_KEEP=${TRAIN_KEEP:-0.20}
VAL_KEEP=${VAL_KEEP:-0.15}
TEST_KEEP=${TEST_KEEP:-0.12}

BATCH_SIZE=${BATCH_SIZE:-512}
WORKERS=${WORKERS:-$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))}
EXTRACT_ARGS=${EXTRACT_ARGS:-}      # e.g. "--amp" or "--model-option num_layers=4"
HEAD_ARGS=${HEAD_ARGS:-}            # e.g. "--arms embedding embedding+all --seeds 0 1 2"
REWEIGHT_YAML=${REWEIGHT_YAML:-}    # the training run's *.auto.yaml, to reuse its exact weights

if (( CHECKPOINT_TRAIN_PERCENTAGE > 50 )) && [[ -z "${ALLOW_OVERLAP:-}" ]]; then
    echo "The default head-training files overlap a checkpoint trained on more than 50% of the data." >&2
    echo "Set HEAD_TRAIN_* to files the checkpoint never saw, then ALLOW_OVERLAP=1." >&2
    exit 2
fi

range_files() {   # range_files PREFIX START:END -> GROUP:path arguments
    local prefix=$1 start=${2%%:*} end=${2##*:} i idx
    for (( i=start; i<=end; i++ )); do
        printf -v idx '%04d' "${i}"
        printf '%s:%s/%s_%s.parquet\n' "${prefix}" "${DATADIR}" "${prefix}" "${idx}"
    done
}

extract_split() {   # extract_split SPLIT KEEP RES2P RES34P QCD
    local split=$1 keep=$2
    local files=()
    mapfile -t files < <(range_files Res2P "$3"; range_files Res34P "$4"; range_files QCD "$5")
    local model=(--preset "${PRESET}")
    [[ "${FULL_SIZE}" == 1 ]] && model+=(--full-size-part)
    echo "---- extract ${split}: ${#files[@]} files, keep fraction ${keep}"
    # shellcheck disable=SC2086
    "${PYTHON}" scripts/extract_embeddings.py \
        --checkpoint "${CHECKPOINT}" --data-config "${DATA_CONFIG}" "${model[@]}" \
        --split "${split}" --keep-fraction "${keep}" --batch-size "${BATCH_SIZE}" \
        --output-dir "${OUT}/shards" --files "${files[@]}" ${EXTRACT_ARGS}
}

echo "Output: ${OUT}"
echo "Steps:  ${STEPS}"
mkdir -p "${OUT}"

for step in ${STEPS}; do
    case "${step}" in
        extract)
            extract_split train "${TRAIN_KEEP}" "${HEAD_TRAIN_RES2P}" "${HEAD_TRAIN_RES34P}" "${HEAD_TRAIN_QCD}"
            extract_split val   "${VAL_KEEP}"   "${HEAD_VAL_RES2P}"   "${HEAD_VAL_RES34P}"   "${HEAD_VAL_QCD}"
            extract_split test  "${TEST_KEEP}"  "${TEST_RES2P}"       "${TEST_RES34P}"       "${TEST_QCD}"
            ;;
        observables)
            "${PYTHON}" scripts/compute_jet_observables.py \
                "${OUT}/shards/train" "${OUT}/shards/val" "${OUT}/shards/test" --workers "${WORKERS}"
            ;;
        heads)
            reweight=()
            [[ -n "${REWEIGHT_YAML}" ]] && reweight=(--reweight-yaml "${REWEIGHT_YAML}")
            # shellcheck disable=SC2086
            "${PYTHON}" scripts/train_embedding_head.py --shards "${OUT}/shards" --output "${OUT}/heads" \
                --data-config "${DATA_CONFIG}" "${reweight[@]}" ${HEAD_ARGS}
            ;;
        *)
            echo "Unknown step ${step}; choose from: extract observables heads" >&2
            exit 2
            ;;
    esac
done
