#!/usr/bin/env bash
# Matched full-attention ParT / angular-state experiments.
set -euo pipefail
usage() {
    cat <<'USAGE'
Usage: bash scripts/run_angular_experiment.sh PRESET [training options] [Weaver arguments...]
Baselines: part_d2, part_d4, part_d8, part_w64_d2.
Angular: moments_l{1,2}_d{2,4}, moments_l1_c5_d2,
         moments_l2_noprod_d2, moments_l2_reset_d2.
Options: --num-epochs N, --seed N, --train-percentage N (also --option=value).
Defaults: EPOCHS=80, SEED=123, TRAIN_PERCENTAGE=10. CLI takes precedence.
All variants use full particle attention, ParT pair bias, and two class blocks.
Environment: DATA_PATH, OUTPUT_PATH, COMMENT, DDP_NGPUS, BATCH_SIZE, NUM_WORKERS.
Default COMMENT: angular_<preset>_seed<seed>. Set a new COMMENT for custom overrides.
AIO_DRY_RUN=1 prints the command without accessing training data.
Both families use the AngularParT adapter; baseline presets instantiate ordinary ParT.
USAGE
}
if [[ $# == 0 || "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
preset=$1; shift
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)
if ! [[ "$preset" =~ ^[a-zA-Z0-9_]+$ ]] || [[ "$preset" == common ]] || [[ ! -f "$repo_root/configs/experiments/angular_moments/$preset.json" ]]; then
    echo "Unknown angular preset: $preset" >&2; exit 2
fi
epochs=${EPOCHS:-80}; seed=${SEED:-123}; percentage=${TRAIN_PERCENTAGE:-10}
extra=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-epochs|--seed|--train-percentage)
            if [[ $# -lt 2 || "$2" == --* ]]; then echo "Missing value for $1" >&2; exit 2; fi
            option=$1; value=$2; shift 2 ;;
        --num-epochs=*|--seed=*|--train-percentage=*)
            option=${1%%=*}; value=${1#*=}; shift ;;
        *) extra+=("$1"); shift; continue ;;
    esac
    case "$option" in
        --num-epochs) epochs=$value ;;
        --seed) seed=$value ;;
        --train-percentage) percentage=$value ;;
    esac
done
if ! [[ "$epochs" =~ ^[1-9][0-9]*$ && "$seed" =~ ^(0|[1-9][0-9]*)$ && "$percentage" =~ ^([1-9]|[1-9][0-9]|100)$ ]]; then
    echo "Require positive epochs, nonnegative seed, and training percentage 1-100" >&2; exit 2
fi
export EPOCHS="$epochs"
export COMMENT="${COMMENT:-angular_${preset}_seed${seed}}"
export PYTHONPATH="${repo_root}/src:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
command=(bash "$script_dir/run_jetclass2.sh" AngularParT kin --train-percentage "$percentage" --seed "$seed"
         -o preset "'$preset'" "${extra[@]}")
if [[ "${AIO_DRY_RUN:-0}" == 1 ]]; then
    printf 'EPOCHS=%q COMMENT=%q ' "$EPOCHS" "$COMMENT"
    printf '%q ' "${command[@]}"; printf '\n'; exit 0
fi
cd -- "$repo_root"
exec "${command[@]}"
