#!/usr/bin/env bash
# Learned-region presets with configurable training and Weaver argument passthrough.
set -euo pipefail
usage() {
    cat <<'USAGE'
Usage: bash scripts/run_representation_experiment.sh PRESET [training options] [Weaver arguments...]
Presets: full_sv, regions16_sv, regions8_sv, regions16_rank2, regions8_rank2,
         regions8_wide, regions16_wide; reference controls: part_small,
         full_sv_pair, full_sv_cls, full_sv_pair_cls.
Training options: --num-epochs N, --seed N, --train-percentage N (integer 1-100).
Both --option VALUE and --option=VALUE are accepted for these options.
Defaults: EPOCHS=80, SEED=123, TRAIN_PERCENTAGE=10; command-line values win.
Other arguments pass through to Weaver via run_jetclass2.sh, including -o options,
batch size, learning rate, resume and test-only options. Presets default to kin data.
Environment: DATA_PATH, OUTPUT_PATH, COMMENT, DDP_NGPUS, BATCH_SIZE, NUM_WORKERS.
AIO_DRY_RUN=1 prints the launch without accessing data or creating output directories.
Set COMMENT yourself; the launcher passes it through unchanged.
GEOMETRY_CHECKPOINT optionally loads exported learned geometry and freezes it.
Model/data overrides are allowed; use a distinct COMMENT for custom variants.
USAGE
}
if [[ $# -eq 0 || "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
preset=$1; shift
case "$preset" in
    full_sv|full_sv_pair|full_sv_cls|full_sv_pair_cls|part_small|regions16_sv|regions8_sv|regions16_rank2|regions8_rank2|regions8_wide|regions16_wide) ;;
    *) echo "Unknown preset: $preset" >&2; exit 2 ;;
esac
epochs=${EPOCHS:-80}
seed=${SEED:-123}
train_percentage=${TRAIN_PERCENTAGE:-10}
extra_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-epochs|--seed|--train-percentage)
            if [[ $# -lt 2 || "$2" == --* ]]; then
                echo "Missing value for $1" >&2; exit 2
            fi
            option=$1; value=$2; shift 2 ;;
        --num-epochs=*|--seed=*|--train-percentage=*)
            option=${1%%=*}; value=${1#*=}; shift ;;
        *)
            extra_args+=("$1"); shift; continue ;;
    esac
    case "$option" in
        --num-epochs) epochs=$value ;;
        --seed) seed=$value ;;
        --train-percentage) train_percentage=$value ;;
    esac
done
if ! [[ "$epochs" =~ ^[1-9][0-9]*$ ]]; then
    echo "--num-epochs / EPOCHS must be a positive integer." >&2; exit 2
fi
if ! [[ "$seed" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "--seed / SEED must be a nonnegative integer." >&2; exit 2
fi
if ! [[ "$train_percentage" =~ ^([1-9]|[1-9][0-9]|100)$ ]]; then
    echo "--train-percentage / TRAIN_PERCENTAGE must be an integer from 1 to 100." >&2; exit 2
fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)
export EPOCHS="$epochs"
geometry_args=()
if [[ -n "${GEOMETRY_CHECKPOINT:-}" ]]; then
    if [[ "$preset" != regions* ]]; then echo "Full reference has no region checkpoint." >&2; exit 2; fi
    if [[ ! -f "$GEOMETRY_CHECKPOINT" ]]; then echo "Missing GEOMETRY_CHECKPOINT" >&2; exit 2; fi
    # Weaver uses literal_eval for options; JSON is a safe quoted string literal here.
    geometry_literal=$(python -c 'import json, os; print(json.dumps(os.path.abspath(os.environ["GEOMETRY_CHECKPOINT"])))')
    geometry_args=(-o geometry_checkpoint "$geometry_literal" -o freeze_geometry True)
fi
export PYTHONPATH="${repo_root}/src:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
model=RegionLLoCa
if [[ "$preset" == part_small ]]; then model=ParT; fi
cmd=(bash "${script_dir}/run_jetclass2.sh" "$model" kin --train-percentage "$train_percentage" --seed "$seed"
     -o preset "'$preset'" "${geometry_args[@]}" "${extra_args[@]}")
if [[ "${AIO_DRY_RUN:-0}" == 1 ]]; then
    printf 'EPOCHS=%q ' "$EPOCHS"
    printf '%q ' "${cmd[@]}"; printf '\n'; exit 0
fi
cd -- "$repo_root"
exec "${cmd[@]}"
