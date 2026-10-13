#!/usr/bin/env bash
# Initial learned-region experiments; full protocol matches run_triplet_experiment.sh.
set -euo pipefail
usage() {
    cat <<'USAGE'
Usage: bash scripts/run_representation_experiment.sh PRESET [Weaver logging arguments...]
Presets: full_sv, regions16_sv, regions8_sv, regions16_rank2, regions8_rank2,
         regions8_wide (optional: regions16_wide).
Environment: DATA_PATH, OUTPUT_PATH, SEED (123), DDP_NGPUS, BATCH_SIZE.
EPOCHS must be 80. Protocol is 10% JetClass-II, kinematics only, from scratch.
AIO_DRY_RUN=1 prints the launch without accessing data or creating output directories.
AIO_RUN_COMMENT overrides the preset/seed name.
GEOMETRY_CHECKPOINT optionally loads exported learned geometry and freezes it.
For a short smoke run or test-only evaluation, use run_jetclass2.sh RegionLLoCa directly.
USAGE
}
if [[ $# -eq 0 || "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
preset=$1; shift
case "$preset" in
    full_sv|regions16_sv|regions8_sv|regions16_rank2|regions8_rank2|regions8_wide|regions16_wide) ;;
    *) echo "Unknown preset: $preset" >&2; exit 2 ;;
esac
if [[ "${EPOCHS:-80}" != 80 ]]; then
    echo "Comparison presets require EPOCHS=80; use the base launcher for smoke runs." >&2; exit 2
fi
for arg in "$@"; do
    case "$arg" in
        -o|--network-option|--network-option=*|--network-config|--network-config=*|--data-config|--data-config=*|--num-epochs|--num-epochs=*|--train-percentage|--train-percentage=*|--seed|--seed=*|--weights|--weights=*|--test-only|--load-model-weights|--load-model-weights=*|--load-epoch|--load-epoch=*)
            echo "Do not override preset model/data/training options: $arg" >&2; exit 2 ;;
    esac
done
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)
seed=${SEED:-123}
export EPOCHS=80
suffix=learned
geometry_args=()
if [[ -n "${GEOMETRY_CHECKPOINT:-}" ]]; then
    if [[ "$preset" == full_sv ]]; then echo "Full reference has no region checkpoint." >&2; exit 2; fi
    if [[ ! -f "$GEOMETRY_CHECKPOINT" ]]; then echo "Missing GEOMETRY_CHECKPOINT" >&2; exit 2; fi
    suffix=frozen
    # Weaver uses literal_eval for options; JSON is a safe quoted string literal here.
    geometry_literal=$(python -c 'import json, os; print(json.dumps(os.path.abspath(os.environ["GEOMETRY_CHECKPOINT"])))')
    geometry_args=(-o geometry_checkpoint "$geometry_literal" -o freeze_geometry True)
fi
export COMMENT=${AIO_RUN_COMMENT:-${preset}-${suffix}-10pct-seed${seed}}
export PYTHONPATH="${repo_root}/src:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
cmd=(bash "${script_dir}/run_jetclass2.sh" RegionLLoCa kin --train-percentage 10 --seed "$seed"
     -o preset "'$preset'" "${geometry_args[@]}" "$@")
if [[ "${AIO_DRY_RUN:-0}" == 1 ]]; then
    printf 'EPOCHS=%q COMMENT=%q ' "$EPOCHS" "$COMMENT"
    printf '%q ' "${cmd[@]}"; printf '\n'; exit 0
fi
cd -- "$repo_root"
exec "${cmd[@]}"
