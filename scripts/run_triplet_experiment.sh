#!/usr/bin/env bash
# Fixed 80-epoch, 10%-data, kinematics-only experiments. Extra arguments go to Weaver.
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: bash scripts/run_triplet_experiment.sh PRESET [Weaver logging arguments...]

Presets:
  attended-one      Compute after block 4; inject only in block 5.
  attended-reuse    Original reference: compute after 4; inject in 5,6,7,8.
  summed-one        Summed matrix-multiplication pool; inject only in block 5.
  summed-reuse      Summed matrix-multiplication pool; inject in 5,6,7,8.
  attended-refresh  Compute after 4 and 6; inject in 5,6,7,8, replacing the old update.
  summed-refresh    Same refresh schedule with the summed matrix-multiplication pool.

Environment: DATA_PATH and OUTPUT_PATH as for run_jetclass2.sh; SEED defaults to 123.
AIO_DRY_RUN=1 prints the exact launch without starting training.
AIO_RUN_COMMENT overrides the generated unique run name. An existing COMMENT is ignored.
EPOCHS, if set, must be 80. These are from-scratch runs, not warm starts.
USAGE
}

if [[ $# -eq 0 || "${1:-}" == --help || "${1:-}" == -h ]]; then
    usage
    exit 0
fi
preset=$1
shift
spec=triangle
dense_matmul=False
inject='[5,6,7,8]'
refresh='[]'
case "$preset" in
    attended-one) inject='[5]' ;;
    attended-reuse) ;;
    summed-one) spec=summed_triangle; dense_matmul=True; inject='[5]' ;;
    summed-reuse) spec=summed_triangle; dense_matmul=True ;;
    attended-refresh) refresh='[6]' ;;
    summed-refresh) spec=summed_triangle; dense_matmul=True; refresh='[6]' ;;
    *) echo "Unknown experiment: $preset" >&2; usage >&2; exit 2 ;;
esac

if [[ "${EPOCHS:-80}" != 80 ]]; then
    echo "These comparison presets require EPOCHS=80. Use run_jetclass2.sh directly for smoke tests." >&2
    exit 2
fi
# Keep model/training choices in the preset. Logging and normal environment options remain available.
for arg in "$@"; do
    case "$arg" in
        -o|--network-option|--network-option=*|--num-epochs|--num-epochs=*|--train-percentage|--train-percentage=*|--seed|--seed=*|--weights|--weights=*|--test-only|--load-model-weights|--load-model-weights=*|--load-epoch|--load-epoch=*)
            echo "Do not override the preset with $arg; use SEED or the base launcher for a different protocol." >&2
            exit 2 ;;
    esac
done

export EPOCHS=80
seed=${SEED:-123}
export COMMENT=${AIO_RUN_COMMENT:-${preset}-b4-10pct-seed${seed}}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)
cmd=(bash "${script_dir}/run_jetclass2.sh" F3ParT kin
     --train-percentage 10 --seed "$seed"
     -o f3_after 4 -o f3_refresh_after "$refresh" -o f3_inject_blocks "$inject"
     -o spec "'$spec'" -o dense_matmul "$dense_matmul"
     -o D 32 -o H3 2 -o d_r 32 -o num_layers 8
     -o anchor_chunk 4096 -o jet_group_size 16 -o ckpt True
     -o use_router False -o message_backend "'auto'"
     "$@")

if [[ "${AIO_DRY_RUN:-0}" == 1 ]]; then
    printf 'EPOCHS=%q COMMENT=%q ' "$EPOCHS" "$COMMENT"
    printf '%q ' "${cmd[@]}"
    printf '\n'
    exit 0
fi
cd -- "$repo_root"
exec "${cmd[@]}"
