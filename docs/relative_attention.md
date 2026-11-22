# Relative attention for JetClass-II

The primary comparison is **C vs H**. Both give each head a learned linear map of the four ParT pair features per particle block. H also rotates value planes into a common frame before aggregation and rotates the result back into the receiving particle's frame. E uses rotary queries and keys as well, eliminating explicit pair features. All three use full softmax attention over valid constituents.

## Run the pilot

Use the same Weaver environment as your existing JetClass-II runs. The shared launcher uses the repo's Weaver fork options, including `--seed` and `--wandb-dir`. Set `DATA_PATH` to the parent of `JetClass2/Pythia/data` and `OUTPUT_PATH` to your run directory. From the checkout:

```bash
export DATA_PATH=/path/to/datasets
export OUTPUT_PATH=/path/to/runs

python scripts/run_relative_experiment.py C
python scripts/run_relative_experiment.py H
```

Each command trains and tests one four-block, width-48 model on the seven kinematic features. Defaults are 10% of the training files, 80 epochs, seed 123, and the existing validation and test splits. The usual `DDP_NGPUS`, `BATCH_SIZE`, `NUM_WORKERS`, `EPOCHS`, `TRAIN_PERCENTAGE`, and `SEED` settings apply. Explicit CLI options take precedence. Outputs go to `$OUTPUT_PATH/RelativeParT/JetClass2_kin/relative_<preset>_seed<seed>/`. Use `COMMENT` to name a single run or `--comment-prefix` for a series. Give overrides a fresh run name.

```bash
# Preview the four primary commands without data or a GPU
python scripts/run_relative_experiment.py suite --dry-run

# Run C, H, B, E sequentially from scratch
python scripts/run_relative_experiment.py suite

# Compare two seeds for one arm, or change depth or data budget
python scripts/run_relative_experiment.py E --seed 123 --seed 124
COMMENT=relative_H_d2_seed124 python scripts/run_relative_experiment.py H \
    --seed 124 --num-epochs 40 --train-percentage 10 -o num_layers 2

# Test the default best checkpoint, or an explicit checkpoint with matching architecture
python scripts/run_relative_experiment.py H --test-only
python scripts/run_relative_experiment.py E_small --weights /path/to/net_best_epoch_state.pt
```

| Preset | Scores | Values | Parameters at depth 4 |
|---|---|---|---:|
| A | Original ParT pair MLP, width 64 | Ordinary | 323,364 |
| B | Content only | Ordinary | 314,064 |
| C | Linear pair bias | Ordinary | 314,144 |
| H | Linear pair bias | 15 planes | 361,424 |
| H_small | Linear pair bias | 6 planes | 333,056 |
| E | Rotary, 30 planes | 15 planes | 456,880 |
| E_small | Rotary, 12 planes | 6 planes | 371,200 |
| D | Rotary, 30 planes | Ordinary | 409,600 |
| C8 | Original ParT pair MLP, width 8 | Ordinary | 314,348 |
| F | Jet-rest-frame harmonics through degree 2 | Ordinary | 314,096 |
| G | Learned low-rank coordinate bias | Ordinary | 323,024 |

A instantiates the original `part_d4`. An existing checkpoint or metrics file can serve as A when preprocessing, depth, seed, and training recipe match. C8, F and G are follow-up controls. F weights the existing dipole and STF quadrupole inner products separately per head. The constant harmonic cancels in the softmax. G maps coordinates through a width-32 GELU MLP to independent rank-8 query and key features per head (`-o control_rank N`).

## Coordinates and switches

Weaver supplies raw jet-axis offsets in `pf_points` and four-vectors scaled by `500 / jet_pt`. The rotary coordinates are Δη, wrapped Δφ in (−π, π], and `log(hypot(px_scaled, py_scaled) / 500)`. Features, four-vectors and offsets share one trimming permutation. The linear bias uses ParT's native order of log kT, log z, log Δ and log m², including its rapidity convention, self pairs and clamps. It computes those features once per batch and projects them independently in each block.

Each head retains 12 ordinary content dimensions. The full score bank starts at eight angular octaves in three directions and six momentum octaves. The full value bank selects alternate octaves. The small score bank slides three adjacent octaves of each family across the heads. For its six values we use all three directions of the middle angular octave and all three selected momentum octaves. Each block has its own frequency parameters. Phase evaluation stays in at least fp32 under the standard bf16 training recipe.

| Weaver option | Effect |
|---|---|
| `-o num_layers 2` | Two particle blocks with the same two class blocks |
| `-o rotate_values False` | Ordinary value channels |
| `-o scale_invariant_pairs True` | Linear score uses only log z and log Δ |
| `-o angular False` or `-o momentum False` | Omit that family and keep its coordinates out of the learned phases |
| `-o freeze_frequencies True` | Freeze the initialized ladders |
| `-o directions 4` | Four angular directions per octave |
| `-o class_at_origin True` | Apply relative attention in the two class blocks |
| `-o bank "'small'"` | Small score and value banks for rotary attention |

By default class attention, FFNs, input embedding and the classifier follow the baseline. In the optional relative class readout the class token sits at coordinate zero and attends to itself and the particles. For linear class scores, which also need a four-vector, it uses a massless reference along the jet axis with transverse momentum equal to the jet's. Rotary scores and values use PyTorch SDPA. Kernel selection and realized memory depend on hardware, dtype, mask and head dimensions. Its softmax computation remains quadratic.

## Check before training

```bash
python scripts/verify_relative_attention.py
python scripts/export_representation_inputs.py \
    --files "$DATA_PATH"/JetClass2/Pythia/data/{Res2P_0200,Res34P_0860,QCD_0280}.parquet \
    --output /tmp/kin-validation.pt
python scripts/fit_relative_pair_bias.py --checkpoint /path/to/part_d4_best.pt \
    --inputs /tmp/kin-validation.pt
```

The first command draws independent synthetic fit and test pairs and prints held-out kernel R². The second and third use a trained baseline and held-out, Weaver-preprocessed jets. The fit samples ordered valid pairs (including self pairs), splits by jet, fits a linear map and a sine/cosine basis for each head, and reports held-out R² and RMSE. The rotary fit uses fixed frequencies and content-independent coefficients. The trained model can also condition amplitudes and phases on content.

## Evaluate and measure cost

Use the existing JetClass-II evaluator on all ROOT prediction shards from one run. It writes a consolidated `metrics.json` beside them, including per-group rejection, pair benchmarks, and working-point details.

```bash
runs="$OUTPUT_PATH/RelativeParT/JetClass2_kin"
python scripts/evaluate_relative_attention.py --predictions "$runs"/relative_H_seed123/results/pred*.root

# Reuse saved metrics and compute (log H - log B) / (log A - log B)
python scripts/evaluate_relative_attention.py --metrics "$runs/relative_H_seed123/results/metrics.json" \
    --reference /path/to/part_d4/results/metrics.json \
    --plain "$runs/relative_B_seed123/results/metrics.json"

# Time the same fixed padded batch and occupancy distribution for every arm
PYTHONPATH=src:. python scripts/benchmark_representation_order.py --relative --device cuda \
    --inputs /tmp/kin-validation.pt --batch-size 32 --amp --backward --presets A C H E
```

The benchmark reports mean valid particles, milliseconds per batch, throughput, parameter count, and CUDA peak allocated memory. Omit `--backward` to time inference. Without `--inputs`, it uses an all-valid synthetic batch. Both timing modes disable random trimming to hold the input shape fixed. Use multiple batches representative of production when comparing costs.

```bash
PYTHONPATH=src:. python -m pytest -q tests/test_relative_attention.py
```
