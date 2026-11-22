# Frozen-embedding head test

**Question:** does a trained tagger fail to extract information that standard substructure observables carry? The test freezes the tagger, exports its class-token embedding, and trains a small head twice on jets the tagger never saw: once on the embedding alone and once on the embedding plus hand-computed observables. If the second head is no better, the embedding already contains what those observables say, and an architecture change that re-expresses the same information cannot help.

No tagger is retrained. The noise is the spread between head seeds, on identical test jets for every arm, which is far smaller than the 4–9% seed spread of full training runs.

## Run it

Use the Weaver environment of the JetClass-II runs, with `DATA_PATH` and `OUTPUT_PATH` set as for `run_jetclass2.sh`.

```bash
export DATA_PATH=/path/to/datasets
export OUTPUT_PATH=/path/to/runs
export CHECKPOINT=/path/to/baseline/training/net_best_epoch_state.pt

# 1. A short run on a handful of files, to check the setup. Its numbers mean nothing.
QUICK=1 ./scripts/run_embedding_head_test.sh

# 2. The full test.
./scripts/run_embedding_head_test.sh
```

The defaults assume the default ParT (width 128, 8 heads, 8 blocks, as trained by `integrations/weaver/ParT.py` on `main`) trained on 10% of the files. For another model:

```bash
# A width-48 preset from this branch
FULL_SIZE=0 PRESET=linear_pair_rotary_values15_d4 NAME=head_test_rotary ./scripts/run_embedding_head_test.sh

# Anything that was passed to Weaver with -o
EXTRACT_ARGS='--model-option num_layers=2' FULL_SIZE=0 PRESET=part_pair_mlp64_d4 ./scripts/run_embedding_head_test.sh
```

The checkpoint is loaded strictly, so a wrong size stops with the mismatched keys.

## What the three steps do

| Step | Script | Needs | Output |
|---|---|---|---|
| `extract` | `scripts/extract_embeddings.py` | GPU, Weaver | `shards/{train,val,test}/<file>.npz`: label, embedding, original logits, `jet_pt`, `jet_sdmass`, and the valid constituent four-momenta exactly as the model received them |
| `observables` | `scripts/compute_jet_observables.py` | CPU cores, NumPy | `<file>.obs.npz` beside each shard, and `observables.json` with the column names |
| `heads` | `scripts/train_embedding_head.py` | GPU | `heads/<arm>/seed<k>/metrics.json` from the repository's evaluator, and `heads/summary.json` |

Every step skips work that is already done, so an interrupted run resumes when the script is started again. Run a subset with `STEPS="observables heads"`. Everything is written under `$OUTPUT_PATH/$NAME` (default `embedding_head_test`).

**Files.** The head must be trained on jets the checkpoint never saw. A 10% run used `Res2P` 0–19, `Res34P` 0–85 and `QCD` 0–27, so the head trains on the same number of files from the middle of the training range (`Res2P` 100–119, `Res34P` 430–515, `QCD` 140–167), validates on the first validation files, and is scored on the usual test files. A random fraction of each file is kept: 20% / 15% / 12%, roughly 2M / 0.4M / 3M jets. Change the ranges with `HEAD_TRAIN_RES2P=START:END` and so on, and the fractions with `TRAIN_KEEP`, `VAL_KEEP`, `TEST_KEEP`. For a checkpoint trained on more than half the files, set `CHECKPOINT_TRAIN_PERCENTAGE`; the script then refuses the default ranges until you give ones the checkpoint did not use and set `ALLOW_OVERLAP=1`. A checkpoint trained on all training files leaves only the validation files to split between head training and head validation.

**Reweighting.** Head-training jets are drawn with Weaver's flat pT × soft-drop-mass reweighting and class weights, as in tagger training. Without it the head could gain from mass and pT priors that the tagger was deliberately kept from learning, and that gain would be mistaken for missing substructure. The weights are taken from `REWEIGHT_YAML` if set, else from the `JetClassII_kin.<md5>.auto.yaml` that Weaver wrote beside the data config when the tagger trained, else rebuilt from the head-training shards with Weaver's method. The log line `flat reweighting: ...` says which.

**Heads.** A two-layer MLP (1024, 1024, GELU, dropout 0.1), AdamW, 40 epochs with warm-up and cosine decay, three seeds. Each embedding arm is residual: its logits are the checkpoint's own logits plus a correction whose last layer starts at zero. Every arm therefore starts exactly at the checkpoint's performance, and the epoch with the best weighted validation loss is kept, so an arm that learns nothing reports exactly the checkpoint's numbers. Change settings with `HEAD_ARGS`, for example `HEAD_ARGS="--arms embedding embedding+all --seeds 0 1 2 3 4"`.

## Arms

| Arm | Inputs | Role |
|---|---|---|
| `parT_head` | none (the checkpoint's own logits) | Check. Must reproduce the checkpoint's known test metrics. |
| `embedding` | embedding | **Reference.** What a larger head and fresh jets buy without new information. |
| `embedding+<group>` | embedding and one observable group | Which kind of observable helps, if any. |
| `embedding+all` | embedding and all 116 observables | **The answer**, read against `embedding`. |
| `obs_only` | observables | Check that the observables carry signal. Trained from scratch. |

## Read the result

The script prints a table and saves it in `heads/summary.json`. The `embedding` row shows values. Every other row shows the relative change against it, averaged over seeds, with the spread across seeds after `±`. Differences are paired by seed.

Columns: hadronic `rej50` geometric mean, the same for the 4-, 3- and 2-prong groups, `X_bb_vs_QCD_bb`, `X_bb_vs_X_cc`, top-1 accuracy, and `val_loss`, the weighted validation cross-entropy (lower is better). `summary.json` also has each hadronic group separately, `rej30`, the 1e-3 signal efficiency, and the remaining pairs.

Before reading the answer, check three things:

1. **`parT_head` matches the checkpoint.** Its `rej50_geomean_hadronic` (in `heads/parT_head/metrics.json`) should be within a few percent of the checkpoint's own test result; the test jets are a random subset of the same files. `extract` also prints the original head's top-1 accuracy per file. A large gap means the model options do not match the checkpoint.
2. **`obs_only` is far above chance.** If not, the observables are broken.
3. **`embedding` is close to `parT_head`.** A large gain here means the head itself was the limit, which is a finding of its own.

Then compare `embedding+all` with `embedding`:

- **No change beyond the seed spread, in rejection and in `val_loss`.** The embedding already holds what these observables carry. Subjet structure, N-subjettiness, energy correlators, pull and the primary Lund plane are not what ParT is missing.
- **A clear gain, several times the seed spread.** ParT's output lacks information that is present in its inputs. The per-group arms and the per-class columns say which, and that is a specific target for an architecture change.
- **A small gain, one or two times the spread.** Add seeds and head-training jets (`TRAIN_KEEP`) before concluding anything.

`val_loss` is the most sensitive column, because it uses every validation jet instead of the tail that sets a rejection.

## Limits of the test

- **A null result covers these observables only.** It does not show that ParT extracts everything.
- **A gain does not say where the information was lost.** It may be absent from the particle tokens or dropped by the class-attention readout. The next step would be the same test with pooled particle tokens as a third input.
- **The head has finite data.** With 116 extra inputs it needs enough jets to find a small gain. The script prints the effective sample size after reweighting; `obs_only` and `val_loss` show whether the head is learning.
- **Approximations in the observables.** The four-point correlator uses the 24 hardest constituents, and the generalised correlators behind `N2` and `N3` use the 32 and 16 hardest. They are still functions of the model's inputs.

## The observables

All 116 are computed from the scaled constituent four-momenta in `pf_vectors` (at most 128 particles, jet pT scaled to 500), which are exactly what the model receives. They add no information the tagger was not given, and none reveals the absolute jet pT. Angles use rapidity and azimuth; the radius is 0.8.

| Group | Count | Contents |
|---|---:|---|
| `basic` | 9 | Multiplicity, mass, width, LHA, thrust, pTD, three leading momentum fractions |
| `subjets` | 50 | Exclusive kT subjets for 2, 3 and 4 subjets: mass, momentum fraction and multiplicity of each, pairwise ΔR and mass, and the merge scales d12, d23, d34 |
| `nsub` | 7 | τ1–τ4 (β = 1, exclusive-kT axes) and τ21, τ32, τ43 |
| `ecf` | 16 | e2, e3, e4, C2, D2, C3, D3 for β = 1 and 2; N2 and N3 for β = 1 |
| `pull` | 5 | Jet pull, and the pull magnitude and pull angle of the two leading kT subjets |
| `lund` | 25 | (ΔR, kT, z) of the first six primary Cambridge/Aachen declusterings and of the hardest one, counts above three kT thresholds, number of primary emissions |
| `softdrop` | 4 | Soft-drop (z > 0.1, β = 0) mass, zg, rg, and number of dropped branches |

The code is `src/aio/analysis/jet_observables.py`, batched NumPy without FastJet. `tests/test_jet_observables.py` checks it against slow per-jet reference implementations and for invariance under permutation, padding, azimuthal rotation and longitudinal boost.

## Cost

Sizes scale with the keep fractions. These are estimates from the array sizes, not measurements: at the defaults, about 14 GB of shards on disk and about 16 GB of memory in the `heads` step. The `observables` step ran at about 600 jets per second per core on synthetic jets in development, which would be about 2.5 core-hours for 5.4M jets; set `WORKERS`. Lower `TEST_KEEP` first if memory is short.

## Tests

```bash
PYTHONPATH=src:. python -m pytest tests/test_jet_observables.py tests/test_head_probe.py
```

These cover the observables, the reweighting, the feature scaling, and a full run on synthetic jets through the observable script, the head script with `--backend sklearn`, and the repository's evaluator. They need NumPy, PyYAML and scikit-learn, not PyTorch. The scikit-learn backend exists for this check only.
