# Baseline controls and checkpoint diagnostics

These additions can run independently of the original region/order jobs. They do
not change the original preset defaults or checkpoint layout. The launcher keeps
the user's configurable training options and explicit `COMMENT` naming.

## First additional trainings

| Preset | What changes | Parameters (7 inputs, 188 classes) |
|---|---|---:|
| `full_sv` | Existing pilot reference | 337,279 |
| `part_small` | Original ParT, width 48, four encoder blocks, four heads | 323,364 |
| `full_sv_pair` | Pilot + receiver-local pair-feature attention bias | 338,627 |
| `full_sv_cls` | Pilot + two ParT class-attention readout blocks | 438,727 |
| `full_sv_pair_cls` | Optional combined pair and readout variant | 440,075 |

Prioritize `part_small` and `full_sv_pair`. Use `full_sv_cls` to test readout
separately; the combined variant measures their interaction if both are useful.
The class readout adds about 30% more parameters; it is not a capacity-matched
comparison. Small ParT is within about 4% of the pilot parameter count, but is not
FLOP- or latency-matched. These runs diagnose alternatives, not a complete causal
decomposition of the performance gap.

Small ParT keeps the original four pair features, three 64-wide pair-embedding
layers, two class blocks, FFN expansion factor four, normalization, dropout and
classification-head conventions. Its particle embedding is `[48,192,48]` and its
encoder has four blocks. It does not use learned regions or canonical frames.

From the repository root, with the same environment/data as the existing jobs:

```bash
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
# DATA_PATH and OUTPUT_PATH should already identify your data and run directories.
COMMENT=part_small-10pct-seed123 bash scripts/run_representation_experiment.sh part_small
COMMENT=full_sv_pair-10pct-seed123 bash scripts/run_representation_experiment.sh full_sv_pair
COMMENT=full_sv_cls-10pct-seed123 bash scripts/run_representation_experiment.sh full_sv_cls
# Optional fourth run:
COMMENT=full_sv_pair_cls-10pct-seed123 bash scripts/run_representation_experiment.sh full_sv_pair_cls
```

Defaults remain 80 epochs, 10% training files, seed 123, and kin inputs. Match the
resolved settings of your completed runs. CLI `--num-epochs`, `--seed`,
`--train-percentage`, other Weaver options, and `AIO_DRY_RUN=1` remain supported.
The `part_small` outputs are under `ParT/JetClass2_kin/COMMENT`; the others are
under `RegionLLoCa/JetClass2_kin/COMMENT`.

## Pair-bias definition and limits

For receiver i and sender j, transform BOTH physical four-momenta into receiver
i's predicted local frame. Construct `(log kT, log z, log deltaR, log m_ij^2)`
using the definitions in the repository's ParT implementation. A shared
`4 -> 32 -> 32 -> heads` GELU MLP maps the features to attention-logit biases.
The same bias is reused across blocks and on the corresponding fixed-neighbor
edges. It is generally asymmetric because different receivers use different frames.

This is a **ParT-inspired local-frame pair-bias ablation**, not the official
LLoCa-ParT implementation: it uses an MLP rather than ParT's BatchNorm pair
embedding. Gradients flow through the learned frames. Numerical floors precede
roots and undefined azimuths; invalid receiving frames disable the corresponding
bias row, and padding is masked. These choices are tested on self/beam/zero pairs.

The original branch materializes all particle pairs, so its low parameter count
does not imply low runtime or memory cost. Dense/global pair bias is permitted
only for full-particle references. The [follow-up controls](region_followups.md)
add `pair_bias_scope="local"`, constructing only local-edge features; this can be
used with either full or regional global communication. A bias between coarse
tokens still needs a separately specified coarse four-momentum/frame convention.

The equivariance claim is conditional on covariantly transformed physical
references and scalar auxiliary features, as for the original pilot. The supplied
JetClass-II preprocessing is detector-dependent. This change does not upgrade
that preprocessing to arbitrary-boost invariance.

## Diagnose existing checkpoints without retraining

The exporter uses Weaver's actual YAML selection, transformations and padding in
evaluation mode. Select held-out validation files; these example indices are in
the repository launcher's validation split:

```bash
python scripts/export_representation_inputs.py \
  --files "$DATA_PATH/JetClass2/Pythia/data/Res2P_0200.parquet" \
          "$DATA_PATH/JetClass2/Pythia/data/Res34P_0860.parquet" \
          "$DATA_PATH/JetClass2/Pythia/data/QCD_0280.parquet" \
  --jets-per-file 128 --output diagnostics/validation_inputs.pt

python scripts/diagnose_representation_order.py \
  --preset regions16_sv \
  --checkpoint /path/to/regions16_sv/training/net_best_epoch_state.pt \
  --inputs diagnostics/validation_inputs.pt --max-jets 384 \
  --device cuda --output diagnostics/regions16_sv.json

python scripts/diagnose_representation_order.py \
  --preset regions16_rank2 \
  --checkpoint /path/to/regions16_rank2/training/net_best_epoch_state.pt \
  --inputs diagnostics/validation_inputs.pt --max-jets 384 \
  --device cuda --tensor-ablations --output diagnostics/regions16_rank2.json
```

Use the same exported inputs for all models. These are the first selected events
per file, without shuffling or training reweighting: useful for inspecting the
mechanism, not representative performance estimates or seed uncertainty. Supply
`--data-config` with the matching resolved preprocessing YAML for custom runs.
The default diagnostic limit is 256 jets; increase `--max-jets` explicitly as above
to inspect every exported event. Input particle ordering is `(px,py,pz,E)`.

Each run saves a summary JSON and a sibling `.pt` containing per-jet assignments,
labels, file provenance, logits and diagnostics:

- **Assignment entropy and maximum probability:** diffuse vs confident membership.
- **Soft effective occupancy and hard occupied regions:** used slots. Soft
  occupancy alone cannot establish distinct regions: uniform assignments use all
  slots but are maximally overlapping.
- **Assignment-column cosine overlap:** zero for disjoint hard regions, one for
  identical nonempty assignment columns. Region IDs are exchangeable across runs.
- **Per-region angular dispersion:** `1 - |mean unit spatial direction|^2`,
  weighted by memberships in the supplied lab frame. This is not an RMS angle or
  a Lorentz invariant; exclude zero-angular-mass regions when summarizing it.
- **Invalid frame fraction:** computed over real particles only.

With `--tensor-ablations`, the script also computes the original per-jet CE,
all-tensors-off delta CE, and each region-off delta CE. A positive delta means
removing that region's tensor contribution increased the loss. It compares these
finite interventions with the first-order estimate `-dCE/dgate` at gate=1.

The intervention removes that region's rank-2 Q, K and V in every global block,
holding the learned grouping and all weights fixed. Effects are not additive.
The off model is **not** a separately trained SV model: it retains the original
attention scale and weights. These label-dependent diagnostics are not a router,
do not skip computation, and do not establish an adaptive speedup. The script
requires approximately K+2 forwards per batch when ablations are enabled.

## Cost measurements

```bash
python scripts/benchmark_representation_order.py \
  --presets full_sv part_small full_sv_pair full_sv_cls full_sv_pair_cls \
  --device cuda --batch-size 32 --particles 128 --amp \
  --warmup 10 --steps 100 --output diagnostics/controls_gpu.json
```

Repeat with batch size 1 if latency is relevant, and with representative lengths
(e.g. 48 and 128). `--backward` includes forward, CE, regularization and backward
but excludes optimizer work. Inference mode times the full model forward only.
The benchmark includes geometry, neighborhood construction, pair features,
pooling/transport and readout. For fixed-shape comparability it disables ParT's
random training trimmer during profiling only; the training preset retains it.
These eager-mode synthetic measurements are not deployment latency or a complete
FLOP accounting. Compare both families under equivalent compilation/precision
settings before making efficiency claims.

## Validation

```bash
python -m pytest tests/test_representation_order.py tests/test_representation_controls.py -q
python scripts/smoke_representation_order.py --preset part_small --steps 20
python scripts/smoke_representation_order.py --preset full_sv_pair_cls --steps 20
```

Numerical tests cover conditional Lorentz covariance, permutation and padding,
finite frame/pair gradients, the explicit attention-bias operator, preserved ParT
components, diagnostic interpretation and finite-difference tensor utility.
Synthetic optimization and timing checks validate execution only. No JetClass-II
training or GPU speed claim is made from local CPU validation.

Implementation validation completed on CPU with PyTorch 2.14.1+cpu and
weaver-core 0.5.3:

- All 28 numerical/integration tests passed, including a Parquet -> Weaver
  preprocessing -> exported tensors -> checkpoint diagnostic round trip.
- Original full-SV and K=8 presets matched the previous branch's initialized
  state dictionaries and predictions exactly under the same random seed.
- Both small ParT and the combined pair/readout model completed 20 synthetic
  optimization steps with finite gradients. All five reference configurations
  completed inference profiling; small ParT and the combined model also completed
  bf16 forward/backward profiling.
- Launcher dry runs, shell syntax, Python compilation and diff checks passed.

GPU, multi-GPU, and actual JetClass-II performance remain to be tested on the
training machine. CPU timings are execution checks, not evidence of a speedup.
