# Region structure and Lund hierarchy experiments

The immediate question is whether rank-2 communication helps when a coarse region
contains unresolved angular structure. A second question is whether a C/A tree
provides better groups than the learned soft assignments. This branch supplies
both tests while preserving all existing presets/checkpoints.

## Start here

1. Diagnose existing `regions8_rank2`, `regions8_wide`, `regions8_sv`, and
   `full_sv_pair` checkpoints on the **same held-out input file**.
2. Train **three new hierarchy controls**: `regions8_ca_sv`,
   `regions8_ca_rank2`, `regions8_ca_wide`, initially seed 123. Compare to the
   existing K=8 learned runs. Repeat promising comparisons before interpreting them.
3. Confirm the small-model reference and K=8 order comparison across seeds.
   Use shared frozen geometry to separate representation from grouping/frame changes.
4. Use the optional local-pair controls when assessing compression on the stronger
   backbone. The full dense-pair reference remains the performance benchmark.

Do not launch every supplied preset as one sweep. K=16, frozen geometry, and
local-pair variants are separate follow-ups selected by the diagnostics.

## Environment and compatibility

Use the environment/data of the completed runs; install the optional dependencies:

```bash
pip install -r requirements-representation.txt -r requirements-lund.txt
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

Install your appropriate CUDA PyTorch build first. The examples assume `DATA_PATH`
and `OUTPUT_PATH` already point at your dataset and run directories. All commands
run from the repo root. Existing launcher defaults remain 80 epochs, 10% training
files and seed 123. Match your actual completed runs' epoch, optimizer, batch and
data settings; CLI arguments still override defaults. `COMMENT` is explicitly
set by the user for each run and is passed through unchanged.

No existing preset is changed. New model options add no state keys to old models.
Checkpoint loading is strict, including Weaver/DDP `mod.`/`module.` prefixes.

## 1. Export a diagnostic sample

Use validation files, not test data for selecting configurations. These example
file indices are in the repository launcher's validation split:

```bash
python scripts/export_representation_inputs.py \
  --files "$DATA_PATH/JetClass2/Pythia/data/Res2P_0200.parquet" \
          "$DATA_PATH/JetClass2/Pythia/data/Res34P_0860.parquet" \
          "$DATA_PATH/JetClass2/Pythia/data/QCD_0280.parquet" \
  --jets-per-file 2048 --output diagnostics/inspection.pt
```

The exporter saves the original jet observers (including `jet_pt` and
`jet_sdmass`), source-file indices, and ordinals after YAML selection. This
preserves physical kinematics even though the model's four-vectors are rescaled.
Supply the resolved training YAML with `--data-config` if it differs from kin.

For focused signal inspection, collect a per-label sample:

```bash
python scripts/export_representation_inputs.py \
  --files "$DATA_PATH/JetClass2/Pythia/data/Res34P_0860.parquet" \
          "$DATA_PATH/JetClass2/Pythia/data/Res34P_0861.parquet" \
  --label-ids 15 18 32 48 63 19 35 50 64 77 29 45 60 74 87 \
  --jets-per-label 128 --jets-per-file 4096 --max-scanned-per-file 100000 \
  --output diagnostics/focused.pt
```

The selected labels sample four-quark, mixed quark/gluon four-prong, all-gluon
four-prong, and three-prong controls. This is a diagnostic subset, not complete
coverage of every broad group. The exporter reports achieved label counts and
unfilled quotas; supply more held-out files if necessary. It reads first qualifying
events, not a random or reweighted performance sample. **Do not estimate rejection
at 1e-3/1e-4 from these small files.** Use the existing full validation evaluator
for working-point comparisons.

## 2. Inspect the trained mechanisms

Set these paths to your actual checkpoints. These variables are used unchanged:

```bash
export R8_RANK2_CHECKPOINT=/path/to/regions8_rank2/training/net_best_epoch_state.pt
export R8_WIDE_CHECKPOINT=/path/to/regions8_wide/training/net_best_epoch_state.pt
export R8_SV_CHECKPOINT=/path/to/regions8_sv/training/net_best_epoch_state.pt
export FULL_PAIR_CHECKPOINT=/path/to/full_sv_pair/training/net_best_epoch_state.pt

python scripts/diagnose_representation_order.py \
  --preset regions8_rank2 --checkpoint "$R8_RANK2_CHECKPOINT" \
  --inputs diagnostics/inspection.pt --max-jets 8192 --batch-size 32 --device cuda \
  --tensor-ablations --communication-ablations --output diagnostics/r8_rank2.json

python scripts/diagnose_representation_order.py \
  --preset regions8_wide --checkpoint "$R8_WIDE_CHECKPOINT" \
  --inputs diagnostics/inspection.pt --max-jets 8192 --batch-size 32 --device cuda \
  --communication-ablations --output diagnostics/r8_wide.json

python scripts/diagnose_representation_order.py \
  --preset regions8_sv --checkpoint "$R8_SV_CHECKPOINT" \
  --inputs diagnostics/inspection.pt --max-jets 8192 --batch-size 32 --device cuda \
  --communication-ablations --output diagnostics/r8_sv.json

python scripts/diagnose_representation_order.py \
  --preset full_sv_pair --checkpoint "$FULL_PAIR_CHECKPOINT" \
  --inputs diagnostics/inspection.pt --max-jets 8192 --batch-size 32 --device cuda \
  --communication-ablations --output diagnostics/full_pair.json

python scripts/plot_region_diagnostics.py \
  --reports diagnostics/r8_rank2.json diagnostics/r8_wide.json \
            diagnostics/r8_sv.json diagnostics/full_pair.json \
  --output-dir diagnostics/plots
```

Repeat using `focused.pt` and distinct output names for the label-focused sample.
Increase `--max-jets` if you export more than 8192 events. Plot filters such as
`--pt-range 400 800 --mass-range 50 150 --multiplicity-range 32 65` are optional;
the first two use physical observers, the third counts valid model inputs after
truncation. Keep enough events in each group and inspect the reported counts.

Outputs:

| Output | Interpretation |
| --- | --- |
| Summary JSON + detailed `.pt` | Group CE, per-event interventions, assignments, shape measures, original observers, input fingerprint |
| Assignment histograms | Hard occupancy, overlap of assignment columns, normalized entropy; soft occupancy alone cannot detect identical slots |
| Assignment maps | Lab-frame rapidity/azimuth and actual soft assignment heatmaps; region IDs are exchangeable between models |
| Utility-versus-structure plots + `utility_bins.csv` | Within-group descriptive averages of per-region tensor-removal effects |
| `group_summary.csv` | Group-level communication effects; verified same-input paired CE differences relative to the first report |
| C/A cut-node Lund plot | Internal splitting of each non-leaf region, optionally colored by its tensor-removal effect |

For assignments a_ik and unit spatial directions u_i, diagnostics form the centered
covariance C_k = sum_i a_ik (u_i - mean_k)(u_i - mean_k)^T / sum_i a_ik.
Angular spread is tr(C_k); anisotropy is (lambda_max - lambda_min)/tr(C_k).
Zero-spread and empty regions are excluded from shape/utility bins. These are
assignment-weighted **detector-frame diagnostics**, not Lorentz scalars or claims
that the learned tensor channel equals this covariance.

Intervention signs: `delta_ce = CE(intervened) - CE(original)`; positive means
the removed contribution helped that event. `global_off` removes the entire
global message, including output-projection bias, in every block. Local attention
and particle readout remain. Pair interventions remove only biases present in the
checkpoint. Tensor interventions remove a region's rank-2 Q/K/V in every block.
Effects are not additive, use labels, retain computation, and do not describe a
model retrained without the branch. Binned regions are correlated within jets;
these plots have no independent-region significance claims.

## 3. Three initial Lund hierarchy runs

```bash
COMMENT=r8-ca-sv-10pct-seed123 \
  bash scripts/run_representation_experiment.sh regions8_ca_sv --seed 123
COMMENT=r8-ca-rank2-10pct-seed123 \
  bash scripts/run_representation_experiment.sh regions8_ca_rank2 --seed 123
COMMENT=r8-ca-wide-10pct-seed123 \
  bash scripts/run_representation_experiment.sh regions8_ca_wide --seed 123
```

These first three have no pair bias, matching the existing learned-region runs.
The local graph, geometric transport, message pooling/unpooling, readout, and
scalar/vector/rank-2 definitions are shared. Learned assignment penalties are
disabled for C/A. C/A has 264 fewer parameters at K=8 (528 at K=16) because the
assignment head is absent; tensor versus wide remains exactly parameter matched
within each family. They are not asserted to be latency matched.

**Precise hierarchy:** FastJet Cambridge/Aachen with E-scheme four-vector merging
builds one binary tree from all supplied valid constituents, including soft ones.
A large reclustering radius (`ca_radius=1000`) suppresses beam removals; failure to
form one tree is an error, not a silent dropped constituent. The exclusive
min(K,N)-subjet cut gives disjoint regions that cover every constituent once.
Increasing K refines that tree cut. Each region is a subtree, not a truth prong.
N<K leaves unused slots masked. No particle rest frame is required.

The detector-frame rapidity/azimuth metric is invariant to common azimuthal
rotation and longitudinal boosts, not arbitrary Lorentz transformations. Learned
particle frames are retained, but the C/A grouping is fixed by the input momenta.
The pilot's local graph may cross these regions and the fine particle readout
remains: the coarse stream is a communication bottleneck, not a complete jet
compression codec.

This first control uses tree cuts for grouping. It does **not** implement the
LundNet GNN, feed all internal nodes as tokens, select only the primary branch, or
feed new Lund variables to the classifier. `lund_delta`, `lund_z`, `lund_kt`, and
`lund_mass` describe a cut node's internal last merge for diagnostics only; leaves
have `lund_split_valid=False`. Momentum magnitudes use the model's rescaled input
units. `lund_kt_fraction = kt / sum constituent pt` avoids a false GeV interpretation.

FastJet's batched Awkward interface runs on CPU inside forward, with transfers
included in the profiler. Per-node Lund diagnostics use the object interface
offline; they do not run during training. Use eager execution; compiled/ONNX
export is not supported for this C/A path. Profile before scaling production
training. Cached dataset-level clustering is a later optimization if this
grouping is useful.

Sources: [LundNet, section 2](https://arxiv.org/html/2012.08526v2),
[FastJet exclusive clustering](https://fastjet.readthedocs.io/en/latest/clustersequence.html),
[FastJet manual](https://arxiv.org/abs/1111.6097).

## 4. Repeat seeds and freeze geometry

Two additional seeds for the compact reference and small ParT establish the
reference uncertainty. Repeat the K=8 rank-2/wide comparison as the order test.
Examples for seed 456 (repeat with 789 and explicit distinct comments):

```bash
COMMENT=full-pair-10pct-seed456 bash scripts/run_representation_experiment.sh full_sv_pair --seed 456
COMMENT=part-small-10pct-seed456 bash scripts/run_representation_experiment.sh part_small --seed 456
COMMENT=r8-rank2-10pct-seed456 bash scripts/run_representation_experiment.sh regions8_rank2 --seed 456
COMMENT=r8-wide-10pct-seed456 bash scripts/run_representation_experiment.sh regions8_wide --seed 456
```

For a controlled order comparison, export geometry from the existing K=8 SV run:

```bash
python scripts/export_region_geometry.py --preset regions8_sv \
  --checkpoint "$R8_SV_CHECKPOINT" --output diagnostics/r8_sv_geometry.pt
export GEOMETRY_CHECKPOINT="$PWD/diagnostics/r8_sv_geometry.pt"
COMMENT=r8-sv-frozen-10pct-seed456 bash scripts/run_representation_experiment.sh regions8_sv --seed 456
COMMENT=r8-rank2-frozen-10pct-seed456 bash scripts/run_representation_experiment.sh regions8_rank2 --seed 456
COMMENT=r8-wide-frozen-10pct-seed456 bash scripts/run_representation_experiment.sh regions8_wide --seed 456
unset GEOMETRY_CHECKPOINT
```

All three train new message/readout weights with the same frozen assignments and
frames. Compare this triplet internally, not one frozen model to an unfrozen old
result. C/A geometry can likewise be exported from `regions8_ca_sv` and shared
within the C/A family to fix learned frames. Learned and C/A geometry checkpoints
are intentionally incompatible. No adaptive order policy is trained here.

## 5. Optional stronger local-pair controls

The trained full-pair checkpoint interventions indicate whether local or global
bias carries the gain. Retraining confirms it, because ablations alter a trained
network's input distribution:

```bash
COMMENT=full-pair-local-10pct-seed123 bash scripts/run_representation_experiment.sh full_sv_pair_local
COMMENT=full-pair-global-10pct-seed123 bash scripts/run_representation_experiment.sh full_sv_pair_global

COMMENT=r8-sv-localpair-10pct-seed123 bash scripts/run_representation_experiment.sh regions8_sv_pair_local
COMMENT=r8-rank2-localpair-10pct-seed123 bash scripts/run_representation_experiment.sh regions8_rank2_pair_local
COMMENT=r8-wide-localpair-10pct-seed123 bash scripts/run_representation_experiment.sh regions8_wide_pair_local
```

The regional triplet has the same local pair treatment as `full_sv_pair_local`.
Compare these four to isolate global coarsening with local pair physics retained;
keep `full_sv_pair` as the stronger dense-pair benchmark. None of these regional
presets defines a new bias between coarse tokens. The local variant computes
features only for existing local edges, O(NL); the existing neighborhood search
still has an O(N^2) distance step. Full/global pair modes retain O(N^2) features.

If the C/A control earns further study, corresponding
`regions8_ca_{sv,rank2,wide}_pair_local` presets are supplied. K=16 counterparts
exist for all three families to retain granularity as a separate axis.

## Measure cost on your hardware

```bash
python scripts/benchmark_representation_order.py --device cuda --batch-size 32 \
  --particles 128 --steps 100 --warmup 10 \
  --presets part_small full_sv_pair full_sv_pair_local \
            regions8_rank2 regions8_wide regions8_ca_rank2 regions8_ca_wide \
  --output diagnostics/profile-b32-n128.json
```

Repeat at realistic constituent counts and batch sizes (including batch 1 if
latency matters), with the same precision. `--backward` includes backward and
geometry regularization, but not optimizer/data loading. C/A timings include
CPU clustering and transfers. These synthetic inputs measure execution, not
task performance. Report measured costs alongside seed-aggregated validation
metrics; do not convert parameter matching into a FLOP/latency claim.

## Interpretation gates

- Distinct regions + reproducible tensor advantage tied to internal structure:
  investigate region-level adaptive order, including actual skipped computation.
- C/A improves SV and rank-2 equally: evidence for hierarchy, not an order effect.
- Rank-2 improves over wide only for one grouping or K: study that interaction
  with shared frames/assignments and repeated seeds.
- Global-message removal has little effect: the present bottleneck may not force
  use of coarse summaries; inspect that before increasing representation order.
- No reproducible tensor advantage: reconsider the summary operator. The current
  rank-2 channels are learned features, not the explicit value-key moment correction.

## Implementation validation

- 40 CPU numerical/integration tests passed across the original pilot, baseline
  controls, and follow-ups. New checks cover independent C/A merging, nested cuts,
  batched/object-interface agreement, padding holes and N<K, longitudinal boosts,
  sparse/dense pair equivalence, global-gate semantics, covariance diagnostics,
  label-capped Weaver exports, and the complete checkpoint-to-plot workflow.
- Seven original presets retain bit-exact initialization and predictions under a
  fixed seed, and load their original checkpoint keys strictly.
- New hierarchy/local-pair variants completed finite-gradient checks and bf16
  forward/backward execution. Two small synthetic models completed 20 optimizer
  steps with decreasing loss; these are execution checks, not physics results.
- All 31 launcher presets passed dry runs with explicit comments and training
  overrides. Shell syntax, Python compilation, and diff checks passed.
- CUDA/DDP and JetClass-II training were not run in the implementation environment.
  FastJet 3.5.2.0 and CPU PyTorch 2.14.1 were used for these checks.
