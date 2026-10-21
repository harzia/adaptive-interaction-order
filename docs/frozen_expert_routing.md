# Frozen-expert routing and the larger granularity grid

Different winners across classes or region counts motivate testing adaptivity;
they do not show that a deployable gate can recognize which expert will help.
This experiment asks whether **input-predictable complementarity** survives on
unseen jets and across independently trained checkpoint seeds.

This is the first feasibility stage for adaptive representation order: a small
**per-jet** gate over frozen SV, rank-2 and wide checkpoints. It does not yet train
a shared backbone or choose a representation separately for each region. The
current rank-2 channels are learned general tensors, not explicitly constructed
physical moments or a pure angular multipole.

## What is implemented

| Method | What is fitted | Experts evaluated per jet |
|---|---|---|
| Each individual checkpoint | Nothing | One |
| Uniform probability average | Nothing | All |
| Constant probability mixture | One input-independent vector of weights | All |
| Soft input gate | Small MLP from input geometry to mixture weights | All |
| Hard input gate | Small MLP from input geometry to one expert choice | One, via sub-batch dispatch |
| Oracle CE diagnostic | Uses the true label to choose the lowest-loss expert | Diagnostic only |

The constant mixture controls for ordinary ensembling/calibration effects. The
hard gate is trained with expected expert cross-entropy,
`sum_e gate_e(x) * CE(expert_e(x), y)`, and selected using **actual hard-dispatch
validation CE**. The soft and constant gates optimize the cross-entropy of the
probability mixture. Soft and hard gates have separate fitted weights: good soft
mixtures do not imply good hard decisions.

The oracle is only an optimistic CE bound for choosing one of these frozen
experts. It is not a bound for mixtures, joint training, or a regional router. Its
class-conditional win rates use labels and cannot be deployed. No oracle ROC,
rejection, or accuracy curve is exported.

The gate uses eleven cheap, permutation-invariant input summaries: constituent
count, mass/pT, pT dispersion, largest and top-three pT fractions, pT entropy,
girth, radial second and fourth moments, angular covariance trace and anisotropy.
These describe multiplicity, concentration and angular structure before any
expert is run. They are rapidity/azimuth observables with a beam reference, not
general Lorentz invariants. Common momentum rescaling leaves them unchanged;
the model inputs are already normalized, so the gate does not pretend to recover
the original dimensional jet pT. No expert logits, embeddings, labels, region
assignments or label-dependent tensor ablations enter the gate.

The defaults are a 16-hidden-unit MLP, Adam at 0.003, up to 40 epochs with eight
epochs of validation patience. `--hidden 0` provides a linear gate control. With
three experts the default gate has 243 parameters. Standardization is fitted
only on the gate-fit split. These defaults are starting points, not tuned jet
results; the selection metric is CE, so rejection must be measured separately.

## Setup and experiment identity

Use the same environment and preprocessing as the checkpoint trainings:

```bash
pip install -r requirements-representation.txt -r requirements-lund.txt
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p routing-artifacts
cp configs/experiments/frozen_routing/experts.example.json routing-artifacts/experts-k8-ca-seed123.json
```

Edit the three checkpoint paths in that manifest. It already contains the exact
188-class ordering used by the existing evaluator. Relative checkpoint paths are
resolved relative to the manifest, not the shell working directory. Match each
`preset` and `model_overrides` to the original training configuration. For
example, an old K=4 checkpoint trained by overriding a K=8 preset can use
`"preset": "regions8_ca_rank2", "model_overrides": {"num_regions": 4}`; the new
`regions4_ca_rank2` preset has the same model configuration.

Weaver/DDP `mod.`, `module.` and `_orig_mod.` prefixes are unwrapped, followed by
strict state-dict loading. Input/class dimensions come from the input cache and
manifest. The cache records checkpoint SHA256, resolved model options,
preprocessing SHA256 and input-shard metadata. Keep checkpoints available at
their recorded paths for direct routed inference; changed contents are rejected.
Use one manifest per K/source/training seed initially. Once those tests pass, a
manifest may mix K values, but then the question includes **granularity as well
as order**. Mixing learned/C/A in the same bank similarly changes the claim.

## 1. Reserve and export disjoint samples

Reserve separate **gate-fit**, **gate-validation**, and **final-test** files.
They must be outside expert training, and the final test must also be outside
expert/gate checkpoint selection. Keep every configuration's file lists,
preprocessing and class sampling identical. Use the resolved training YAML,
especially if any preprocessing was automatic. Do not fit the gate on the
uploaded full-test results or select the next model from a new final-test sweep.

Export manageable `.pt` input shards with the existing exporter. For example,
set these paths to three distinct reserved files (repeat with more files/shards
to obtain representative class coverage):

```bash
python scripts/export_representation_inputs.py \
  --data-config "$RESOLVED_YAML" --files "$GATE_FIT_FILE" \
  --jets-per-file 20000 --max-scanned-per-file 20000 \
  --output routing-artifacts/fit-000.pt
python scripts/export_representation_inputs.py \
  --data-config "$RESOLVED_YAML" --files "$GATE_VAL_FILE" \
  --jets-per-file 20000 --max-scanned-per-file 20000 \
  --output routing-artifacts/val-000.pt
python scripts/export_representation_inputs.py \
  --data-config "$RESOLVED_YAML" --files "$FINAL_TEST_FILE" \
  --jets-per-file 20000 --max-scanned-per-file 20000 \
  --output routing-artifacts/test-000.pt
```

Start with tens of thousands of representative jets for fitting and validation,
then use a sufficiently large held-out evaluation sample. The exporter takes
the **first selected** jets from each file and can apply label caps; this is not
automatically a random or population-representative sample. Its metadata retains
the sampling policy. Label-capped diagnostic samples are unsuitable for claiming
population performance. A few thousand QCD jets cannot resolve rejection in the
1e-3/1e-4 tails. Use full held-out files and report survivor counts for that stage.

Source IDs identify the file and ordinal **after YAML selection**, not a raw ROOT
row. Reuse the same YAML and source paths across splits. Additional hashes of
unpadded model inputs catch copied files and constituent permutations. Duplicate
rows or overlapping source/content identities are rejected. These checks do not
establish that two different detector records are independent at generator level;
reserve files according to the dataset's event provenance. Old input exports can
be used if their recorded YAML is still accessible; new exports embed its hash.

## 2. Cache aligned frozen predictions

```bash
python scripts/cache_representation_experts.py \
  --experts routing-artifacts/experts-k8-ca-seed123.json \
  --inputs routing-artifacts/fit-*.pt --output routing-artifacts/cache-fit \
  --device cuda --batch-size 128
python scripts/cache_representation_experts.py \
  --experts routing-artifacts/experts-k8-ca-seed123.json \
  --inputs routing-artifacts/val-*.pt --output routing-artifacts/cache-val \
  --device cuda --batch-size 128
python scripts/cache_representation_experts.py \
  --experts routing-artifacts/experts-k8-ca-seed123.json \
  --inputs routing-artifacts/test-*.pt --output routing-artifacts/cache-test \
  --device cuda --batch-size 128
```

Every expert sees the same input rows in one exporter. Existing prediction ROOT
files with only `truth_label` and `output` do not prove event alignment, so this
workflow re-evaluates checkpoints instead of assuming equal row order or labels
prove identity. It supports regional checkpoints and the existing small ParT.

The caches contain `.npy` arrays and a JSON manifest. Probabilities are memory
mapped on CPU and copied to the selected device a batch at a time. One 100k-jet,
three-expert, 188-class probability array occupies about 226 MB before identity
and feature arrays. Input `.pt` shards are loaded one at a time; keep each shard
small enough for host memory. All expert weights reside on the chosen device
during caching/direct dispatch. An interrupted export has no valid manifest;
outputs are never silently overwritten.

## 3. Inspect complementarity, then fit

```bash
python scripts/run_frozen_expert_routing.py diagnose \
  --cache routing-artifacts/cache-val \
  --output routing-artifacts/complementarity-k8-ca-seed123.json

python scripts/run_frozen_expert_routing.py fit \
  --fit routing-artifacts/cache-fit --validation routing-artifacts/cache-val \
  --test routing-artifacts/cache-test --output routing-artifacts/gate-k8-ca-seed123 \
  --seed 123 --device cpu --jetclass2-metrics
```

The report includes every expert, uniform/fitted mixtures, soft/hard gates,
per-class and canonical topology-group CE, expert prediction disagreement,
paired CE differences, routing fractions, and the clearly separated oracle CE.
The reference single expert is chosen on **validation**, not by cherry-picking
test outcomes. Paired normal CE intervals condition on the trained checkpoints;
they do not account for training-seed variability, event correlations or multiple
comparisons. Repeat the entire comparison using the independently trained seed
124 experts; changing only the gate seed is not a substitute.

`--jetclass2-metrics` writes per-method ROOT predictions and calls the repository's
unchanged evaluator, including hadronic R50/R30 and fixed-QCD-efficiency metrics.
It requires the canonical class order. ROOT directories `expert_0`, `expert_1`,
etc. follow the manifest's expert order. This deliberately duplicates prediction
storage and can be omitted for the first CE screen. No oracle ROOT file is made.
`per_event.npz` retains aligned CE columns, hard choices and identities for paired
analysis across routing runs. Summary computation retains these compact arrays
in host memory, while the much larger probability array stays memory mapped.

The fitted weights are in `router.pt`. Direct inference actually dispatches only
the selected expert for each jet and restores the original batch ordering:

```bash
python scripts/predict_routed_representation.py \
  --router routing-artifacts/gate-k8-ca-seed123/router.pt \
  --inputs routing-artifacts/test-000.pt \
  --output routing-artifacts/hard-predictions.pt --device cuda --batch-size 128
```

This executable uses no labels to route, even if the input file carries labels.
It evaluates one chosen expert per jet; it still keeps the whole expert bank in
memory. Benchmark preprocessing, gate, dispatch, C/A clustering, expert execution
and device synchronization together before claiming a speedup. Uneven sub-batches
can reduce GPU efficiency. A soft-mixture gain alone is evidence for ensembling,
not reduced compute or adaptive regional order.

## More regions and learned versus C/A

There is no settled winner between learned and C/A regions. Keep C/A as the
primary fixed-partition control and compare learned regions at matched K,
representation, seed, local pair bias and readout. C/A removes assignment learning,
not the learned local frames. Existing learned assignments were diffuse, so
nominal K alone does not establish the same effective granularity. Reuse the
assignment entropy, overlap, hard occupancy and effective-size diagnostics in
`docs/region_followups.md`; shared frozen geometry is an additional control.

Presets now cover **K=4,8,16,32**, learned/C/A, SV/rank-2/wide, with either no pair
bias or the same sparse local pair bias. Existing presets and backbone weights
are unchanged. Rank-2 and wide match the added message-component budget, not
necessarily FLOPs or latency. For short jets C/A has `min(K,N)` active regions;
measure that saturation before interpreting K=32 as a distinct resolution regime.

Priority after the current runs:

1. Repeat K=4 C/A SV/rank-2/wide at seed 124; fill K=16 C/A at seeds 123 and 124.
   This tests whether the K=4 pattern repeats and extends the fixed-partition curve.
2. Screen K=32 C/A at seed 123, repeating it if it changes the conclusion. Run the
   frozen-expert test separately at K=4,8,16 first; extend to K=32 when available.
3. Complete matched learned-region seeds at the informative K values. The supplied
   K=4/K=32 learned presets make this possible without custom overrides.
4. Evaluate the local-pair variants as a separate matched matrix. Do not compare
   a no-pair tensor arm to a pair-enabled wide arm and attribute the gap to order.

The matrix launcher prints commands by default, with unique comments:

```bash
# Preview the 9 highest-priority missing runs; append --execute before -- to run.
python scripts/run_region_sweep.py --regions 4 --sources ca --seeds 124 \
  --comment-prefix aro_repeat -- --train-percentage 10 --num-epochs 80
python scripts/run_region_sweep.py --regions 16 --sources ca --seeds 123 124 \
  --comment-prefix aro_k16 -- --train-percentage 10 --num-epochs 80

# Broader matched plan only: 4 K values x 2 sources x 3 arms x 2 seeds = 48 runs.
python scripts/run_region_sweep.py --regions 4 8 16 32 --sources ca learned \
  --seeds 123 124 --pair-bias none --comment-prefix aro_matrix

# Optional stronger-backbone comparison, executed serially with existing launcher.
python scripts/run_region_sweep.py --regions 8 --sources ca learned --seeds 123 \
  --pair-bias local --comment-prefix aro_localpair --execute -- --train-percentage 10
```

Match epochs and all other training settings to the completed runs, rather than
assuming the example's 80 epochs. Do not relaunch existing checkpoints blindly.
`DATA_PATH`, `OUTPUT_PATH`, batch size and DDP settings follow the existing
launcher. A failure stops the sweep. There is no automatic resumption or GPU
scheduler, and listing a matrix is not a recommendation to train all 48 runs.

## How to interpret the gate

| Held-out result, repeated across checkpoint seeds | Next step |
|---|---|
| Tiny oracle CE improvement | These experts offer little useful selection headroom under CE |
| Large oracle gap, no learned-gate gain | Complementarity is not captured by these cheap gate features/training; this does not disprove every possible router |
| Soft beats a single model but matches the constant mixture | Ordinary ensemble effect; no evidence for input adaptivity |
| Soft beats the fitted constant mixture, hard does not | Input-dependent mixing helps; conditional one-expert execution remains unsupported |
| Hard beats the validation-selected single expert and improves the relevant working points | Proceed to a shared-backbone/regional router, then evaluate matched total cost |

Even the strongest result here is a per-jet frozen-expert result. To establish
adaptive **representation order per region**, the follow-up must hold geometry
and lower-order features shared, provide competing SV/wide/rank-2 message paths,
and show that conditional selection beats dense/fixed allocations at measured
cost. None of that is established by an oracle curve or a post-hoc class winner.

## Validation

The implementation was checked on CPU with the existing 40 representation/Lund
tests and 11 new tests. The new tests cover masked/permuted/rescaled gate inputs,
beam-axis rotation/longitudinal-boost behavior, source/content overlap rejection,
class/expert order checks, a learnable synthetic gate and an uninformative-input
control, a real-checkpoint cache/fit/direct-dispatch round trip, skipped expert
calls, ROOT schema compatibility, and the matched preset/launcher matrix.

```bash
PYTHONPATH=src:. python -m pytest -q tests/test_representation_order.py \
  tests/test_representation_controls.py tests/test_region_followups.py \
  tests/test_frozen_expert_routing.py
```

No new JetClass-II checkpoint training or GPU latency measurement was performed
as part of implementing this workflow.
