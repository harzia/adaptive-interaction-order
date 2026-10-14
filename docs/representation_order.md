# Learned regions and representation order: first experiments

This adds an **independent small LLoCa-style pilot backbone** with the existing
`src/aio`, Weaver adapter, dataset config, launcher and evaluation layout.
Existing ParT and F3 code is unchanged.

It is **not** a reproduction of published LLoCa-ParT or its reported accuracy.
The new model omits ParT pair biases and class attention, uses a smaller locally
conditioned frame predictor, and uses a masked mean readout. The full-attention
arm is the reference for THIS family. All arms share depth, hidden width, local
branch and readout.

## Operator

1. Predict input-conditioned particle frames and **learned soft assignments** to K
   coarse regions. Regions are not fixed subjets. K is fixed per experiment.
2. Apply local scalar/vector attention on the same geometric k-nearest-neighbor
   graph in every arm. This graph does not depend on K or representation order.
3. Project each particle into learned scalar/vector/(optional) rank-2 Q/K/V in
   its local frame. Transport those messages into a common frame **before pooling**.
4. Pool geometric Q/K/V using learned memberships. Apply K-by-K regional attention
   with Minkowski contractions and `log(region_mass)` key-logit weighting.
5. Interpolate the output to particles, transform geometric outputs into each
   receiver's frame, and apply residual/FFN updates.

The full arm replaces regional communication with all-particle geometric attention,
while retaining the shared local branch. Local neighborhoods remain fixed-size even
at K=1. The final readout sees all fine states: this is a **communication bottleneck**,
not a study of compressed inputs as sufficient statistics for jets.

The tensor is a **general learned 4x4 Lorentz tensor** in Q, K AND V. It is not a
symmetric trace-free tensor, angular ell=2 multipole, bivector, or MomentKV correction.
Rank denotes geometric tensor rank, not matrix rank or interaction body order.
Scalar channels overlap with the tensor trace; irreducible decomposition is a later
ablation. Queries are pooled too, so regional attention is not an exact approximation
for every original particle query.

There is **no learned order selector or adaptive region count yet**. Fixed-order
variants establish whether richer representations help. The optional `tensor_gate`
is an intervention for diagnostics; it does not skip tensor computation.

## Presets

All arms use 4 blocks, hidden width 64, FFN width 128, 4 heads, an 8-neighbor local
branch, and the launcher's `fc_params=[(512,0.1)]`. Geometry is computed once per
forward and reused. Assignments depend on inputs, not order-dependent hidden states.

| Preset | Regions | Global message channels per head | Components |
|---|---:|---|---:|
| `full_sv` | All particles | 8 scalars + 2 vectors | 16 |
| `regions16_sv` | 16 | 8 scalars + 2 vectors | 16 |
| `regions8_sv` | 8 | 8 scalars + 2 vectors | 16 |
| `regions16_rank2` | 16 | 8 scalars + 2 vectors + 1 full tensor | 32 |
| `regions8_rank2` | 8 | 8 scalars + 2 vectors + 1 full tensor | 32 |
| `regions8_wide` | 8 | 8 scalars + 6 vectors | 32 |
| `regions16_wide` | 16 | Optional corresponding width control | 32 |

Tensor and wide arms at the same K have exactly matched trainable parameter counts
and message-component counts. They are **not automatically runtime-matched**:
tensor transport differs. Profile on the target GPU before making an efficiency claim.

Balance, assignment entropy and observer-frame angular compactness penalties are
added to the TRAINING loss. Validation reports CE only. Their explicit coefficients
in `common.json` are starting values, not tuned physics results. K is a slot budget,
not a guarantee of K disjoint, nonempty subjets. Inspect entropy/mass/locality diagnostics.

## Install and verify

From the repository root, with Python >=3.10 and an appropriate PyTorch build:

```bash
python -m pip install -r requirements-representation.txt
python -m pip install -e . --no-deps
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest tests/test_representation_order.py -q
python scripts/smoke_representation_order.py --preset regions8_rank2 --steps 20
```

The smoke program optimizes a fixed SYNTHETIC batch. No datasets or trained
checkpoints are included. The adapter reuses this repository's evaluator and Weaver's
standard training loop, wrapping the training objective to add region penalties.
Single GPU and Weaver's DDP path are supported; DataParallel training is rejected
because replica-local auxiliary losses would not be collected correctly.

## Launch the first runs

The data paths, 188 labels, splits, reweighting, optimizer and learning rate come
from `run_jetclass2.sh`, as in the triplet experiments. The experiment wrapper defaults
to 80 epochs, seed 123 and 10% training data; these are configurable defaults.
Kinematics-only YAML and `(px,py,pz,E)` input order are unchanged.

```bash
export DATA_PATH=/path/to/data
export OUTPUT_PATH=/path/to/outputs
export SEED=123

# Prints the command without starting a job or requiring data:
AIO_DRY_RUN=1 bash scripts/run_representation_experiment.sh regions8_rank2

bash scripts/run_representation_experiment.sh full_sv
bash scripts/run_representation_experiment.sh regions16_sv
bash scripts/run_representation_experiment.sh regions8_sv
bash scripts/run_representation_experiment.sh regions16_rank2
bash scripts/run_representation_experiment.sh regions8_rank2
bash scripts/run_representation_experiment.sh regions8_wide
```

Each command launches one run, retaining the base script's train-then-test behavior.
Repeat the important comparisons across seeds. For example:

```bash
bash scripts/run_representation_experiment.sh regions8_rank2 \
  --num-epochs 30 --seed 42 --train-percentage 25 \
  --batch-size 256 --start-lr 1e-4
```

The three training options accept both `--option VALUE` and `--option=VALUE`.
Environment defaults `EPOCHS`, `SEED` and `TRAIN_PERCENTAGE` are also supported;
command-line options take precedence. Training percentage is an integer from 1 to 100.
The base launcher selects the first ceil(percentage * file_count / 100) training
files in each category and scales the samples-per-epoch budget. It does not randomly
select that exact fraction of events. Validation and test file lists stay unchanged;
the seed does not change the selected file subset.

All other arguments pass through unchanged to the base launcher / Weaver, including
`--batch-size`, `--start-lr`, `--num-workers`, `-o NAME VALUE`, resume options,
and `--test-only --weights PATH`. Weaver validates its own options. Model and data
overrides are allowed, but change the preset comparison.

The default run name includes preset, learned/frozen geometry, percentage, seed and
epochs, e.g. `regions8_rank2-learned-25pct-seed42-epochs30`.
Use `AIO_RUN_COMMENT` for a custom name and to distinguish other overrides such as
learning rate or model options. For a short actual-data check:

```bash
bash scripts/run_representation_experiment.sh regions8_rank2 \
  --num-epochs 1 --train-percentage 1 --seed 123
```

Use `AIO_DRY_RUN=1` with any of these commands to inspect the resolved options.

These first commands compare **jointly learned** grouping and representation. Their
groups can diverge during training even with identical seeds. For an isolated order
comparison, freeze shared learned geometry as below.

## Share learned regions across order variants

Train `regions8_sv`, then export its grouping AND frame predictor:

```bash
python scripts/export_region_geometry.py \
  --preset regions8_sv \
  --checkpoint "$OUTPUT_PATH/RegionLLoCa/JetClass2_kin/regions8_sv-learned-10pct-seed123-epochs80/training/net_best_epoch_state.pt" \
  --output learned_geometry_k8.pt

export GEOMETRY_CHECKPOINT="$PWD/learned_geometry_k8.pt"
bash scripts/run_representation_experiment.sh regions8_sv
bash scripts/run_representation_experiment.sh regions8_rank2
bash scripts/run_representation_experiment.sh regions8_wide
unset GEOMETRY_CHECKPOINT
```

These reruns train fresh backbones with exactly the same frozen input-to-region
mapping and frames. The `frozen` suffix distinguishes their output directories.
K=16 requires its corresponding checkpoint; incompatible specifications are rejected.
SV-trained geometry can favor SV: report provenance and keep joint training separate.
This stronger comparison needs additional runs beyond the first six configurations.
Keep upstream preprocessing identical when sharing geometry checkpoints.

## Profile actual cost

```bash
python scripts/benchmark_representation_order.py --device cuda \
  --batch-size 128 --particles 128 --amp --backward --output training_cost.json
python scripts/benchmark_representation_order.py --device cuda \
  --batch-size 128 --particles 48 --amp --output inference_cost.json
```

Profile both actual multiplicity and the 128-particle padded input used by the loader.
This model does not dynamically trim padding. Timing includes geometry, assignment,
frame transforms, pooling/unpooling, attention, FFNs and readout. Backward timing
excludes the optimizer. CPU numbers are functional checks, not GPU speedup claims.

Local attention gathers N*L neighbors; regional attention evaluates K*K pairs.
**kNN construction still computes an N*N distance matrix once per forward**.
Assignments cost N*K and pooling costs depend on message width. No end-to-end
linear-time or linear-memory claim is made. Frame transport is factorized per token;
no N*N*4*4 pairwise transport tensor is constructed.

## Symmetry and numerical conventions

Internal vectors are `(E,px,py,pz)`, metric `diag(1,-1,-1,-1)`. The model uses a unit
timelike observer and unit spacelike beam: by default `(1,0,0,0)` and `(0,0,0,1)`.
Optional `references=[B,2,4]` must have norms +1/-1 and zero mutual inner product.

For genuinely scalar auxiliary inputs and references transformed with momenta,
the geometry is covariant under proper orthochronous Lorentz transformations.
JetClass-II auxiliary features and jet-pT scaling are lab-defined: the supplied
training pipeline has detector-reference dependence, not arbitrary-boost invariance.
Transforming only momenta while holding references/features fixed is a different input.

Frames use float64 internally; geometric attention uses at least float32, even with
bf16 autocast. Learned boost directions are bounded relative to the observer.
Degenerate transverse frames are flagged and their geometric channels zeroed;
scalar processing remains. Identity placeholders are never used as valid geometric
frames. Monitor invalid-frame frequency. No special permutation-invariant tie-breaking
rule is supplied for exactly tied neighbors. Tests cover generic distinct particles.

Input momenta must be physical and units consistent. Masked padding is zeroed before
geometry, including NaN padding; all-empty jets are rejected. No infrared/collinear
safety theorem is claimed. No ONNX or fully compiled performance claim is made.

## Region and tensor diagnostics

```python
logits, geometry = model(features, vectors, mask, return_aux=True)
assignments = geometry["assignments"]  # [batch, particles, regions]
```

Plot these assignments against constituent coordinates to check locality/collapse.
The forward also returns masses, frames, masks and regularization. For tensor utility:

```python
gate = torch.ones(features.shape[0], model.num_regions,
                  device=features.device, requires_grad=True)
logits = model(features, vectors, mask, tensor_gate=gate)
loss = torch.nn.functional.cross_entropy(logits, labels, reduction="sum")
utility = -torch.autograd.grad(loss, gate)[0]
```

Positive utility predicts increased loss on removal. Validate selected cases with
actual zero-gate forward ablations. Gates act on regional tensor Q/K/V and are shared
across blocks. They neither change membership nor skip computation.

## Attribution

- Local canonicalization and tensor transport: [LLoCa](https://arxiv.org/abs/2508.14898),
  [official implementation](https://github.com/heidelberg-hepml/lloca).
- Hierarchical equivariant pooling: [EGHN](https://arxiv.org/abs/2202.10643).
- [MomentKV](https://arxiv.org/abs/2606.01563) concerns a distinct moment correction;
  that operator is not implemented or claimed novel here.

This code is the experiment instrument. Accuracy gains, colour-flow interpretation,
GPU speedups and the usefulness of adaptive order require the actual experiments.
