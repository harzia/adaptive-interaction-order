# Angular-moment attention on small ParT

The experiment asks whether explicit angular correlations in a particle block can
replace some depth. It starts from the repository's **actual ParT**: full particle
self-attention, its four pair features and pair embedding, the original FFNs, and
two class-attention blocks. There is no region bottleneck in this experiment.

The extension gives the existing attention weights geometric values to aggregate,
then forms invariant contractions and typed products before the FFN. This tests a
single mechanism: **learn angular moments, correlate them, and carry their state
between blocks**. Adaptive regional allocation is a later question, after this
dense operator demonstrates a benefit against ordinary ParT.

## Precisely what the features are

Use the valid, retained particle four-momenta, after the same sequence trimming
and permutation as ParT. Boost them to one common jet rest frame. Define

\[
 z_i=E_i^*/\sum_j E_j^*,\qquad n_i=\mathbf p_i^*/|\mathbf p_i^*|,
 \qquad Q_i=n_i n_i^\top-I/3.
\]

For each attention head and feature channel, learn a scalar amplitude from the
ordinary particle embedding, using the same amplitude for every retained degree:

\[
 a_i^c=z_i\,\operatorname{softplus}(f_c(h_i)),\qquad
 S_i^c=a_i^c,\quad V_i^c=a_i^c n_i,\quad T_i^c=a_i^c Q_i.
\]

| State | Components per channel | Meaning at initialization |
|---|---:|---|
| Scalar `S` | 1 | Learned energy weight |
| Vector `V` | 3 | Weighted angular dipole |
| Tensor `T` | 5 | Weighted symmetric trace-free spatial quadrupole |

The quadrupole uses an orthonormal five-component STF basis; decoding gives a
3-by-3 symmetric trace-free matrix. These are SO(3) angular degrees, not arbitrary
reshaped hidden channels and not general 4-by-4 Lorentz tensors. Learned maps mix
feature channels; they do not arbitrarily mix spatial components.

The physical distinction is visible even before learning: equal-weight planar
and tetrahedral arrangements can have identical zero dipoles but different
quadrupoles. This is a statement about these moments, not an impossibility theorem
for wider vector networks with learned weights.

## One block, including the nonlinear operations

1. Keep the ordinary scalar queries, keys, pair bias, and softmax. Mix geometric
   channels independently within each angular degree.
2. Concatenate ordinary and typed **values** into one attention operation. Both
   receive exactly the same attention weights and attention-dropout realization.
   Query/key width and the attention scaling are unchanged.
3. Normalize the geometric messages using one invariant scale per particle and
   head, shared over the retained degrees:

   \[
   d_i=\sqrt{10^{-8}+\sum_{\ell}\frac1C\sum_c
           \|M_i^{\ell,c}\|^2},\qquad \widehat M_i=M_i/d_i.
   \]

4. Form scalar features `S`, `V.n_i`, `T:Q_i`, and `log1p(d_i)`. With explicit
   products enabled, also form the upper-triangular cross-channel Gram features
   `V^a.V^b` and `T^a:T^b`. A learned linear map adds these scalars to the normal
   attention residual **before the existing FFN**.
5. Except at the last particle block, update each typed state by adding the
   normalized message through a learned scalar sigmoid gate. For the L=2 product
   model, also add learned channel combinations of `T^a V^b` to the vector update
   and `STF(V^a outer V^b)` to the tensor update. Gates are computed from the
   ordinary attention residual. The scalar FFN output and updated typed state
   enter the next block.

The final particle block only scalarizes its messages: it has no unused state
update parameters. Class blocks and the classifier consume the resulting ordinary
particle embeddings. They are unchanged.

The scalar residual projection starts at zero. The typed product coefficients
also start at zero, while channel mixing starts at the identity. Common ParT
parameters receive the same initialization as the baseline at a matched seed.
The scalar residual learns first; its upstream geometric parameters start learning
after it opens. A test checks this with actual optimizer steps from initialization.

Later typed states are learned covariant features, not literal positive energy
moments. The initial moment construction explains their physical basis; the
specified algebra explains their subsequent use.

## Why products, rather than tensor-shaped capacity

For an initial pooled tensor with effective weights `w_j`, its contraction obeys

\[
 \left(\sum_j w_j Q_j\right):\left(\sum_k w_k Q_k\right)
 =\sum_{j,k}w_jw_k\left[(n_j\!\cdot n_k)^2-\frac13\right].
\]

Thus a single attention aggregation followed by a tensor contraction exposes an
angular correlation among senders. The receiver-dependent attention weights
condition which radiation contributes. The corresponding vector contraction uses
`n_j.n_k`. Products of normalized moments retain this angular interpretation with
an additional common scale factor. After state updates the channels have the
same transformation types, but need not have this simple initial moment formula.

**Angular degree L and product degree are different controls.** L=2 introduces
quadrupoles; product degree 2 introduces explicit products of pooled features.
Attention, scalar FFNs, gates, and normalization are already nonlinear, so the
product flag does not impose an exact body-order bound on the whole network.
No novelty or performance claim follows just from this identity.

## Presets and comparisons

Configs live in `configs/experiments/angular_moments/`. Ordinary baselines
instantiate `ParticleTransformer` directly. All use four attention heads, the
same pair embedding, FFN expansion, dropout recipe, two class blocks, and the
512-wide classification head. Width is 48 except the explicit width control.

Parameter counts below use seven input features and 188 classes, including the
pair embedding and classifier. They are not FLOP or latency matches.

| Preset | Particle blocks | Geometric state | Parameters | Purpose |
|---|---:|---|---:|---|
| `part_d2` | 2 | None | 265,756 | Shallow ordinary ParT |
| `part_d4` | 4 | None | 323,364 | Existing `part_small` configuration |
| `part_d8` | 8 | None | 438,580 | Deeper small ParT |
| `part_w64_d2` | 2 | None, scalar width 64 | 376,252 | Wider ordinary control |
| `moments_l1_d2` | 2 | S/V, 2 channels per head | 270,068 | Dipole extension |
| `moments_l2_d2` | 2 | S/V/T, 2 channels per head | 272,476 | Quadrupole extension |
| `moments_l1_d4` | 4 | S/V, 2 channels per head | 332,380 | Dipole depth control |
| `moments_l2_d4` | 4 | S/V/T, 2 channels per head | 337,652 | Quadrupole depth control |
| `moments_l1_c5_d2` | 2 | S/V, 5 channels per head | 279,080 | Wider geometric control |
| `moments_l2_noprod_d2` | 2 | S/V/T, no explicit products | 270,108 | Product ablation |
| `moments_l2_reset_d2` | 2 | Fresh S/V/T each block | 271,236 | No recurrent typed state |

The wider S/V control has 20 extra value components per head versus 18 for S/V/T.
This approximately matches component count, not runtime or parameter count.
`part_w64_d2` is a capacity control candidate, not a declared cost match.

`noprod` retains linear directional contractions and invariant normalization; it
removes explicit Gram features and the learned TV/VV state products. It is not a
fully linear model. `reset` reseeds moments from the current scalar embedding at
every block. It also removes recurrent gates and typed cross-products whose
outputs would otherwise be discarded. It tests the recurrent typed pathway as a
whole, rather than perfectly isolating persistence from those products.

## First runs

Use the existing JetClass-II environment. The extension adds no package
dependency beyond the existing PyTorch/Weaver setup. From the repository root:

```bash
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"

# Inspect the default eight-run plan: four models, seeds 123 and 124.
python scripts/run_angular_suite.py

# Inspect one fully resolved launcher command without touching training data.
AIO_DRY_RUN=1 bash scripts/run_angular_experiment.sh moments_l2_d2

# DATA_PATH and OUTPUT_PATH must point to your existing data/run locations.
# Execute the same eight runs serially; a failed job stops the suite.
python scripts/run_angular_suite.py --execute -- \
  --train-percentage 10 --num-epochs 80
```

The default plan is `part_d2`, `part_d4`, `moments_l1_d2`, `moments_l2_d2`.
It first measures the depth gap and the L=1/L=2 effect on exactly the same shallow
backbone. Each run preserves the existing kin preprocessing, file splits,
reweighting, optimizer, scheduler, AMP, evaluation, and checkpoint convention.
Defaults are 10% training files, 80 epochs, and seed 123 for a single launch.
Use the same resolved recipe as earlier runs when making a comparison.

To include the deeper baseline immediately:

```bash
python scripts/run_angular_suite.py \
  --presets part_d2 part_d4 part_d8 moments_l1_d2 moments_l2_d2 \
  --seeds 123 124 --execute -- --train-percentage 10 --num-epochs 80
```

Single launches and follow-up controls:

```bash
bash scripts/run_angular_experiment.sh moments_l2_d2 --seed 123
bash scripts/run_angular_experiment.sh moments_l1_c5_d2 --seed 123
bash scripts/run_angular_experiment.sh moments_l2_noprod_d2 --seed 123
bash scripts/run_angular_experiment.sh moments_l2_reset_d2 --seed 123
bash scripts/run_angular_experiment.sh part_w64_d2 --seed 123
bash scripts/run_angular_experiment.sh moments_l2_d4 --seed 123
```

Do not run a suite and its individual commands under the same run names.
`COMMENT` defaults to `angular_<preset>_seed<seed>`. Set a distinct `COMMENT` for
custom hyperparameters or data fractions; use `--comment-prefix` for a suite.
CLI options override `EPOCHS`, `SEED`, and `TRAIN_PERCENTAGE`. Existing
`DDP_NGPUS`, `BATCH_SIZE`, and `NUM_WORKERS` controls remain available.

Both families use the `AngularParT` Weaver adapter, so outputs are under
`$OUTPUT_PATH/AngularParT/JetClass2_kin/COMMENT/`. The baseline model inside that
adapter is still ordinary ParT. For test-only evaluation, use the matching preset:

```bash
bash scripts/run_angular_experiment.sh moments_l2_d2 \
  --weights /path/to/net_best_epoch_state.pt
```

`part_d4` has the same config and checkpoint keys as the old `part_small`. An
existing run can serve as that reference if the preprocessing and training recipe
match. A fresh seed remains useful for uncertainty. For warm starts in Python,
`AngularParticleTransformer.load_part_backbone()` accepts an exact, unwrapped
ordinary ParT state dict; fresh training is the default comparison.

## Measure cost before calling a result efficient

```bash
python scripts/benchmark_representation_order.py --angular --device cuda \
  --presets part_d2 part_d4 part_d8 moments_l1_d2 moments_l2_d2 \
  --batch-size 512 --particles 128 --amp --steps 100 --warmup 10 \
  --output angular-artifacts/inference_cost.json

python scripts/benchmark_representation_order.py --angular --device cuda \
  --batch-size 512 --particles 128 --amp --backward --steps 50 --warmup 10 \
  --output angular-artifacts/training_cost.json
```

Use a common batch size that fits every compared model. Also benchmark batch size
1 for latency-sensitive use. Timing includes frame construction, pair embedding,
geometric operations, class blocks, and classifier. Fixed-N timing disables random
sequence trimming for both families. The report includes parameters, batch time,
throughput, peak allocated CUDA memory, and frame-fallback diagnostics. The
training mode times forward/backward, not optimizer updates.

Appending geometric values makes value width differ from query/key width. This
can change SDPA kernel selection or prevent FlashAttention on a particular GPU /
PyTorch version. The extra arithmetic and float64 boost can outweigh the savings
from fewer blocks. These commands measure that; neither parameter counts nor CPU
smoke timings establish a compute advantage. FLOPs are not estimated here.

Compare aggregate and per-class held-out rejection, calibration/loss, and measured
cost across matched seeds. The core target is whether two richer particle blocks
recover the four-block baseline's performance, and whether quadrupoles outperform
spending the same available budget on vectors, scalar width, or depth. Heterogeneous
class gains alone do not establish that an adaptive router can select those gains.

## Numerical and symmetry boundaries

- The frame uses only valid retained inputs; padding is excluded. The scalar
  backbone preserves ParT's finite wrap-padding before input BatchNorm, avoiding a
  hidden preprocessing change. Geometry follows exactly the trimmer's indices.
- Rest-frame calculations use float64 after removing the common energy scale.
  For `M^2/E^2 <= 1e-6`, non-timelike totals, or negative candidate boosted
  energies, the whole jet falls back to the detector frame. Zero spatial momentum
  has zero dipole and quadrupole, with a scalar contribution only. Empty jets,
  negative input energies, and non-finite valid momenta are rejected.
- The typed algebra is rotation covariant for invariant scalar coefficients and
  attention weights. The full model retains detector-dependent input features and
  ParT pair bias, and the fallback is detector referenced. **This is not a claim
  of full Lorentz equivariance or IRC safety.**
- Geometry and nonlinear typed algebra run outside autocast (float64 and parameter
  precision respectively); values inside attention use its autocast precision.
- `moment_scale=0` bypasses the whole extension and recovers the ordinary ParT
  execution path exactly. It is an evaluation/debug control, not a DDP training
  preset with unused parameters. An active zero-initialized residual is
  numerically equivalent initially but may select a different attention kernel.
- Eager ParT v1 classification is supported. Compilation, ONNX export, global
  class tokens inside particle blocks, and segmentation are outside this initial
  implementation. No GPU training or deployment claim is implied by CPU tests.

## Validation and source map

```bash
PYTHONPATH=src:. python -m pytest -q tests/test_angular_moments.py
PYTHONPATH=src:. python -m pytest -q tests
```

Tests cover the STF contraction identity; rest-frame boost/scale behavior and
degenerate inputs; rotation covariance with nonzero learned geometric updates;
identical ordinary/typed attention weights including dropout and gradients; SDPA
agreement with explicit attention; baseline initialization and training BatchNorm
statistics; finite fp32/bf16 gradients; optimizer updates from zero initialization;
particle permutation, padding, trimmer alignment, strict checkpoints; Weaver
integration; and all preset launch paths. Symmetry tests of the geometric operator
are separate from the weaker symmetry properties of the whole classifier.

| File | Responsibility |
|---|---|
| `src/aio/core/geometry/angular_moments.py` | Frame, physical basis, typed algebra |
| `src/aio/models/jets/angular_particle_transformer.py` | ParT subclass and state lifecycle |
| `src/aio/models/jets/particle_transformer.py` | Optional extra-value attention and pre-FFN hook; default path unchanged |
| `src/aio/models/jets/angular_experiments.py` | Shared configuration factory |
| `integrations/weaver/AngularParT.py` | Existing training/evaluation integration |
| `scripts/run_angular_experiment.sh` / `scripts/run_angular_suite.py` | Single jobs and explicit suites |

## Grounding in prior work

- [ParT](https://arxiv.org/abs/2202.03772) supplies the backbone and pair bias.
- [Energy Flow Networks](https://arxiv.org/abs/1810.05165) and
  [Energy Flow Polynomials](https://arxiv.org/abs/1712.07124) motivate an explicit
  energy/angular basis and angular correlations; this unrestricted architecture
  does not inherit their IRC-safe constructions automatically.
- [SE(3)-Transformer](https://arxiv.org/abs/2006.10503) and
  [Equiformer](https://arxiv.org/abs/2206.11990) ground the use of typed geometric
  values, contractions, and equivariant products.
- [MACE](https://arxiv.org/abs/2206.07697) motivates testing explicit products of
  pooled features as an alternative to obtaining complexity only through depth.

The experiment tests this particular mechanism on the existing jet task. It does
not claim that angular moments, equivariant attention, or tensor products are new.
