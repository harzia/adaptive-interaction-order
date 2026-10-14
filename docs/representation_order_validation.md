# Implementation validation

Validation performed on CPU with Python 3.12, PyTorch 2.14.1+cpu and Weaver 0.5.3.
No JetClass-II files or GPU were available in the implementation environment.

- Numerical tests cover proper Lorentz frames, covariant predictions with transformed
  physical reference vectors, scalar/vector and full rank-2 transport against an
  explicit receiver-frame attention calculation, permutation/padding behavior,
  degenerate and short jets, identity grouping, unequal region-mass weighting,
  tensor gate gradients, frozen geometry reuse, matched parameter counts,
  bf16 autocast, and the Weaver adapter's regularized objective.
- The actual stock Weaver training function ran two optimizer steps through the
  adapter on synthetic inputs. The repository's validation evaluator then completed
  one synthetic batch. This checks the integration, not data loading or physics quality.
- All six primary presets completed synthetic inference profiling at 48 particles.
  These tiny CPU timings are not evidence for a GPU accuracy/cost frontier.
- The rank-2 smoke model's CE on one fixed synthetic batch decreased from approximately
  1.151 to 0.573 over 20 AdamW updates; all trainable parameters had finite gradients.
- Shell syntax, the preset dry-run launch, and Python import/compilation were checked.

The K=8 tensor and wider-vector presets each have 403,847 trainable parameters,
versus 337,543 for K=8 scalar/vector. They have equal component counts, not certified
equal runtime. The default classifier has 188 outputs and a 512-unit hidden layer.

Still required on the user's training environment: actual JetClass-II training,
multi-seed comparisons, region-locality inspection, target-GPU timing/memory,
and multi-GPU execution if DDP is used. No accuracy improvement, infrared/collinear
safety, adaptive-order speedup, or published LLoCa-ParT reproduction is claimed.
