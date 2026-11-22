# Relative particle attention for JetClass-II

Train the hybrid model with exact linear pair scores and 15 rotary value planes, or the fully rotary model with 30 query/key planes and 15 value planes. Both use four particle blocks and the same ParT class readout. Set `DATA_PATH` and `OUTPUT_PATH` as in your JetClass-II Weaver environment.

```bash
python scripts/run_relative_experiment.py linear_pair_rotary_values15_d4
python scripts/run_relative_experiment.py rotary_qk30_rotary_values15_d4
```

The [model presets](src/aio/models/jets/presets) use the same names as the launcher and output directories. The [run guide](docs/relative_attention.md) lists the matched baselines and ablations, the original proposal labels, checkpoint evaluation, and profiling commands.

The [frozen-embedding head test](docs/embedding_head_test.md) checks whether a trained tagger misses information that standard substructure observables carry, without retraining it:

```bash
CHECKPOINT=/path/to/net_best_epoch_state.pt ./scripts/run_embedding_head_test.sh
```
