# PrivacyBench-FL Project Flow

```text
Adult Income Dataset
        |
        v
Schema validation + leakage checks
        |
        v
Temporal client partitioning (Hospital / client silos)
        |
        v
Local preprocessing with fixed feature vocabulary
        |
        v
T1 -> T2 -> T3 continual-learning schedule
        |
        +--> Local drift detector --> adapt LR / local epochs
        |
        +--> DP-SGD + clipping --> per-client epsilon budget gate
        |
        +--> Personalized MLP head stays private on each client
        |
        v
Quantized client update (16-bit warm-up -> 4-bit late rounds)
        |
        v
Cosine-similarity-aware aggregation
        |
        v
Shared global encoder update
        |
        +--> global utility, F1 and loss
        +--> client privacy, bandwidth and similarity telemetry
        +--> T1/T3 retention, forgetting and backward transfer
        |
        v
CSV history + benchmark report + Streamlit dashboard
```

## Fresh-run behavior

`main.py` clears only generated artifacts before preprocessing begins:

- `experiments/full_system_results.npy`
- `experiments/training_history.csv`
- `results/benchmark_report.md`
- contents of `checkpoints/`

It never deletes the raw dataset, client partitions, source code, or configuration.
