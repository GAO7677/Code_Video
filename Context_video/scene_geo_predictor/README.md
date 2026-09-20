# Scene Geo Predictor

This directory is the code-only snapshot of the Future Query Predictor scene-geometry experiments.
The source snapshot is `visual_scene_context8_20260918`; large results, caches, checkpoints,
and videos remain under `/data/gaoya/agent-data` and are referenced by the experiment report.

## Layout

- `code/`: predictor, geometry, data preparation, training, evaluation, and visualization scripts.
- `tests/`: CPU/unit/interface tests copied from the source snapshot.
- `configs/`: small experiment configuration files.
- `docs/`: source notes and the complete experiment report.
- `SOURCE_MANIFEST.json`: SHA256 manifest of this code snapshot.

## Start here

Read [`docs/EXPERIMENT_REPORT_20260920.md`](docs/EXPERIMENT_REPORT_20260920.md). The main model
implementation is [`code/future_query_predictor.py`](code/future_query_predictor.py). The latest
independent-history diagnostic launcher is
[`code/train_independent_history_holdout.py`](code/train_independent_history_holdout.py).

This snapshot does not include the data or outputs. Reproduction commands in the report point to
the original workspace and explicitly label diagnostics versus formal training.

## Verification

The portable code/config subset passes 27 CPU unit and interface tests. The real-cache
scene-integration test is retained under `tests/` but is not run here because this snapshot
intentionally excludes the aligned-scene and observed-context results.
