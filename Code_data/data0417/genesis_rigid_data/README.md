# Genesis Rigid Data Tools

This directory now has two primary generation entry points and a small set of
shared inspection / repair tools. Older scratch files and the retired early
rigid generator have been removed to reduce ambiguity.

## Environment

Run everything in the `wan` conda environment.

```bash
conda activate wan
cd /home/gaoya/Code_Video/Code_data/data0417
```

## Current Entry Points

### 1. Multi-object train generation

Use:

- `generate_physxnet_train_rigid_multi.py`

Purpose:

- Generate the current rigid training layout under `train/rigid/...`
- Organize samples by `scene_composition` and `object_count_bucket`
- Reuse the current PhysXNet rigid backend instead of a separate legacy export pipeline

Example:

```bash
python genesis_rigid_data/generate_physxnet_train_rigid_multi.py \
  --output_root /data/gaoya/AAA_test_video/Dataset_physV/0417data/physxnet_train_rigid_multi \
  --num_samples 64 \
  --seed 20260419
```

Useful quick check:

```bash
sh /home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/run0417.sh
```

`run0417.sh` now runs a smoke test with the current train generator and then
visualizes the first generated sample.

### 2. Benchmark generation

Use:

- `generate_rigid_benchmark.py`

This is the unified benchmark CLI. It replaces older one-off benchmark drivers.

Supported subcommands:

- `physxnet_pool`
  - Build the flat benchmark pool under `train/rigid/<scene>/<count_bucket>/<sample>`.
- `stage1_heldout`
  - Build the held-out single-object motion benchmark and optionally rebuild subsets.
- `benchmark_v1`
  - Build the `benchmark_v1` dev/test subsets.
- `qa_existing`
  - Scan existing benchmark samples and write QA summaries.
- `motion_qa`
  - Run motion-based QA and optional quarantine handling.

Example:

```bash
python genesis_rigid_data/generate_rigid_benchmark.py physxnet_pool \
  --output_root /data/gaoya/AAA_test_video/Dataset_physV/0417data_benchmark \
  --num_objects 50 \
  --random_seed 20260423 \
  --cases_per_object 3 \
  --case_pool 5 6 7 \
  --rigid_target_object_count 1 \
  --overwrite
```

### 3. Captions and inspection

Main tools:

- `generate_video_captions.py`
  - Generate structured captions for exported samples.
- `visualize_sample.py`
  - Create per-sample visualization videos from physics arrays.
- `inspect_physics_sample.py`
  - Create static summary figures for one sample.
- `batch_inspect_physics_samples.py`
  - Batch-build sample pages and local browsing assets.
- `validate_saved_dataset_states.py`
  - Validate exported states and build a richer inspection portal.

### 4. Audit / repair utilities

Use these only when needed:

- `audit_benchmark_inertial_origins.py`
- `regenerate_affected_benchmarks.py`
- `filter_single_object_motion_cases.py`
- `validate_energy_cases.py`

## Shared Modules

These are meant to be reused by the entry points above:

- `utils_io.py`
  - Shared JSON, array, image, and video helpers.
- `scene_templates.py`
  - Shared scene sampling helpers reused by rigid generation code.

## Low-level Backends

These files are still active, but they are implementation backends rather than
the preferred user-facing entry points:

- `try1_physxnet_benchmark.py`
  - Shared rigid export backend used by `generate_physxnet_train_rigid_multi.py`
    and parts of `generate_rigid_benchmark.py`.
- `try1_physxnet_articulation_mpm0417.py`
  - Shared backend used by the held-out / repair flows and existing wrapper
    scripts.

If you are starting a new workflow, prefer the unified entry points first and
only call the `try1_*` scripts directly when you specifically need their
low-level behavior.

## Wrapper Scripts

Current shell wrappers that still map to active code:

- `run0417.sh`
  - Current smoke test for the train generator.
- `run_add_count01_count02_150.sh`
  - Extra benchmark pool generation for `count_01` and `count_02`.
- `run_existing_single_ids_case900_901.sh`
  - Repair / backfill `case900` and `case901` for existing ids.
- `run_stage1_count01_benchmark.sh`
  - Build the stage1 count-01 held-out benchmark.
- `run_try1_physxnet_all_objects_all_cases0419.sh`
  - Full rigid-only all-object generation using the articulation backend.

## Notes

- The old `generate_rigid_dataset.py` entry point is no longer part of this
  directory.
- The old early rigid generator script has been retired; current flows are
  centered on `generate_physxnet_train_rigid_multi.py` and
  `generate_rigid_benchmark.py`.
- If you need to understand sample schema, inspect actual generated
  `metadata.json`, `scene_input.json`, and the visualization tools above rather
  than relying on older notes.
