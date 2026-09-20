# Historical Experiments: Code, Outputs, and Reproduction Map

This document is the provenance map for the seven historical attention/object-query
experiments under `/data/gaoya/agent-data/outputs`.  It records the relationship
between code, input data, model weights, caches, generated media, manifests and
reports.  The generated artifacts are large and are intentionally kept outside the
code repositories.

The document is an audit/reproduction guide, not a claim that every old run is
fully hermetic.  A run is **replayable** only when its input list, model weights,
environment, launcher and output protocol are all available.  Otherwise it is
still **auditable** through its manifests and logs.

## Repository roots

| Role | Path |
|---|---|
| Wan/DiT capture and analysis code | `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit` |
| Object-query metrics and dashboards | `/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/object_query_ablation_metrics` |
| Physical/video input JSON and context videos | `/data/gaoya/AAA_test_video` |
| Wan2.2 and auxiliary checkpoints | `/data/gaoya/ckpt` |
| Runtime/query/model caches | `/data/gaoya/agent-data/cache` |
| Large generated results | `/data/gaoya/agent-data/outputs` |

The two source repositories were observed at these commits during this audit:

```text
Code_data repository: a6f88ca5032a3a1528e79da52137bc2bc62d6703
Code_Video repository: 2ecc02b85cc64e21efbcc629c5994eb9f72a9bd3
```

Both working trees contain unrelated local changes.  For a new reproducible run,
record `git status --short`, the exact commit, Python environment and all checkpoint
paths before launching a job.

## The artifact chain

Every experiment follows this logical chain:

```text
input JSON / context video / case manifest
        + model checkpoint + LoRA/xSSC/PhysRVG weights
        + query-region or attention cache
        ↓
capture or generation launcher
        ↓
per-case manifest.json + generated video + attention/QK/NPZ artifacts
        ↓
metric scripts (VBench/CoTracker/SAM2/RAFT/DINO/LPIPS or custom metrics)
        ↓
report.json / CSV / Markdown summary / dashboard assets
```

`manifest.json` is the primary provenance object.  It should be retained even if
large videos or attention arrays are later removed.  It records, depending on the
experiment, the input JSON, context video, model name, checkpoint, seed, sampling
steps, resolution, selected heads, mask mode, source manifests and output files.

## Seven historical output groups

| Output group | Generation/capture entry points | Provenance and result files | Status and dependency warning |
|---|---|---|---|
| `wan22_ti2v_legacy_firstlatent_physiciq67_pck50` | `run_*physiciq*`, attention-zero/temporal-tube generation scripts; see `AAA_wan_dit` and the visual-sample manifests | `task_summary.json`, `physiciq67_cases.json`, `aggregate/summary.json`, `runs/**/manifest.json`, `visual_samples/**/manifest.json` | **Strongly auditable.** This is upstream for the object-query redesign; its `visual_samples` directory is referenced by downstream inventory and stage manifests. Do not delete it before migrating those references. |
| `three_model_allblocks_allsteps_headwise_50case` | `capture_allblocks_wan_lora_ball_query.py`, `capture_allblocks_xssc_ball_query.py`, `capture_allblocks_physrvg_ball_query.py`; launcher template `run_allblock_ball_query_test5.sh` | `RESULTS.md`, `THREE_MODEL_COMBINED_RESULTS.md`, `three_model_combined_summary.csv`, `block_step_head_summary.csv`, `best_head_per_block_step.csv`, `top_combinations.json`, and per-case manifests | **Strongly auditable.** Reports and manifests are enough to identify input, model, query cache, seed and step. Raw videos/attention tensors can be regenerated only if the referenced code, weights and caches remain available. |
| `wan_dit_fulltoken_head_roles_50seeds` | `run_fulltoken_head_roles_50seeds_tmux.sh` → `run_fulltoken_head_roles_worker.py` → `finalize_fulltoken_head_roles_batch.py`; raw capture helper `run_fulltoken_moving_capture.sh` | `fulltoken_head_roles_test5_50seeds.json`, `state/**/seed-*.json`, `worker_logs`, `finalizer_logs`, `partial_analysis`, `claims`, `capture` | **Strongly auditable and close to replayable.** The configuration records the 20 cases, 50 seeds, 3 models, blocks 0–29 and steps 5/15/25/35. The historical launcher includes GPU 4; under the current workspace rule, remove GPU 4 before replaying. |
| `object_query_information_flow_redesign/latest3350_v1` | Metric/analysis code in `DiffTrack-main/AAA_my_test/object_query_ablation_metrics`; upstream generation is the `wan22` visual-sample pipeline | `stage0_inventory/inventory.json`, `stage4_runtime/stage4_manifest.json`, stage reports, `training_free_*/**/manifest.json`, `guidance_grid_manifest.json` | **Downstream synthesis.** Its `source_manifests`, `ablation_root`, `head_scopes` and baseline videos point back to the `wan22` output tree. Keep the stage manifests and reports even if bulky derived media is pruned. |
| `attention_lora_object_query_frozen_trajectory_case001460` | Historical wrapper recorded in logs as `./AAA_my_test/run_object_query_frozen_trajectory_single_seed_gpu.sh`; per-seed generation/selection jobs | `case_list.txt`, `logs/*.log`, `monitor_*.log`, `seeds/*/manifest.json`, original/Top30 videos and selection files | **Auditable, replayability partial.** Each case manifest records the input JSON, context video, seed, 40 steps and ranking/selection path. The exact historical wrapper is not self-contained in this output directory, so locate and version it before replay. |
| `three_model_stable_heads_alltoken_qk_50case` | Same `AAA_wan_dit` all-token QK/stable-head capture family; exact historical launcher is not recorded at the output root | `gt/cases/**/manifest.json`, `lora/cases/**/manifest.json`, `gt/logs`, `lora/logs` | **Partially auditable.** Manifests identify Wan checkpoint, case, query cache, layers, steps and seed, but there is no root-level summary or fully explicit launcher. Preserve manifests/logs before removing NPZ tensors. |
| `object_query_s09_fixed_delta_mask_multiseed_case001460` | S09 fixed-delta/mask intervention jobs; mapping is consumed by the object-query analysis code | `REVERSE_ATTENTION_TRANSPLANT.json`, `ATTENTION_PROBABILITY_REPLACEMENT_EXPERIMENT.json`, per-seed/case manifests, `mapping_csv` and donor/capture roots | **Strongly auditable.** The intervention direction, mask size, selected heads, donor/target, seed, step and mapping CSV are explicit. Raw donor/target arrays are large but are required for exact replay unless regenerated from the same capture protocol. |

## Canonical input and weight references

The paths below occur in the manifests or launchers.  Do not substitute a different
dataset split or prompt and then label the result as the historical experiment.

```text
PhysicIQ JSON list:
/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt

Full-token 20-case list:
/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt

ToyDataset case manifests:
/data/gaoya/AAA_test_video/Dataset_physV/0718ToyDataset/cases/**/case_manifest.json

Wan base checkpoint:
/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B

Wan physical-state LoRA (historical all-block script):
/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500

xSSC weights (historical all-block script):
/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/offcial_xSSC/train_xssc_context_slots/checkpoints/step-001500

PhysRVG DiT/LoRA (historical all-block script):
/data/gaoya/ckpt/HappyP4nda-PhysRVG/dit/diffusion_pytorch_model.safetensors
/data/gaoya/ckpt/HappyP4nda-PhysRVG/lora/checkpoint
```

## Reproduction procedure

### 1. Freeze the run specification

```bash
export WAN_CODE=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit
export METRIC_CODE=/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/object_query_ablation_metrics
export REPRO_ROOT=/data/gaoya/agent-data/outputs/repro_runs

git -C /home/gaoya/Code_Video/Code_data rev-parse HEAD
git -C /home/gaoya/Code_Video rev-parse HEAD
git -C /home/gaoya/Code_Video/Code_data status --short
git -C /home/gaoya/Code_Video status --short
mkdir -p "$REPRO_ROOT"
```

Write the resolved command line, `CUDA_VISIBLE_DEVICES`, environment name,
checkpoint checksums, input-list checksum and output root into a run manifest
before starting.  Never use GPU 4 in a new run in this workspace.

### 2. Reproduce an all-block pilot

The launcher below is the historical entry point.  Use a new output root so that
the old result is not overwritten:

```bash
cd "$WAN_CODE"
# The capture script requires a completed query map and matching input list.
# Point these at an existing compatible map, or build the map first; OUTPUT_ROOT
# alone does not create them.
export QUERY_MAP=/path/to/motion_query_map.json
export INPUT_LIST=/path/to/test_5_unique.txt
test -s "$QUERY_MAP" && test -s "$INPUT_LIST"

GPU=5 MODEL=wan_lora SEED=42 \
OUTPUT_ROOT="$REPRO_ROOT/allblocks_wan_lora_test5" \
QUERY_MAP="$QUERY_MAP" INPUT_LIST="$INPUT_LIST" \
bash ./run_allblock_ball_query_test5.sh
```

The script uses 512×896, 49 frames, 8 context frames, 40 denoising steps, CFG 5,
seed 42, blocks 0–29 and steps 5/15/25/35 by default.  For `xssc` or `physrvg`,
change only `MODEL` and use a free GPU other than 4.  This command is a pilot
template; the historical three-model directory layout may have been produced by
a coordinator wrapper, so compare the new per-case manifests before claiming
bitwise equivalence.

### 3. Reproduce full-token head-role capture

The configuration is:

```text
$WAN_CODE/fulltoken_head_roles_test5_50seeds.json
```

The historical tmux launcher starts workers on GPUs 0–5, including GPU 4.  Do not
run it unchanged here.  Create a reproduction copy of the JSON with:

```json
"execution": {
  "gpus": [0, 1, 2, 3, 5],
  "workers_per_gpu": 1
}
```

and set `storage.output_root` to a new directory under `$REPRO_ROOT`.  Then run
the same launcher with `CONFIG=/path/to/repro_config.json`, or invoke
`run_fulltoken_head_roles_worker.py` once per permitted GPU and finish with
`finalize_fulltoken_head_roles_batch.py`.  The config fixes the input list, 20
unique cases, 50 seeds, models `wan_lora/xssc/physrvg`, blocks 0–29 and capture
steps 5/15/25/35.

### 4. Recompute object-query metrics

The metrics entry point is `bench.sh`.  It evaluates already generated videos and
does not silently regenerate the upstream head ranking:

```bash
cd /home/gaoya/Code_Video/DiffTrack-main
GPU=5 bash AAA_my_test/object_query_ablation_metrics/bench.sh \
  /data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/attention_zero_seed47326/attention_matrix_ablations_temporal_tube_v1/0613pybullet_sample_001460_w002/seed_47326
```

For a new run, point the command at the new seed directory under `$REPRO_ROOT`.
The complete pipeline writes reports below
`/data/gaoya/agent-data/outputs/object_query_ablation_metrics`; use
`--dry-run` first and keep the generated `report.json` plus its input manifest.

### 5. Rebuild downstream analysis

For the `latest3350_v1` analysis, start from the saved
`stage0_inventory/inventory.json` and `stage4_runtime/stage4_manifest.json`.
The external experiment specification is:

```text
/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/object_query_ablation_metrics/experiment_spec_latest3350.json
```

Run only after the source-manifest paths, ranking snapshot and generated-video
completeness checks pass.  The stage scripts are in the same
`object_query_ablation_metrics` directory; each stage should write a new output
root rather than modifying `latest3350_v1` in place.

## What can be safely pruned later

Keep these first:

- root reports (`*.md`, `*.csv`, summary `*.json`);
- every per-case `manifest.json` and seed/state manifest;
- input lists, experiment specs, mapping CSVs and selection JSONs;
- worker/finalizer logs and the exact code commit/configuration;
- downstream `source_manifests` and ranking snapshots.

Only after reference checks should the following be considered for removal:

- generated MP4/JPG galleries;
- raw attention/QK/NPZ tensors;
- duplicate replay videos;
- regenerable model/query caches.

The `wan22` `visual_samples` tree is an upstream dependency of the object-query
redesign and must not be deleted merely because a downstream report exists.  A
compact provenance bundle can replace bulky media only after all manifests and
reports are rewritten to point to the bundle and a replay test succeeds.

## Known reproducibility caveats

1. Some old launchers are outside the output directories and are referenced only
   from logs.  Preserve the code repository and commit before pruning outputs.
2. The historical full-token launcher includes GPU 4; current workspace policy
   forbids GPU 4, so replay requires a config copy with GPU 4 removed.
3. `task_summary.json` in the Wan22 group is an early snapshot; the later
   `aggregate/summary.json` is the completion summary.  Use the latter for the
   final completed-run count.
4. A manifest can prove what was run without proving that every external cache is
   still available.  For a stronger replay guarantee, archive checkpoint hashes,
   input-list hashes, package/environment metadata and the resolved command line
   next to the new run root.
