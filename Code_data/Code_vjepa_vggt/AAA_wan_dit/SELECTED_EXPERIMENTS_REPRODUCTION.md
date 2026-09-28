# Selected Wan DiT Experiments: Reproduction and Cleanup Map

This document records the five output groups reviewed on 2026-09-28. It is
kept next to the Wan DiT capture and analysis code; large generated artifacts
remain under `/data/gaoya/agent-data/outputs`.

The historical outputs are **auditable** through their manifests, JSON/CSV
summaries and logs. They are replayable only when the input JSONs, checkpoints,
auxiliary caches and the source repositories are still available. A new run
must use a new output root and must not overwrite an old result.

## Code and data roots

```text
Wan DiT capture/analysis:
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit

Object-query generation and metrics:
/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test

Object-query metric pipeline:
/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/object_query_ablation_metrics

Input JSONs and context videos:
/data/gaoya/AAA_test_video

Wan/LoRA/xSSC/PhysRVG checkpoints:
/data/gaoya/ckpt and /data/gaoya/AAA_test_video/0529,
/data/gaoya/AAA_test_video/0623

Historical outputs:
/data/gaoya/agent-data/outputs
```

Before replaying, record the exact commits and local state. The historical
launchers include unrelated local assumptions, so a command is a reproduction
template rather than a promise of bitwise equality.

```bash
export WAN_CODE=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit
export DIFFTRACK_CODE=/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test
export REPRO_ROOT=/data/gaoya/agent-data/outputs/repro_runs

git -C /home/gaoya/Code_Video/Code_data rev-parse HEAD
git -C /home/gaoya/Code_Video rev-parse HEAD
git -C /home/gaoya/Code_Video/Code_data status --short
git -C /home/gaoya/Code_Video status --short
mkdir -p "$REPRO_ROOT"
```

GPU 4 is forbidden by the workspace rules. The original full-token and paired
launchers mention GPU 4; copy the configuration/launcher and replace it with a
permitted GPU before starting a new run.

## Common model settings

The Wan-based captures use 30 DiT blocks, 24 heads per block, 512x896 output,
49 frames, 8 context frames, 40 denoising steps and CFG 5 unless a section
below states otherwise. The main role analyses capture one-based denoising
steps 5/15/25/35 and use the 13x16x28 latent grid (5824 key tokens).

Canonical checkpoints referenced by the historical launchers are:

```text
Wan base:
/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B

Wan physical-state LoRA:
/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500

xSSC context-slot checkpoint:
/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/offcial_xSSC/train_xssc_context_slots/checkpoints/step-001500

PhysRVG DiT and LoRA:
/data/gaoya/ckpt/HappyP4nda-PhysRVG/dit/diffusion_pytorch_model.safetensors
/data/gaoya/ckpt/HappyP4nda-PhysRVG/lora/checkpoint
```

## Experiment map

| Output group | Experiment question | Main code | Current evidence boundary |
|---|---|---|---|
| `wan_dit_fulltoken_head_roles_50seeds` | Which full-token attention heads behave as spatial, trajectory, fixed-temporal, context, global or mixed heads? | `fulltoken_head_roles_test5_50seeds.json`, `run_fulltoken_head_roles_worker.py`, `finalize_fulltoken_head_roles_batch.py`, `analyze_fulltoken_head_roles_batch.py` | Strong role-consistency evidence, but the saved snapshot is not a complete 50-seed result for every model. |
| `wan_dit_paired_query_50seeds` | Does the same head keep its role when the query protocol changes from moving-object to anchor-at-t2? | `paired_query_head_stability_test5_50seeds.json`, `run_paired_query_50seeds_worker.py`, `analyze_paired_query_50seeds.py` | Useful paired comparison; completed seeds are asymmetric (LoRA 26, xSSC 3, PhysRVG 13). |
| `wan_dit_ball_query_attention` | Where are ball-query roles specialized across blocks, steps and cases? | `run_multiblock_ball_query_attention_case001460.sh`, `run_ball_query_attention_case001460.sh`, `analyze_multiblock_ball_query_heads.py`, `analyze_cross_case_head_stability.py` | Multi-block/cross-case descriptive evidence; not causal head attribution. |
| `attention_lora_object_query_frozen_trajectory_case001460` | Does freezing or restricting object-query attention along a tracked trajectory change the generated object motion? | `run_object_query_frozen_trajectory_*_seed_gpu.sh`, `build_object_query_frozen_trajectory_masks.py`, `render_object_query_frozen_trajectory_apply.py` | Generation and overlays completed for six seeds/variants, but this directory has no aggregate intervention-effect table. |
| `vbench_single_case` | What are the visual and temporal quality scores of individual generated/intervened videos? | `object_query_ablation_metrics/bench.sh` and the official VBench runner | Raw per-video records only; no aggregate report in this directory, so no global method ranking is supported. |

## Reproduction commands

### 1. Full-token head roles

The run specification is
`$WAN_CODE/fulltoken_head_roles_test5_50seeds.json`. It fixes the 20 cases in
`/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt`, 50 requested seeds,
three models, all 30 blocks and steps 5/15/25/35. The historical launcher
starts GPUs 0-5 and must not be run unchanged because it includes GPU 4.

Make a reproduction config with a new output root and permitted GPUs:

```bash
cp "$WAN_CODE/fulltoken_head_roles_test5_50seeds.json" \
   "$REPRO_ROOT/fulltoken_head_roles_repro.json"

REPRO_ROOT="$REPRO_ROOT" \
  /home/gaoya/miniconda3/envs/wan-cu128/bin/python - <<'PY'
import json, os
from pathlib import Path

src = Path(os.environ["REPRO_ROOT"]) / "fulltoken_head_roles_repro.json"
d = json.loads(src.read_text())
d["execution"]["gpus"] = [0, 1, 2, 3, 5]
d["storage"]["output_root"] = str(Path(os.environ["REPRO_ROOT"]) / "wan_dit_fulltoken_head_roles_50seeds")
src.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n")
PY
```

Run one worker per permitted GPU, then finalize and aggregate:

```bash
cd "$WAN_CODE"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
CONFIG="$REPRO_ROOT/fulltoken_head_roles_repro.json"

for GPU in 0 1 2 3 5; do
  "$PYTHON" run_fulltoken_head_roles_worker.py \
    --config "$CONFIG" --gpu "$GPU" --worker-id "gpu${GPU}" \
    > "$REPRO_ROOT/fulltoken_worker_gpu${GPU}.log" 2>&1 &
done
wait

"$PYTHON" finalize_fulltoken_head_roles_batch.py --config "$CONFIG"
"$PYTHON" analyze_fulltoken_head_roles_batch.py --config "$CONFIG"
```

The final role labels are relative-rank/descriptive labels. The historical
analysis found 100% role agreement whenever compared heads were stable in both
models, early-block spatial/global specialization, middle-block trajectory
specialization and late-block fixed-temporal/context specialization. Mixed or
unstable heads remain common, so this is not a causal proof.

### 2. Paired moving-query / anchor-t2 roles

The specification is
`$WAN_CODE/paired_query_head_stability_test5_50seeds.json`. It uses the same
20-case list and captures moving-object queries together with an anchor query at
video frame 8 / latent time 2. The historical `run_paired_query_50seeds_tmux.sh`
hard-codes LoRA on GPU 4; copy it or invoke the workers directly with permitted
GPUs:

```bash
cd "$WAN_CODE"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
CONFIG="$WAN_CODE/paired_query_head_stability_test5_50seeds.json"
ROOT="$REPRO_ROOT/wan_dit_paired_query_50seeds"

# One worker index corresponds to one model. Choose free GPUs other than 4.
"$PYTHON" run_paired_query_50seeds_worker.py --config "$CONFIG" \
  --gpu 5 --worker-index 0 --worker-count 3 \
  > "$REPRO_ROOT/paired_wan_lora.log" 2>&1 &
"$PYTHON" run_paired_query_50seeds_worker.py --config "$CONFIG" \
  --gpu 6 --worker-index 1 --worker-count 3 \
  > "$REPRO_ROOT/paired_xssc.log" 2>&1 &
"$PYTHON" run_paired_query_50seeds_worker.py --config "$CONFIG" \
  --gpu 7 --worker-index 2 --worker-count 3 \
  > "$REPRO_ROOT/paired_physrvg.log" 2>&1 &
wait

"$PYTHON" analyze_paired_query_50seeds.py \
  --config "$CONFIG" --root "$ROOT" \
  --output-dir "$ROOT/_case_audit_zh"
```

The historical B0-H10 focus head was classified predominantly as fixed-position
temporal (`P`) under both protocols. Cross-case consistency was about 0.80 for
moving queries and 0.75 for anchor-t2; within-case seed consistency was higher.
These numbers should not be reported as a completed 50-seed result unless all
three models finish the configured cohort.

### 3. Ball-query multi-block analysis

The single-case input is
`$WAN_CODE/ball_query_case001460.txt`, using case
`0613pybullet_sample_001460_w002`, four ball-query patches at output frame 8,
and attention steps 5/15/25/35. The six-block analysis used blocks
`0,5,11,17,19,29`:

```bash
cd "$WAN_CODE"
GPU_WAN=0 GPU_XSSC=1 GPU_PHYRVG=2 \
BLOCK_IDS="0 5 11 17 19 29" \
OUTPUT_ROOT="$REPRO_ROOT/wan_dit_ball_query_attention/case001460_frame08_multiblock" \
bash ./run_multiblock_ball_query_attention_case001460.sh
```

The launcher requires the corresponding baseline videos under
`/data/gaoya/agent-data/outputs/wan_dit_block17_self_attention/test5_first5/generated`.
After capture, run `analyze_multiblock_ball_query_heads.py` with the generated
`--multiblock-root`, `--block17-root`, case name and a new `--output-dir`; then
run `analyze_cross_case_head_stability.py` with the all-case root and the
model-specific query-map root. The exact command-line options are exposed by
`--help` in both scripts.

The saved analysis showed role specialization distributed across depth, with
mean cross-case consistency around 0.73. Candidate common heads are useful for
follow-up interventions, but the labels remain descriptive and not causal.

### 4. Frozen object-query trajectory intervention

The historical case list contains only
`0613pybullet_sample_001460_w002`. The six completed seeds were
`21890, 32466, 35075, 47326, 49530, 90094`. Each seed needs an original
baseline video from
`/data/gaoya/agent-data/outputs/attention_lora_seed_sweep_case001460`, plus a
160-file probe capture before masks can be built.

Use a copied wrapper with `ROOT` changed to a new directory. For the ordinary
p95/p99 trajectory masks:

```bash
cd /home/gaoya/Code_Video/DiffTrack-main
# Run one seed on a permitted GPU; repeat for the six seeds.
bash AAA_my_test/run_object_query_frozen_trajectory_seed_gpu.sh 5 47326
```

The companion wrappers add the variants represented in the old directory:

```text
run_object_query_frozen_trajectory_single_seed_gpu.sh   # p95/p99 single component
run_object_query_frozen_trajectory_dilate1_seed_gpu.sh  # removal dilation 1
run_object_query_frozen_trajectory_backtrack3_seed_gpu.sh # 3-frame backtrack
```

The intervention changes attention probability/masks; it does not modify QKV
weights. The old directory proves that generation, masks and overlays completed,
but it does not contain a common pre/post intervention metric table. To measure
the effect after regenerating videos, use the metric pipeline below.

### 5. VBench and object-query metrics

For a regenerated case/seed directory containing the baseline and ablation
videos, run the complete auditable metric pipeline on a permitted GPU:

```bash
cd /home/gaoya/Code_Video/DiffTrack-main
GPU=5 bash AAA_my_test/object_query_ablation_metrics/bench.sh \
  /path/to/CASE/seed_47326 \
  --output-base "$REPRO_ROOT/object_query_ablation_metrics"
```

Use `--dry-run` first. The default pipeline reads or computes the seven official
VBench dimensions and can additionally run CoTracker, SAM2, RAFT, DINOv2,
LPIPS and the custom trajectory/object-retention reports. The historical
`vbench_single_case` directory contains only per-video VBench JSON records and
no aggregate report; it therefore cannot support a method-wide ranking by itself.

## Recorded conclusions

- **Full-token roles:** stable heads preserve role identity across Wan+LoRA,
  xSSC and PhysRVG; model differences are mostly stable-versus-mixed changes.
  Spatial/global roles are early, trajectory roles are middle, and fixed-temporal
  or context roles are late.
- **Paired queries:** the focus B0-H10 head is overwhelmingly `P` under both
  moving and anchor-t2 protocols. Query motion and head role are not
  interchangeable concepts.
- **Ball queries:** specialization is distributed across multiple blocks and is
  moderately stable across cases; stable candidate heads are suitable for an
  intervention shortlist, not a causal claim.
- **Frozen trajectory:** the intervention generation is complete for the saved
  cohort, but an outcome metric aggregation is still required before saying that
  freezing the trajectory improves motion or physical consistency.
- **VBench:** the records measure visual/temporal quality side effects, not
  simulator correctness. Aggregate before comparing methods.

## Cleanup policy for the five reviewed output groups

Snapshot sizes on 2026-09-28 were approximately:

```text
wan_dit_fulltoken_head_roles_50seeds                 17G
attention_lora_object_query_frozen_trajectory...     11G
wan_dit_paired_query_50seeds                          9.3G
wan_dit_ball_query_attention                          4.3G
vbench_single_case                                    2.6G
```

Keep all Markdown files. Also keep JSON/JSONL manifests, selection files, CSV
summaries, text input lists, logs and HTML dashboards. The following are
reproducible generated artifacts and may be removed **only inside these five
exact output roots** after checking that no job is running:

```text
*.mp4  *.webm  *.jpg  *.jpeg  *.png  *.npz  *.npy
```

Do not delete model weights, input videos, source JSONs or the upstream
`wan22_ti2v_legacy_firstlatent_physiciq67_pck50` tree. After cleanup, the
retained manifests and this document are the replay index; a future run must
write to a new directory under `/data/gaoya/agent-data/outputs/repro_runs`.
