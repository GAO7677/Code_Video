# V-JEPA xSSC Stage 1 Causal State Project Handoff

Last updated: **2026-08-14 16:08 UTC**  
Project root:
`/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/xssc_rsfq2_ytvis_dinov3_vitl16_256`

This document is the operational handoff for the **MOVi-C 24-frame
prefix-causal xSSC adaptation and Stage 1 state audit**. Read
[`plan.md`](plan.md) for the formal experiment contract. The most recent
training branch described here takes precedence over the older
step-50000/two-GPU example still shown in [`README.md`](README.md). The latest
run is **not currently alive**; see Section 6 before attempting to continue it.

## 1. Research question and scope

The project asks whether the object slots learned by xSSC can form a useful
strictly causal object-state space. Stage 1 answers only:

1. Are xSSC slot trajectories stable when state at time `t` cannot see frames
   after `t`?
2. Is one current object state enough to predict motion, or is object history
   of length 2 or 4 required?
3. Does future-state prediction need only the object's own history, or does it
   improve when other object slots are visible as set context?

Stage 1 does **not** implement an ODE, Euler/RK4 integration, Newtonian latent
constraints, collision-specialized dynamics, or a new query-only decoder. The
current feature decoder is used only as the causal-adaptation training
objective. It is skipped during slot-cache extraction and downstream audits.

## 2. Difference from the source training

The source MOVi-C run is noncausal and uses 10 raw frames. V-JEPA jointly
encodes all 10 frames, so an early feature can depend on later frames even if
xSSC subsequently processes features in time order.

The active branch uses 24 raw frames and repeated-prefix V-JEPA encoding:

```text
state 0: encode raw frames [0, 1]
state 1: encode raw frames [0, 1, 2, 3]
state 2: encode raw frames [0, 1, 2, 3, 4, 5]
...
state11: encode raw frames [0, ..., 23]
```

Only the last tubelet feature from each prefix is retained. V-JEPA tubelet
size is 2, so 24 raw frames produce 12 causal feature states. These are passed
sequentially through xSSC. The transition retains its trained `dt=5` rolling
window; it does not acquire access to future features.

The active branch is a **model-only fork** from step 25000. It loads xSSC model
weights, reloads the frozen external V-JEPA weights independently, and starts a
fresh Adam optimizer and a fresh warmup/cosine phase. It does not inherit the
source Adam moments, scheduler state, sampler position, or RNG state.

## 3. Fixed representation and data contract

- Dataset: local official MOVi-C 1.0.0.
- Dataset root: `/data/gaoya/dataset/kubric-movi/movi-c`.
- Train/validation/test sizes: 9737 / 250 / 999 videos.
- Raw frames per example: 24 at 12 FPS.
- V-JEPA tubelet size: 2.
- Causal xSSC states: 12 at 6 Hz.
- State labels: raw frames `[1, 3, 5, ..., 23]`, i.e. the second frame of
  each tubelet.
- Slots: 11.
- Slot dimension: 512.
- Static channels: `[0:384]`.
- Dynamic channels: `[384:512]`.
- Cached trajectory shape: `[12, 11, 512]`.
- Identity-calibration prefix: states 0--3.
- Longest future rollout: states 4--11, i.e. an 8-state stress test.

MOVi-C examples contain exactly 24 raw frames. Do not manufacture longer
trajectories by concatenating unrelated videos or by treating overlapping
tubelets as additional physical time points.

## 4. Most recent training lineage

### Source checkpoint

```text
/data/gaoya/agent-data/checkpoints/
  xssc_vjepa2_1_video_noncausal_movi_c_10f_transfer16000_clip2_steps50000/
  rsfq2_c-movi_c-vjepa2_1_vitl16_256-video-10f-slot512-transfer16000-clip2/
  42/step-025000.pth
```

Source metadata was checked before launch:

```text
format: xssc_model_checkpoint_v1
variant: vjepa2_1_vitl16_video_256_movi_c_10f_slot512_transfer16000_clip2_noncausal
optimizer_step: 25000
world_size: 2
effective_global_batch_size: 384
```

### Most recent variant

```text
vjepa2_1_vitl16_video_256_movi_c_24f_slot512_prefix_causal_from25000_gpu0
```

Configuration:

```text
start_step: 25000
total_step/max_step: 35000
causal-adaptation optimizer steps: 10000
physical GPU: 0
micro-batch: 32
gradient accumulation: 12
effective global batch: 384
precision: bfloat16
gradient clip norm: 2.0
base/peak LR: 5e-5
warmup: 500 steps, step 25000 -> 25500
post-warmup schedule: cosine to 5e-8 at step 35000
validation interval: 500 steps
checkpoint interval: 1000 steps
validation subset: all 250 official validation videos
```

The effective batch remains equal to the source run:

```text
32 samples/GPU x 1 GPU x 12 accumulation = 384
```

## 5. Code entry points

Most recent training config:

```text
upstream/config-randsfq/
  rsfq2_c-movi_c-vjepa2_1_vitl16_256-video-24f-slot512-
  prefix-causal-from25000-gpu0.py
```

Generic 24-frame causal base config:

```text
upstream/config-randsfq/
  rsfq2_c-movi_c-vjepa2_1_vitl16_256-video-24f-slot512-
  prefix-causal-stage1.py
```

Most recent launcher:

```text
run_train_stage1_movic_24f_prefix_causal_from25000_gpu0.sh
```

GPU wait/capacity-probe wrapper:

```text
stage1_causal_state_probe/wait_probe_and_train_gpu0.sh
```

Capacity smoke test:

```text
stage1_causal_state_probe/probe_training_capacity.py
```

The capacity probe tested a real forward, backward, gradient clip, and Adam
step. Batch 32 used 39.54 GiB in the probe and was selected over 24/16. The
full trainer later reached 46.03 GiB PyTorch reserved memory because DDP,
callbacks, metrics, workers, and persistent optimizer/runtime allocations add
overhead not present in the isolated probe.

## 6. Current process status

There is **no live causal-adaptation process** at this handoff. The previous
tmux session was named:

```text
xssc_stage1_causal_from25k_gpu0
```

At 2026-08-14 16:08 UTC, `tmux ls` no longer listed that session, no matching
`torchrun` or trainer process existed, and physical GPU 0 was idle. Therefore,
do not use the former attach command as evidence that training is running. The
last completed optimizer record is step 25708. The process stopped while
working on the next accumulation window, before reaching the first scheduled
causal checkpoint at step 26000.

GPU inspection command:

```bash
nvidia-smi --id=0 --query-gpu=index,memory.used,memory.free,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader
```

Do not use GPU 4 for any part of this project. Do not stop unrelated GPU
processes, and do not use broad commands such as `killpy`.

### Last recorded snapshot

The final complete record, written at 2026-08-14 03:40:37 UTC, was:

```text
optimizer step: 25708 / 35000
causal phase progress: 708 / 10000 = 7.08%
epoch field: 28
train loss: 0.7945532799
gradient norm before clipping: 1.9667290449
clip threshold: 2.0; this step was not clipped
learning rate: 4.9941507642e-5
PyTorch peak reserved memory: 46.025390625 GiB
```

No NaN, Inf, CUDA OOM, non-finite gradient, or trainer traceback was found in
`train.log`. The log ends in the middle of a normal progress bar after step
25708, so the termination cause is **unknown**; absence of a traceback is not
proof of a clean shutdown. Local W&B debug logs include abandoned-handle
tracebacks associated with the terminated process. Do not claim that the W&B
run completed successfully without checking its remote state.

### W&B

Project:

```text
xssc_stage1_causal_state_from25000
```

Most recent run:

```text
https://wandb.ai/875222004-gy/xssc_stage1_causal_state_from25000/runs/m54vf3jk
```

Run ID: `m54vf3jk`.

An earlier capacity-start attempt created run `q2u9n54k` with batch 16. It was
stopped before completing an optimizer step and produced no model checkpoint.
Do not confuse it with the step-25708 run.

## 7. Outputs and logs

Most recent checkpoint/run directory:

```text
/data/gaoya/agent-data/checkpoints/xssc_stage1_causal_state_from25000_gpu0/
  rsfq2_c-movi_c-vjepa2_1_vitl16_256-video-24f-slot512-
  prefix-causal-from25000-gpu0/42/
```

Important files:

```text
checkpoint_load_report.json     strict source-load audit
step_metrics.jsonl              one JSON row per optimizer step
val_subset.json                 fixed 250-video validation selection
wandb_run.json                  W&B identity and URL
wandb/latest-run/logs/          local W&B service logs
```

External operational logs:

```text
/data/gaoya/agent-data/outputs/xssc_stage1_causal_state_from25000_gpu0/logs/
  wait_gpu0.log
  capacity_probe.log
  selected_capacity.env
  train.log
```

Monitor the latest optimizer metrics:

```bash
tail -20 /data/gaoya/agent-data/checkpoints/xssc_stage1_causal_state_from25000_gpu0/rsfq2_c-movi_c-vjepa2_1_vitl16_256-video-24f-slot512-prefix-causal-from25000-gpu0/42/step_metrics.jsonl
```

Extract completed validation summaries:

```bash
/home/gaoya/miniconda3/envs/wan-cu128/bin/python - <<'PY'
from pathlib import Path
import re

log = Path(
    "/data/gaoya/agent-data/outputs/"
    "xssc_stage1_causal_state_from25000_gpu0/logs/train.log"
)
text = log.read_text(errors="replace").replace("\r", "\n")
for line in text.splitlines():
    line = line.strip()
    if re.match(r"^\d+-val\s+\{", line):
        print(line)
PY
```

## 8. Current metrics interpretation

The only training loss is reconstruction MSE from xSSC decoder output to the
detached frozen V-JEPA feature. GT segmentation metrics are diagnostics and do
not contribute gradients.

The user correctly noticed a temporary loss increase near the end of warmup.
The complete pattern at handoff is:

| Optimizer steps | Mean train loss | Slope per 100 steps | Mean grad norm | Clip rate |
|---|---:|---:|---:|---:|
| 25001--25100 | 0.817151 | -0.020171 | 0.519 | 0% |
| 25101--25200 | 0.806971 | -0.001548 | 0.933 | 2% |
| 25201--25300 | 0.806772 | -0.001371 | 1.496 | 9% |
| 25301--25400 | 0.805968 | +0.003859 | 1.788 | 23% |
| 25401--25500 | 0.806598 | +0.005015 | 2.002 | 44% |
| 25501--25600 | 0.804314 | -0.004745 | 1.817 | 25% |
| 25601--25700 | 0.800372 | -0.005506 | 1.652 | 12% |
| 25701--25708 | 0.796708 | -0.027338 | 1.879 | 25% |

Interpretation:

- Warmup raised LR from 0 to `5e-5` over steps 25000--25500.
- The final warmup segment caused a genuine but small loss rebound and more
  frequent gradient clipping.
- After warmup, loss resumed decreasing and clip frequency fell. The recorded
  metrics show no evidence of numerical divergence through step 25708, but the
  unexplained process termination is an operational failure and must be kept
  separate from the loss interpretation.
- Do not compare the absolute 24-frame causal train MSE directly with the
  10-frame noncausal source MSE; sequence length, causal encoder distribution,
  augmentation, and aggregation differ.

First validation at step 25500:

```text
val/recon  = 0.1495001465
val/ari    = 0.4427925348
val/ari_fg = 0.5593671203
val/mbo    = 0.1960007846
val/miou   = 0.1807569861
```

This is the first causal-adapted validation point, so it is a baseline rather
than evidence of improvement. Step 26000 would have been the next meaningful
comparison, but the run stopped before reaching it.

## 9. Immediate risks and missing checkpoint

### GPU memory

During the last live snapshot:

```text
PyTorch peak reserved: 46.03 GiB
nvidia-smi process/GPU use: approximately 48.15 GiB
reported free memory: approximately 0.38 GiB
```

This was much tighter than the original capacity-probe safety target. The run
nevertheless completed 708 fixed-shape optimizer steps without a logged OOM.
Do not increase batch size. For a new branch, batch 24 with accumulation 16
preserves effective batch 384 and provides a safer memory margin; record this
as a configuration change rather than silently treating it as the same run.

### Checkpoint availability

The first causal checkpoint was configured for step 26000, but the process
stopped at step 25708. The run directory contains no `step-026000.pth` and no
`resume-latest.pth`. Consequently, the 708 causal optimizer steps cannot be
resumed exactly: model weights, Adam moments, scheduler state, sampler state,
and RNG state after step 25000 were never checkpointed. The metrics remain
useful as diagnostics, but they are not a recoverable model artifact.

On the next successful run, verify all of the following at step 26000:

1. `step-026000.pth` and its metadata exist.
2. `resume-latest.pth` exists and references step 26000.
3. A second `26000-val {...}` summary exists.
4. W&B contains the step-26000 train and val records.
5. ARI/ARI-FG/mBO/mIoU have not jointly collapsed relative to step 25500.

### Disk

`/data` was 98% full with roughly 96 GiB free at the final verification. A new
run should fit only if other concurrent jobs do not consume the remaining
space. Do not store checkpoints, cache, or generated data under `/home/gaoya`;
keep large artifacts under
`/data/gaoya/agent-data`.

## 10. Recovery and restart procedure

An exact continuation from step 25708 is impossible because no causal
checkpoint was saved. Do **not** point `--resume-file` at a nonexistent path,
and do not describe a new run as a continuation from step 25708.

The defensible recovery is a new model-only fork from the unchanged source
`step-025000.pth`, with a new save base and a new W&B project/run so the failed
lineage is not overwritten. Because this is an expensive multi-day run and
batch 32 left very little memory margin, confirm whether to retain batch 32 or
switch to batch 24 before launching. A safe foreground template for the batch
24 option is:

```bash
cd /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/xssc_rsfq2_ytvis_dinov3_vitl16_256

SAVE_DIR=/data/gaoya/agent-data/checkpoints/xssc_stage1_causal_state_from25000_gpu0_retry1 \
WANDB_PROJECT=xssc_stage1_causal_state_from25000_retry1 \
STAGE1_BATCH_SIZE_T=24 \
WANDB_MODE=online \
bash run_train_stage1_movic_24f_prefix_causal_from25000_gpu0.sh
```

This foreground command makes failures visible. If the user explicitly
requests tmux execution, use a new session name such as
`xssc_stage1_causal_from25k_gpu0_retry1`, retain the same foreground command
inside that session, and capture the pane/log paths in this handoff. The active
config computes accumulation as `384 / batch`, so batch 24 implies accumulation
16.

## 11. Stage 1 downstream sequence after causal adaptation

Do not start ODE experiments after training. The required order is:

### 11.1 Select a causal checkpoint on validation only

Compare validation records every 500 steps. Select on the official validation
split without consulting the test split. Retain all provenance: checkpoint
hash, config, source step, temporal mode, and validation values.

### 11.2 Run the future-perturbation causality gate

Use the active branch config for provenance:

```bash
cd /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/xssc_rsfq2_ytvis_dinov3_vitl16_256

PYTHONPATH="$PWD:$PWD/upstream:/home/gaoya/Code_Video/vjepa2-main" \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
stage1_causal_state_probe/test_future_perturbation.py \
  --checkpoint /data/gaoya/agent-data/checkpoints/REPLACE_WITH_SELECTED_CAUSAL.pth \
  --config-file upstream/config-randsfq/rsfq2_c-movi_c-vjepa2_1_vitl16_256-video-24f-slot512-prefix-causal-from25000-gpu0.py \
  --device cuda:0 \
  --output /data/gaoya/agent-data/outputs/xssc_stage1_causal_state/causality_gate.json
```

Do not accept caches unless features and slots at/before each cut point remain
invariant when future raw frames are replaced, shuffled, or zeroed.

### 11.3 Cache causal slot trajectories

Run `cache_causal_slots.py` separately for train, validation, and test. Pass the
selected checkpoint and active config explicitly. Large cache artifacts belong
under:

```text
/data/gaoya/agent-data/cache/xssc_stage1_causal_state
```

The extraction path calls `extract_slot_trajectory`, skips the feature decoder,
and stores fixed prefix and boundary mappings. It must not run future per-frame
Hungarian matching for predictor inputs or targets.

### 11.4 Compute train-only slot statistics

Run `compute_slot_stats.py` only after the complete train cache is present.
Never compute normalization statistics from validation or test records.

### 11.5 Representation audit and frozen GT probes

Run:

- `audit_representation.py` for causal stability, drift, coverage, and frozen
  identity behavior;
- `train_gt_probes.py` for position, velocity, bbox, and presence probes on real
  slots;
- `evaluate_gt_probes.py` for held-out probe ceilings.

The GT probes attach physical meaning to representations. GT physical targets
must not enter latent predictor training.

### 11.6 Predictor factorial matrix

Train the fixed-capacity matrix:

```text
representation: dyn / dyn_static / full
history: H=1 / H=2 / H=4
context: individual / set
seed: 42 / 43 / 44
```

Use one-step normalized latent prediction losses only. Then evaluate open-loop
horizons 1/2/4 and the separately labelled 8-step stress test with the frozen
GT probes.

### 11.7 Analysis and visualization

Run `analyze_results.py`, `visualize_cases.py`, and `build_dashboard.py` only
after all required seeds and mappings exist. Report video-level paired
bootstrap intervals; objects and origins are not independent samples.

If the dashboard is served locally, workspace policy requires a foreground
service and the exact command must be reported. The documented command is:

```bash
python3 -m http.server 8899 --bind 127.0.0.1 --directory /data/gaoya/agent-data/outputs/xssc_stage1_causal_state/dashboard
```

## 12. Identity protocol that must not be changed silently

- Prefix-oracle assignment uses mean mask-IoU over observed states 0--3.
- Boundary-frozen assignment uses state 3 only.
- Each mapping is solved once and then frozen for every future state.
- Native recurrent slot indices are used for predictor inputs/targets.
- Per-frame future Hungarian is a labelled diagnostic ceiling only.
- Future GT matching must never reorder rollout predictions or determine their
  headline score.

## 13. Acceptance gates before Stage 2

Stage 2 continuous/discrete dynamics comparison is blocked until:

1. Future-perturbation invariance passes for causal V-JEPA features and slots.
2. Fixed-identity trajectories have usable coverage and a bounded gap from the
   per-frame oracle ceiling.
3. Frozen probes recover position/velocity/bbox/presence above trivial
   baselines.
4. At least one predictor beats latent copy on held-out videos.
5. History and context conclusions are stable across three seeds and
   video-level paired bootstrap.

Only after these gates should the project compare residual discrete dynamics,
Euler, RK4 Neural ODE, and the existing RSFQ transition.

## 14. Repository and test status

At handoff, the Stage 1 implementation files and launchers are untracked in the
current Git worktree. They have not been committed. Preserve unrelated user
changes and inspect the full worktree before staging or committing anything.

Previously completed lightweight checks included shell syntax, Python compile,
config import, expected shape/config values, and source checkpoint metadata.
The intended test command is:

```bash
cd /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/xssc_rsfq2_ytvis_dinov3_vitl16_256
/home/gaoya/miniconda3/envs/flux/bin/python -m pytest -q stage1_causal_state_probe/tests
```

On 2026-08-14, two attempts at this pytest command remained alive without
producing output for more than 30--70 seconds; those test processes were
terminated by the agent to avoid leaving duplicate CPU work. Do not record the
suite as passing. Re-run it with verbose collection or per-test timeouts and
diagnose the hang before relying on the result, for example:

```bash
/home/gaoya/miniconda3/envs/flux/bin/python -m pytest -vv -s \
  stage1_causal_state_probe/tests/test_cache_schema.py
```

Those pytest processes were terminated by exact PID and were not intentionally
used to signal the GPU trainer. The causal training process disappeared later;
the available logs do not establish a causal connection between the events.

## 15. Recommended next actions

1. Do not assume training is active or resumable; preserve the failed run
   directory and W&B run as immutable provenance.
2. Before any expensive retry, determine whether the termination came from an
   external signal or resource manager if host logs make that observable.
3. Confirm the retry memory choice with the user: batch 32 / accumulation 12
   matches the failed run but had only about 0.38 GiB free; batch 24 /
   accumulation 16 is safer and keeps effective batch 384.
4. Create a distinct retry config/variant and reduce the initial checkpoint
   interval to 250 or 500 steps so another pre-checkpoint exit cannot discard
   roughly a day of training. Do not overwrite the failed lineage.
5. At the retry's first checkpoint, verify the model, metadata,
   `resume-latest.pth`, validation record, W&B record, and an actual one-step
   resume smoke test.
6. Continue monitoring 50-step train-loss mean, gradient clip frequency, GPU
   memory, disk free space, and W&B upload status. Do not interpret a single
   batch loss or a single validation point as a representation conclusion.
7. After several validation points, select the causal checkpoint on validation
   only and run the future-perturbation gate before any cache or predictor work.
