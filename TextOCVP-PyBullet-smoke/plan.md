# SAVi Collapse Diagnosis and Optimization Plan

## Objective

Identify why the current Pixel SAVi learns a mean-color reconstruction and
mostly single-slot masks, then introduce one change at a time. Preserve every
checkpoint and use the same fixed Kubric-9600 train/validation indices.

## Shared Protocol

- Dataset: `/data/gaoya/agent-data/datasets/savi_indices_kubric9600`
- Frames: 10 consecutive frames, stride 1
- Slots: 8, slot dimension 256
- Effective batch size: 64
- Precision: BF16 DDP on GPUs 5 and 6
- Seed: 14
- Validation and checkpoint interval: 500 optimizer steps
- Diagnostic checkpoints: every 500 steps through step 4000
- Validation set: the fixed 150-sample monitor split
- Reconstruction audit: fixed seed-42 Kubric val-10 plus the four PhysicIQ cases

Success requires all of the following, not only lower global MSE:

- Main foreground objects are visible in Pixel reconstruction.
- Multiple slots have coherent, temporally stable spatial support.
- Dynamic/object-region error is no longer hidden by background area.
- Instance-mask loss improves materially from the initialization baseline.
- No persistent dominant slot above 0.70 usage and no growing dead-slot count.

## Phase 0: Preserve and Stop the Current Baseline

1. Keep all checkpoints from the current high-resolution plus mask-loss runs.
2. Stop only these sessions:
   - `savi_pixel_kubric9600_maskloss_20260717T045237Z` on GPUs 0/1
   - `savi_vjepa_kubric9600_maskloss_20260717T045237Z` on GPUs 2/3
3. Verify GPUs 0-3 are released and no orphan trainer remains.

Existing baseline for comparison:

- Pixel: 216x384, mask weight 1.0, mask warmup 500
- Result: mean-color reconstruction, near-single-slot masks
- Step-500 validation: reconstruction MSE 0.01367, mask loss 0.28207

## Phase 1: Isolate Resolution and Mask-Loss Effects

Run these controls sequentially on both GPUs 5/6.

### Experiment 1A: Low-Resolution Pure RGB Baseline

- Resolution: 64x64
- Loss: global RGB MSE only
- Mask supervision: disabled
- Warmup: 2000 steps
- Maximum steps: 4000
- Per-GPU batch: 32, accumulation: 1 (maximum at effective batch 64)

Purpose: determine whether SAVi can learn non-collapsed decomposition on this
dataset at the original architecture's spatial scale.

### Experiment 1B: High-Resolution Pure RGB Control

- Resolution: 216x384
- Loss: global RGB MSE only
- Mask supervision: disabled
- Warmup: 2000 steps
- Maximum steps: 4000
- Per-GPU batch: 4, accumulation: 8

Purpose: isolate the effect of high resolution by comparing against 1A, and
isolate the added mask loss by comparing against the existing baseline.

Decision table:

| 1A low-res | 1B high-res | Interpretation |
|---|---|---|
| succeeds | succeeds | Current mask-loss design is the primary cause. |
| succeeds | collapses | High-resolution token grid/decoder is the primary cause. |
| collapses | collapses | Dataset imbalance and global RGB MSE shortcut are primary. |
| collapses | succeeds | Investigate preprocessing/config mismatch before proceeding. |

Do not start Phase 2 until step-4000 reconstruction audits are available. The
first 2000 steps are learning-rate warmup; steps 2001-4000 test post-warmup
optimization and avoid drawing a conclusion from a warmup-only checkpoint.

## Phase 2: Fix the Reconstruction Objective

Use the best non-collapsed Phase-1 resolution.

1. Add mutually exclusive GT regions: dynamic, static geometry, background.
2. Compute a mean inside each region before combining losses.
3. Use:

   `L_region = 0.50 L_dynamic + 0.25 L_static + 0.25 L_background`

   `L_rgb = 0.50 L_global + 0.50 L_region`

4. Compare against the corresponding Phase-1 pure-MSE run for 2000 steps.

Gate: dynamic objects must become visible without a material regression in
background reconstruction or temporal consistency.

## Phase 3: Reintroduce Mask Supervision Conservatively

Starting from the Phase-2 objective:

- Mask loss maximum weight: 0.02
- Mask warmup: 2000 steps
- Union weight: 0.10
- Instance weight: 0.20
- Static weight: 0.02
- Background weight: 0.01
- Unused-slot weight: 0.01

Gate: scaled mask loss should remain approximately 20-30% of total loss, and
instance loss/IoU must improve. Do not increase the global mask weight when the
instance metric is flat.

## Phase 4: Restore High Resolution with a Latent Grid

Only after a lower-resolution objective succeeds:

1. Keep 216x384 input/output for visualization.
2. Downsample encoder features before Slot Attention to 27x48 or 54x96.
3. Decode from a low-resolution spatial-broadcast grid and progressively
   upsample to 216x384.
4. Compare against Phase 1B at equal effective batch and optimizer steps.

## Phase 5: Repair V-JEPA Feature Reconstruction

Run separately from Pixel experiments:

1. Layer-normalize frozen target and predicted features.
2. Replace raw MSE with `0.5 SmoothL1 + 0.5 cosine loss`.
3. If feature error remains spatially uniform, use a fixed PCA/whitening target
   with 256 or 384 dimensions before increasing decoder capacity.
4. Add region-balanced feature loss only after normalized reconstruction works.

Monitor feature MSE, cosine loss, spatial heatmap CV, slot usage min/max, mask
entropy, and per-instance segmentation quality.

## Execution State

- [x] Existing Pixel/V-JEPA reconstruction audit completed.
- [x] Phase 0 old sessions stopped and GPUs 0-3 released.
- [x] Phase 1A launched on GPUs 5/6.
- [ ] Phase 1A step-4000 audit completed.
- [ ] Phase 1B launched on GPUs 5/6.
- [ ] Phase 1B step-4000 audit completed.
- [ ] Phase-1 decision recorded before implementing Phase 2.
