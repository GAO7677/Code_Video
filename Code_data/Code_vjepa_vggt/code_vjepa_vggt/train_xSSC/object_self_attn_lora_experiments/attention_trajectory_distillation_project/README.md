# Frozen Motion Probe attention-trajectory distillation

## Entry point

`train_xssc_object_self_attn_lora_frozen_motion_probe.py` is independent of the
existing xSSC feature-loss entry.  It keeps the original Wan flow-matching loss
and adds:

```text
L = L_flow
  + motion_probe_heatmap_weight
      * sum_h w_h KL(A_h^teacher || A_h^student)
  + motion_probe_trajectory_weight
      * Huber(trajectory(A_PCK^student), trajectory(A_PCK^teacher))

w_h = PCK_h^gamma / sum_j PCK_j^gamma, gamma = 30
A_PCK = sum_h w_h A_h
```

The loss operates on all 100 physical head distributions before aggregation.
The PCK score source is the `selection_source` recorded by the Top100 head
configuration, using its `pck32` field and a configurable power sharpening
exponent (`gamma=30` by default). The loader verifies the ranking step,
Top100 identity, missing/duplicate heads, and collector order before registering
the normalized weights. Equal-head aggregation and the legacy aggregate
`KL(Student || Teacher)` remain diagnostics only; neither is optimized.

Both the main Student base and the measurement probe load the official
`/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B` DiT shards.  Historical OpenVid LoRA,
preset LoRA, and full-model training arguments are rejected.  The existing
`full_sa` or `t_head` code still injects a new zero-initialized trainable Student
adapter.

## Data contract for the fixed GT query

Every raw sample must contain at least one usable GT query source, directly or
under `metadata` (the priority is token indices, mask, then points):

1. `object_query_token_indices`: exact flattened Wan rows from one fixed latent
   frame; or
2. `object_query_mask`: `[H,W]`, `[T,H,W]`, or `[O,T,H,W]`; or
3. `object_query_points`: normalized/pixel `[P,2]`, `[T,P,2]`, or `[O,T,P,2]`.

The same resolved rows are reused by Teacher and Student.  More strictly, each
selected layer-head's Q vectors are extracted once from the stop-gradient GT
Teacher pass, detached, and reused verbatim when the Student map is computed
against the Student pass's K field.  Therefore the Student neither chooses the
query location nor supplies the Q representation used by the loss.  Missing
metadata is a hard error; the Student prediction is never used to create its own
query.  The existing `xssc_replay_mix` no-GT-box datasets do not currently emit
these keys, so they must be augmented from simulator masks/tracks (or a cache)
before this entry is launched.  OpenVid should not be mixed in unless equivalent
frozen query metadata is supplied.

The default is the established fixed F04/latent-1 query.  Full masks use the
same any-intersection membership rule as the prior overlay implementation: a
Wan cell is selected if any object-mask pixel falls inside it.  Query-row sums
from the legacy view and means used here differ only by a constant and become
identical after heatmap probability normalization.

## Probe corruption and timestep

The shared fixed noise is sampled once per training example:

```text
x_probe = (1 - probe_noise_level) * x0 + probe_noise_level * epsilon_p
```

`epsilon_p`, `probe_noise_level`, and `probe_timestep` are identical between the
GT and Student branches.  `probe_timestep` is the DiT time-conditioning value;
it is deliberately not randomized.  The logged scheduler sigma makes any
intentional mismatch between timestep conditioning and corruption strength
auditable.

## Gradient flow

```text
attention/trajectory loss
  -> explicit checkpoint map output (detached Teacher GT-Q x Student K)
  -> frozen probe input
  -> x0_pred
  -> v_pred
  -> first Student DiT
```

The probe is kept outside the parent module registry, has
`requires_grad=False`, stays in eval mode, and is absent from the optimizer,
DDP state, and Student checkpoints.  Block checkpoints explicitly return both
the block state and selected Q/K maps; Q/K capture is not a non-reproducible
Python side effect.

## Difference from the old Scheme B

| Item | Old Scheme B | Frozen Motion Probe entry |
|---|---|---|
| Student measurement pass | Trainable Student DiT | Separate pretrained DiT, fully frozen |
| Teacher/Student instrument | Teacher/EMA vs Student parameters | Exactly the same frozen parameters |
| Query used by Student map | Could move with Student Q representation | Detached GT Teacher Q at fixed GT rows |
| Permitted loss shortcut | Update probe Q/K weights | Probe weights cannot update; loss must change `x0_pred` |
| Main optimization target | Flow plus attention auxiliary | Same flow loss plus KL and trajectory Huber |

## Command pattern

Use the same dataset/training arguments as the existing experiment entry, but
do **not** pass `--lora_base_model`, `--lora_checkpoint`, `--preset_lora_path`,
or `--trainable_models`.  The new required arguments are:

```bash
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/attention_trajectory_distillation_project/train_xssc_object_self_attn_lora_frozen_motion_probe.py \
  <existing dataset, optimizer, checkpoint and Wan arguments> \
  --disable_object_branch \
  --self_attn_adaptation_mode t_head \
  --head_selection_config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/configs/physiciq67_pck32_s039_latest3350_top100_heads.json \
  --head_selection_subset_id T_physiciq67_pck32_s039_latest3350_top100 \
  --head_selection_expected_role T \
  --head_selection_feature_subtype physiciq67_pck32_s039_latest3350 \
  --head_selection_expected_num_heads 100 \
  --probe_timestep 500 \
  --probe_noise_level 0.5 \
  --motion_probe_pck_weight_power 30 \
  --motion_probe_query_latent_frame 1 \
  --motion_probe_heatmap_weight 0.1 \
  --motion_probe_trajectory_weight 0.1
```

The entry currently enforces per-GPU batch size 1 so every loss has one
unambiguous fixed object query set.

## Formal GT latent-mask CE training entry

`train_xssc_object_self_attn_lora_frozen_motion_probe_latent_mask.py` is the
formal training entry for the GroundingDINO + SAM2 GT-role latent-mask scheme.
It reuses the original flow-matching pass and the same frozen Wan2.2 Motion
Probe, but it does not optimize the old Teacher/Student attention KL or the
trajectory Huber term:

```text
L = L_flow + motion_probe_latent_mask_weight * CE(Y_mask, A_student)

A_student = sum_h normalize(PCK_h^gamma) * A_h(Teacher post-RoPE Q, Student K)
```

`Y_mask` is the normalized soft occupancy obtained by area-pooling the tracked
pixel mask to the Wan latent token grid. Pixel anchors `F00,F04,...,F48` map to
`L00,L01,...,L12`; the default source is `F04/L01`, and the loss is averaged
over valid future frames `L02-L12`. All occupied source tokens contribute Query
vectors, weighted by their fractional source-mask occupancy. Teacher attention
is computed only as a stop-gradient diagnostic; it is not a target in this
objective.

The per-sample offline cache contract is:

```text
<mask_cache_root>/cases/<metadata.sample_key>/object_masks.npz
  masks_othw:   [objects, mask_frames, pixel_height, pixel_width], values in [0,1]
  frame_indices: optional [mask_frames] source-video frame indices
```

Alternatively, the same tensors may be supplied directly as
`raw_sample.object_tracking_masks` and
`raw_sample.object_tracking_masks_frame_indices`. A non-identity sampled video
must have explicit `frame_indices`, or a full source-timeline mask whose length
equals `metadata.source_frame_count`; ambiguous alignment is a hard error.
Every forward also expands latent support back to pixel space and requires zero
missed GT foreground pixels. Reverse recall, precision, IoU, and missed-pixel
count are logged.

Use the normal dataset, optimizer, and checkpoint arguments with this entry:

```bash
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/attention_trajectory_distillation_project/train_xssc_object_self_attn_lora_frozen_motion_probe_latent_mask.py \
  <existing dataset, optimizer, checkpoint and Wan arguments> \
  --output_path /data/gaoya/agent-data/checkpoints/frozen_motion_probe_latent_mask \
  --disable_object_branch \
  --train_batch_size 1 \
  --self_attn_adaptation_mode t_head \
  --head_selection_config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/configs/physiciq67_pck32_s039_latest3350_top100_heads.json \
  --head_selection_subset_id T_physiciq67_pck32_s039_latest3350_top100 \
  --head_selection_expected_role T \
  --head_selection_feature_subtype physiciq67_pck32_s039_latest3350 \
  --head_selection_expected_num_heads 100 \
  --probe_timestep 500 \
  --probe_noise_level 0.5 \
  --motion_probe_pck_weight_power 30 \
  --motion_probe_query_latent_frame 1 \
  --motion_probe_query_object_index 0 \
  --motion_probe_latent_mask_weight 0.01 \
  --motion_probe_mask_cache_root /data/gaoya/agent-data/cache/uniform_multiobject_correspondence_diagnostics
```

Configuration uses the existing CLI rather than a separate YAML/JSON loader.
The default cache currently contains only the three diagnostic cases F1-F3;
the trainer intentionally fails on a missing sample instead of silently
dropping the mask loss. Precompute GroundingDINO + SAM2 tracked masks for every
sample selected by the formal training dataset before launching a full run.

## Training-case diagnostic and noise sweep

The completed report retains the original `training t=500 / Probe=0.5`
diagnostic and appends five controlled training-noise stages with two lower
Probe corruptions:

```text
training timestep: 100, 300, 500, 700, 900
Probe (noise level, timestep): (0.1, 100), (0.2, 200)
```

Within each case, all training stages share one `epsilon_train`; both Probe
levels and every Teacher/Student pair share one `epsilon_p`. Run the complete
PCK-weighted forward and equal-vs-PCK render pipeline in the foreground with:

```bash
GPU_ID=0 PCK_WEIGHT_POWER=30 ./run_training_case_pck_weighted_gpu0.sh
```

The report is available at:

```text
/data/gaoya/agent-data/outputs/frozen_motion_probe_training_diagnostics/index.html
```

Completed sharpened results include side-by-side equal, linear PCK (`gamma=1`),
and active PCK (`gamma=30`) videos and trajectories. Their six-row timeline is
ordered as Teacher/Student for all three weighting modes from `L00/F00` through
`L12/F48`, under one shared color scale. The interrupted render completed all
three original Probe results and six F1 sweep combinations (`t=100/300/500` x
Probe `0.1/0.2`). Remaining sweep sections retain their existing equal/linear
media while reporting the completed `gamma=30` loss and gradient values.
Regenerate only completed timelines and HTML without loading the DiT or VAE:

```bash
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/gaoya/Code_Video/DiffTrack-main:/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main:/home/gaoya/Grounded-SAM-2-main \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
run_training_case_diagnostics.py refresh-report --device cpu --pck-weight-power 30
```

## Conditional spatial correspondence diagnostic

`run_pybullet_correspondence_diagnostics.py` evaluates the direct Main Student
Q/K correspondence proposal on the same three PyBullet train cases. This first
version intentionally supervises conditional spatial correspondence within each
target frame; it does not claim to reproduce Wan's full-sequence attention
distribution. Eight points are sampled inside the frozen F04 SAM2 identity mask
and tracked bidirectionally as CoTracker pseudo-GT. Pixel coordinates use a
token-cell-center mapping, and the `L01/F04` source Query is sampled bilinearly.
Future latent frames use Gaussian labels with `sigma=0.75` token. The Top100
probability maps are mixed with normalized PCK32 weights before CE is computed:

```text
L_corr = 0.01 * SNR/(SNR+1) * CE(Y, sum_h w_h A_h)
```

There is no high-noise hard cutoff and no coordinate Huber term in the default
objective. The report retains aggregate soft-argmax only as a PCK diagnostic.

Run all trajectory preparation, five-stage 5B forward diagnostics, and overlay
rendering in the foreground on GPU 0 with:

```bash
GPU_ID=0 ./run_pybullet_correspondence_diagnostics_gpu0.sh --overwrite
```

The generated page and large artifacts are stored under:

```text
/data/gaoya/agent-data/outputs/noise_gated_correspondence_diagnostics/
```

Serve that page in the foreground with:

```bash
/home/gaoya/miniconda3/envs/wan-cu128/bin/python -m http.server 8765 --bind 0.0.0.0 --directory /data/gaoya/agent-data/outputs/noise_gated_correspondence_diagnostics
```

This is a forward-only step-0 diagnostic, not an optimizer run. Differentiable
Q/K behavior, Gaussian boundary continuity, token-cell mapping, bilinear Query
sampling, aggregate CE gradients, and smooth gate behavior are covered by
`test_noise_gated_correspondence.py`.

## GT latent-mask correspondence diagnostic

`run_pybullet_latent_mask_correspondence_diagnostics.py` replaces point-track
supervision with each object's full per-frame supervision mask. The current
PyBullet cases do not export native simulator instance masks, so the cached
GroundingDINO + SAM2 tracked masks are pseudo-labels treated as GT-role
supervision by this diagnostic. Pixel masks at
`F00,F04,...,F48` are area-pooled from `512x896` to `13x16x28`, so each spatial
token stores the foreground fraction of its exact `32x32` pixel footprint. The
normalized target-frame occupancy is the soft region label. At `F04/L01`, every
occupied source token supplies a post-RoPE Query; per-token attention maps are
weighted by source occupancy before the Top100 PCK32 head mixture and CE:

```text
L_corr = 0.01 * mean_objects(mean_future_frames(CE(Y_mask, A_mask)))
```

The mapping audit expands each occupied token back to its pixel footprint and
requires zero missed GT pixels. This hard support is intentionally a superset
of the original contour, so its precision and IoU report token quantization;
it is not presented as a lossless inverse mask reconstruction.

Run the mapping audit, all 15 controlled 5B forwards, and report rendering in
the foreground on an idle non-4 GPU with:

```bash
GPU_ID=2 ./run_pybullet_latent_mask_correspondence_diagnostics_gpu.sh all --overwrite
```

The generated page and artifacts are stored under:

```text
/data/gaoya/agent-data/outputs/latent_mask_correspondence_diagnostics/
```

Serve the report in the foreground with:

```bash
/home/gaoya/miniconda3/envs/wan-cu128/bin/python -m http.server 8771 --bind 0.0.0.0 --directory /data/gaoya/agent-data/outputs/latent_mask_correspondence_diagnostics
```

This remains a forward-only preflight diagnostic with no optimizer step. The
area mapping, reverse-support invariant, region-attention normalization,
equal-object reduction, and Q/K gradients are covered by
`test_latent_mask_correspondence.py`.
