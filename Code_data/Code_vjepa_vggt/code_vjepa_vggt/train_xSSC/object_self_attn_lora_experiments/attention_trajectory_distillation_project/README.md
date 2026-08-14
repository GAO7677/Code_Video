# Frozen Motion Probe attention-trajectory distillation

## Entry point

`train_xssc_object_self_attn_lora_frozen_motion_probe.py` is independent of the
existing xSSC feature-loss entry.  It keeps the original Wan flow-matching loss
and adds:

```text
L = L_flow
  + motion_probe_heatmap_weight * KL(student_attention || teacher_attention)
  + motion_probe_trajectory_weight * Huber(student_trajectory, teacher_trajectory)
```

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
  --motion_probe_query_latent_frame 1 \
  --motion_probe_heatmap_weight 0.1 \
  --motion_probe_trajectory_weight 0.1
```

The entry currently enforces per-GPU batch size 1 so every loss has one
unambiguous fixed object query set.
