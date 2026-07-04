# V-JEPA Guidance Workspace

This directory now keeps the actively used V-JEPA guidance code for the
current LoRA + Wan2.2 workflow, plus lightweight evaluation helpers.

The training-free idea is unchanged:

1. sample a single video;
2. decode a low-resolution preview from selected denoising steps;
3. compute V-JEPA past-to-future predictive surprise;
4. apply one small latent correction step with `-∇E`.

## Current Active Files

- `vjepa_surprise.py`
  Differentiable V-JEPA surprise energy wrapper. Reuses local
  `/home/gaoya/Code_Video/vjepa2-main`.
- `wan_latent_guidance.py`
  Shared latent guidance utilities: preview `x0` reconstruction, low-res
  decode, step picking, and latent correction.
- `wan_openvid_0613pybullet_lorav2v_vjepa.py`
  Main active LoRA + Wan2.2 + V-JEPA generation script.
- `run_lora_vjepa_modes.py`
  Batch runner for the 7 preset guidance modes on the current LoRA workflow.
- `experiment_presets.py`
  Shared mode definitions for baseline and guided variants, including the
  current `train0705` presets:
  - `baseline`
  - `ladder_s20` / `pilot_best`
  - `knee_mid_s18` / `current_candidate` / `current_balanced`
  - `target_w24_s15_ratio_003` / `current_local_best`
- `batch_compute_wmreward.py`
  Batch scoring script for WMReward and optional V-JEPA surprise.
- `build_wmreward_visualization.py`
  Local HTML visualization builder for video + score inspection.
- `run_train0705_current_modes.py`
  Batch runner for the current `train0705 -> Wan2.2 v2v` preset family.
- `run_train0705_guard_ablation.py`
  Thin wrapper for the current round3 `guard_ablation` batch.
- `run_train0705_ratio_cap_sweep.py`
  Thin wrapper for the round5 `ratio_cap_sweep` batch.
- `run_train0705_s15_local_sweep.py`
  Thin wrapper for the round6 local sweep around `target_w24_s15_ratio_003`.
- `score_train0705_s15_local_sweep.py`
  Scores the round6 local sweep against the overlap-5 baseline / `ladder_s20`
  / `knee_mid_s18` references and writes a markdown ranking table.

## Archived Files

Older one-off prototype scripts and experiment-suite helpers were moved to:

- `archive/2026-07-cleanup/`

That archive currently contains:

- old Wan2.2 standalone TI2V prototype;
- old Wan2.1 1.3B standalone prototype;
- manifest builders;
- smoke-suite runners;
- one-off setup scripts for `videophy_62` and earlier case batches.

These files were kept for reference, but are not part of the current active
LoRA evaluation path.

## Current Recommended Entry Points

For generation:

- `wan_openvid_0613pybullet_lorav2v_vjepa.py`
- `run_lora_vjepa_modes.py`
- `run_train0705_current_modes.py`
- `run_train0705_guard_ablation.py`
- `run_train0705_ratio_cap_sweep.py`

For the current `train0705` guidance family:

- `pilot_best`
  Alias of `ladder_s20`. This was the best `wmreward` preset in the small
  pilot, but round2/test5 did not confirm a positive mean `wmreward` delta.
- `current_candidate` / `current_balanced`
  Alias of `knee_mid_s18`. This is the current more stable trade-off preset:
  round2/test5 still does not show a positive mean `wmreward` delta, but it has
  lower cross-metric tension than `pilot_best`.
- `current_local_best`
  Alias of `target_w24_s15_ratio_003`. This is the latest round5 ratio-only
  local candidate: dense mid-band `context_anchored`, `window_size=24`,
  `latent_step_size=0.15`, and `max_correction_ratio=0.03` with decoded-video
  L1 guard disabled. It is not a solved final preset, but it is the current
  direction for round6 local refinement.
- `target_w24_ratio_005`
  Best round3 5-case guard-ablation candidate, but round4 full 17-case only
  gave near-zero `wmreward` gain and a clear `physics_iq` drop. Treat it as a
  diagnostic anchor, not a solved guidance preset.

Available runner mode groups:

- `current`
  Expands to `baseline`, `ladder_s20`, `knee_mid_s18`, and
  `target_w24_s15_ratio_003`.
- `guard_ablation`
  Expands to `target_w24_old`, `target_w24_ratio_005`,
  `target_w24_guard_l1_003`.
- `ratio_cap_sweep`
  Expands to the round5 ratio-cap probes around `target_w24_ratio_005`.
- `s15_local_sweep`
  Expands to the round6 local search around `target_w24_s15_ratio_003`.
- `all`
  Expands to every registered `train0705` preset above.

For scoring and inspection:

- `batch_compute_wmreward.py`
- `build_wmreward_visualization.py`

## Notes

- This folder intentionally stays lighter than the upstream repos and reuses
  local dependencies instead of forking them:
  - `/home/gaoya/Code_Video/vjepa2-main`
  - local Wan runtime / LoRA code under `Code_vjepa_vggt`
- If an archived script needs to come back into the active path, restore it
  explicitly rather than mixing old and new experiment layers again.
