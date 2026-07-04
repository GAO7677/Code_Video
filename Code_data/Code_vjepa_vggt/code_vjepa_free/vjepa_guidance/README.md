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
- `batch_compute_wmreward.py`
  Batch scoring script for WMReward and optional V-JEPA surprise.
- `build_wmreward_visualization.py`
  Local HTML visualization builder for video + score inspection.
- `run_train0705_current_modes.py`
  Batch runner for the current `train0705 -> Wan2.2 v2v` preset family.
- `run_train0705_guard_ablation.py`
  Thin wrapper for the current round3 `guard_ablation` batch.

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

For the current `train0705` guidance family:

- `pilot_best`
  Alias of `ladder_s20`. This was the best `wmreward` preset in the small
  pilot, but round2/test5 did not confirm a positive mean `wmreward` delta.
- `current_candidate` / `current_balanced`
  Alias of `knee_mid_s18`. This is the current more stable trade-off preset:
  round2/test5 still does not show a positive mean `wmreward` delta, but it has
  lower cross-metric tension than `pilot_best`.

Available runner mode groups:

- `current`
  Expands to the current baseline + guided comparison family.
- `guard_ablation`
  Expands to `target_w24_old`, `target_w24_ratio_005`,
  `target_w24_guard_l1_003`.
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
