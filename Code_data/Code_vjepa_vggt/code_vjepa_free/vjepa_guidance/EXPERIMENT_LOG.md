# V-JEPA Guidance Experiment Log

Created: 2026-07-05
Maintainer: Codex + gaoya
Scope: `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance`

This file is the project-local registry for experiments run from this
workspace. Starting now, every new experiment under `vjepa_guidance/` should be
recorded here with:

- date or date range;
- experiment name / purpose;
- main code entry points;
- model / preset / input scope;
- output directory;
- score or summary files;
- short conclusion / status.

Companion docs:

- code overview: `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/README.md`
- progress memo: `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/PROGRESS_wmreward_guidance.md`

## Update Template

Copy this block for future runs:

```md
## YYYY-MM-DD — <experiment name>

- Goal:
- Code:
  - /abs/path/to/script.py
- Inputs:
  - /abs/path/to/input_manifest_or_json.txt
- Outputs:
  - /abs/path/to/output_root
- Scores / summaries:
  - /abs/path/to/summary.json
  - /abs/path/to/scores.json
- Status:
- Conclusion:
```

## Historical Backfill

The entries below were backfilled on 2026-07-05 from existing code, output
trees, and summary files. When an exact start day was not recoverable, the date
is marked as inferred from directory names or progress docs.

## 2026-06-26 to 2026-06-30 (inferred) — Early LoRA 7-mode and case-batch prototype suite

- Goal:
  - Validate whether simple training-free V-JEPA guidance presets change Wan2.2
    + LoRA v2v outputs on `test_5` and on a larger `v2v_jsons` batch.
- Code:
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/wan_openvid_0613pybullet_lorav2v_vjepa.py`
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_lora_vjepa_modes.py`
  - historical helpers:
    `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/archive/2026-07-cleanup/run_mode_smoke_suite.py`
    and
    `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/archive/2026-07-cleanup/run_manifest_all_cases.py`
- Inputs:
  - `/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt`
  - `/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons`
- Outputs:
  - `/data/gaoya/AAA_test_video/0626vjepa_free/vjepa_guidance/test/results/lora_test5`
  - `/data/gaoya/AAA_test_video/0626vjepa_free/vjepa_guidance/test/results/v2v_jsons_full_wan22`
- Main modes seen in artifacts:
  - `baseline`
  - `g1_mid1_s001`
  - `g2_mid2_s001`
  - `g3_mid2_s002`
  - `g4_wide4_s001`
  - `g5_wide4_s002`
  - `g6_wide6_s002`
- Scores / summaries:
  - per-video sidecar JSONs under each mode directory, for example
    `/data/gaoya/AAA_test_video/0626vjepa_free/vjepa_guidance/test/results/lora_test5/wan_openvid_0613pybullet_lorav2v_step000500_test5_vjepa_baseline/0613pybullet_sample_000301_w000.json`
  - suite config:
    `/data/gaoya/AAA_test_video/0626vjepa_free/vjepa_guidance/test/results/v2v_jsons_full_wan22/suite_config.json`
  - manifest:
    `/data/gaoya/AAA_test_video/0626vjepa_free/vjepa_guidance/test/results/v2v_jsons_full_wan22/manifests/manifest.csv`
- Status:
  - Completed historical prototype suite.
- Conclusion:
  - Established the initial 7-mode comparison pattern that later evolved into
    the active `experiment_presets.py` / `run_lora_vjepa_modes.py` workflow.

## 2026-07-01 to 2026-07-02 (inferred) — One-case timestep / step-index visual diagnostics

- Goal:
  - Probe whether guidance timing matters by forcing guidance at selected
    denoising timesteps and comparing intermediate / final outputs visually.
- Code:
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/build_timestep_sweep_viewer.py`
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/compare_guidance_videos.py`
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/diag_anchored_onestep.py`
- Inputs:
  - one-case manifests saved under:
    `/data/gaoya/agent-data/outputs/vjepa_timestep_sweep/one_case.txt`
    and
    `/data/gaoya/agent-data/outputs/vjepa_stepindex_sweep/one_case.txt`
- Outputs:
  - `/data/gaoya/agent-data/outputs/vjepa_timestep_sweep`
  - `/data/gaoya/agent-data/outputs/vjepa_timestep_sweep_1460`
  - `/data/gaoya/agent-data/outputs/vjepa_stepindex_sweep`
- Scores / summaries:
  - run manifests:
    `/data/gaoya/agent-data/outputs/vjepa_timestep_sweep/run_sweep.sh`
    and
    `/data/gaoya/agent-data/outputs/vjepa_stepindex_sweep/run_sweep.sh`
  - viewer:
    `/data/gaoya/agent-data/outputs/vjepa_stepindex_sweep/index.html`
- Status:
  - Diagnostic only; mainly visual / qualitative.
- Conclusion:
  - Helped narrow the useful guidance region before the later single-case
    `phase5/phase6/phase7` probe work.

## 2026-07-02 — Probe Sweep Phase 4: wmreward coupling check on single case

- Goal:
  - Test whether small anchored-energy decreases actually move `wmreward`.
- Code:
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/probe_energy_persistence.py`
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/score_guided_videos.py`
- Inputs:
  - single case `physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed`
- Outputs:
  - `/data/gaoya/agent-data/outputs/probe_sweep/phase4`
- Scores / summaries:
  - `/data/gaoya/agent-data/outputs/probe_sweep/phase4/phase4_summary.json`
  - `/data/gaoya/agent-data/outputs/probe_sweep/phase4/wmreward_scores.json`
- Status:
  - Completed.
- Conclusion:
  - Small anchored-energy reductions mostly sat in the noise floor and did not
    translate into reliable `wmreward` gains.

## 2026-07-03 — Probe Sweep Phase 5: strong fixed-step intensity ladder

- Goal:
  - Sweep larger latent step sizes to find the point where guidance actually
    writes into the latent enough to move final metrics.
- Code:
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/probe_energy_persistence.py`
- Outputs:
  - `/data/gaoya/agent-data/outputs/probe_sweep/phase5`
- Scores / summaries:
  - `/data/gaoya/agent-data/outputs/probe_sweep/phase5/phase5_summary.json`
  - `/data/gaoya/agent-data/outputs/probe_sweep/phase5/wmreward_scores.json`
  - `/data/gaoya/agent-data/outputs/probe_sweep/phase5/phase5_multimetric_scores.json`
- Status:
  - Completed.
- Conclusion:
  - Confirmed the earlier issue was mainly weak write strength. `ladder_s20`
    became the first clearly positive single-case candidate on `wmreward`.

## 2026-07-03 — Probe Sweep Phase 6: timing × inner-k refinement

- Goal:
  - Refine around the phase5 knee using earlier/later timing, repeated inner
    correction, and backtracking variants.
- Code:
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/probe_energy_persistence.py`
- Outputs:
  - `/data/gaoya/agent-data/outputs/probe_sweep/phase6`
- Scores / summaries:
  - `/data/gaoya/agent-data/outputs/probe_sweep/phase6/phase6_summary.json`
  - `/data/gaoya/agent-data/outputs/probe_sweep/phase6/wmreward_scores.json`
  - `/data/gaoya/agent-data/outputs/probe_sweep/phase6/phase6_multimetric_scores.json`
  - regression-light score dumps:
    `/data/gaoya/agent-data/outputs/probe_sweep/phase6/phase6_multimetric_scores_regression_light.json`
    and
    `/data/gaoya/agent-data/outputs/probe_sweep/phase6/phase6_multimetric_scores_regression_light2.json`
- Status:
  - Completed.
- Conclusion:
  - Backtracking was too conservative. `knee_mid_s18`,
    `knee_early_s15`, and `knee_mid_s10_k2` emerged as strong alternatives,
    but `ladder_s20` still remained the best single-case `wmreward` result.

## 2026-07-04 — Probe Sweep Phase 7: target window size / future horizon

- Goal:
  - Hold strong guidance timing fixed and compare anchored future window sizes.
- Code:
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/probe_energy_persistence.py`
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_phase7_target_shape.py`
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/wait_for_phase7_gpus.py`
- Outputs:
  - `/data/gaoya/agent-data/outputs/probe_sweep/phase7`
- Scores / summaries:
  - `/data/gaoya/agent-data/outputs/probe_sweep/phase7/phase7_summary.json`
  - `/data/gaoya/agent-data/outputs/probe_sweep/phase7/phase7_multimetric_scores.json`
- Status:
  - Completed.
- Conclusion:
  - `target_w24` became the best single-case target-shape variant, better than
    `w16` and `w32` on final `wmreward`.

## 2026-07-04 (inferred) — Probe Sweep Phase 8 follow-up

- Goal:
  - Later-phase probe follow-up after phase7; exact memo not yet backfilled.
- Code:
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/probe_energy_persistence.py`
- Outputs:
  - `/data/gaoya/agent-data/outputs/probe_sweep/phase8`
- Scores / summaries:
  - `/data/gaoya/agent-data/outputs/probe_sweep/phase8/phase8_summary.json`
  - `/data/gaoya/agent-data/outputs/probe_sweep/phase8/phase8_multimetric_scores.json`
- Status:
  - Artifacts present; narrative summary still needs manual backfill.
- Conclusion:
  - Keep as recorded artifact set until the detailed notes are restored.

## 2026-07-03 to 2026-07-04 — Phase4 multicase pilot3 on LoRA baseline + guided presets

- Goal:
  - Compare the main guided candidates on a small multicase subset before
    moving to the heavier `train0705` rounds.
- Code:
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_phase4_multicase.py`
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/score_multicase_methods.py`
- Outputs:
  - `/data/gaoya/agent-data/outputs/vjepa_phase4_multicase`
- Main methods in artifacts:
  - `phase4_pilot3_baseline`
  - `phase4_pilot3_ladder_s20`
  - `phase4_pilot3_knee_early_s15`
  - `phase4_pilot3_knee_mid_s18`
  - `phase4_pilot3_knee_mid_s10_k2`
- Scores / summaries:
  - `/data/gaoya/agent-data/outputs/vjepa_phase4_multicase/phase4_pilot3_all_methods_scores.json`
  - `/data/gaoya/agent-data/outputs/vjepa_phase4_multicase/phase4_pilot3_baseline_vs_ladder_s20_scores.json`
  - runtime summaries under the corresponding `*_runtime/summary.json`
- Status:
  - Completed.
- Conclusion:
  - Served as the bridge from single-case tuning to subset-level A/B
    comparison, and fed the later `train0705 current modes` presets.

## 2026-07-03 — train0705 current modes, pilot3 round1

- Goal:
  - Move the best single-case presets onto the custom `train0705 stage1b`
    branch and compare them on a small 3-case pilot.
- Code:
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_train0705_current_modes.py`
- Outputs:
  - `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/pilot3_round1`
  - compare reports:
    `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/pilot3_round1_compare_partial`
    and
    `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/pilot3_round1_compare_full`
- Scores / summaries:
  - per-method summaries under each method folder, for example
    `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/pilot3_round1/train0705_pilot3_round1_ladder_s20/summary.json`
  - aggregate comparison:
    `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/pilot3_round1_compare_full/aggregate_summary.md`
    and
    `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/pilot3_round1_compare_full/aggregate_summary.json`
- Status:
  - Completed.
- Conclusion:
  - `ladder_s20` had the best mean `Δsurprise` on the 3-case pilot, while
    `knee_mid_s18` looked more balanced on cross-metrics.

## 2026-07-03 — train0705 round2: full test_5 baseline vs ladder_s20 vs knee_mid_s18

- Goal:
  - Check whether pilot3 winners hold up on the 17-case `test_5` subset.
- Code:
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_train0705_current_modes.py`
- Outputs:
  - `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round2_test5`
- Scores / summaries:
  - `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round2_test5/round2_test5_compare_summary.md`
  - `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round2_test5/round2_test5_compare_summary.json`
  - `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round2_test5/round2_test5_scores.json`
- Status:
  - Completed.
- Conclusion:
  - Neither `ladder_s20` nor `knee_mid_s18` improved mean `wmreward surprise`
    over the 17-case baseline. `knee_mid_s18` was retained as the more stable
    trade-off preset.

## 2026-07-03 — train0705 round3: guard ablation on overlap-5 subset

- Goal:
  - Compare old dense target guidance against ratio-cap and L1-guard variants.
- Code:
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_train0705_guard_ablation.py`
- Outputs:
  - `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round3_guard_ablation`
- Scores / summaries:
  - `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round3_guard_ablation/round3_guard_ablation_compare_summary.md`
  - `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round3_guard_ablation/round3_guard_ablation_compare_summary.json`
  - score dumps such as
    `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round3_guard_ablation/target_w24_ratio_005_vs_baseline5_scores.json`
- Status:
  - Completed.
- Conclusion:
  - `target_w24_old` moved `wmreward` the most but destroyed `physics_iq`.
    `target_w24_ratio_005` became the best “not obviously broken” diagnostic
    anchor for later rounds.

## 2026-07-03 to 2026-07-04 — train0705 round4/5/6 local refinement around ratio-cap and s15

- Goal:
  - Refine dense mid-band `context_anchored` guidance around
    `target_w24_ratio_005` and later around the `s15` family.
- Code:
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_train0705_ratio_cap_sweep.py`
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_train0705_s15_local_sweep.py`
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/score_train0705_s15_local_sweep.py`
- Outputs:
  - `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round4_test5_ratio_only`
  - `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round5_ratio_cap_sweep_overlap5`
  - `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round6_s15_local_sweep_overlap5`
- Scores / summaries:
  - round4:
    `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round4_test5_ratio_only/round4_test5_ratio_only_scores.json`
  - round5:
    `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round5_ratio_cap_sweep_overlap5/overlap5_target_w24_s20_ratio_010_scores.json`
  - round6:
    `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round6_s15_local_sweep_overlap5/round6_full_scores.json`
- Status:
  - Completed.
- Conclusion:
  - These rounds established the current local candidate family around
    `target_w24_s15_ratio_003` and nearby ratio-cap variants.

## 2026-07-04 — train0705 round7 expansion on full test_5

- Goal:
  - Re-test the current dense-mid local candidates on the full 17-case subset.
- Code:
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_train0705_round7_expansion.py`
- Outputs:
  - `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round7_test5_expansion`
- Scores / summaries:
  - `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round7_test5_expansion/round7_test5_scores.json`
  - `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round7_test5_expansion/round7_baseline_vs_0025_scores.json`
  - `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round7_test5_expansion/round7_baseline_vs_0035_scores.json`
- Status:
  - Completed.
- Conclusion:
  - Confirmed that the local dense-mid family remained a live direction, but
    did not yet produce a clean full-subset win.

## 2026-07-04 to 2026-07-05 — train0705 round8 step-005000 guided pass

- Goal:
  - Re-run the current guided family on the later `train0705 step-005000`
    checkpoint.
- Code:
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_train0705_current_modes.py`
- Outputs:
  - `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round8_step5000_test5_guided`
- Scores / summaries:
  - `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round8_step5000_test5_guided/train0705_round8_step5000_test5_target_w24_s15_ratio_0025/summary.json`
- Status:
  - Partial historical artifact; full comparison write-up not yet backfilled.
- Conclusion:
  - Captured the step-005000 guided branch that later fed into the broader
    model-weight A/B test.

## 2026-07-05 — Unified model-weight A/B on test_5

- Goal:
  - Run baseline vs guided comparisons across multiple model lines:
    official Wan2.2 TI2V, early LoRA, `train0705 step-002500`,
    `train0705 step-005000`, and Wan2.1 T2V 1.3B prototype.
- Code:
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_model_weight_ab_test5.py`
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/score_multicase_methods.py`
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/wan21_t2v_1_3b_batch.py`
  - supporting generators:
    `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/wanti2v.py`
    and
    `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/wan_openvid_0613pybullet_lorav2v_vjepa.py`
- Inputs:
  - `/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt`
  - deduped manifest:
    `/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/inputs/test5_unique.txt`
- Outputs:
  - `/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705`
- Scores / summaries:
  - summary placeholders / runtime summaries already present under:
    `/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/train0705_step002500/baseline/summary.json`
    `/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/train0705_step002500/guided/summary.json`
    `/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/train0705_step005000/baseline/summary.json`
    `/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/train0705_step005000/guided/summary.json`
  - reuse reports:
    `/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/reuse_reports/lora_baseline_reuse.json`
    and
    `/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/reuse_reports/official_baseline_reuse.json`
- Status:
  - In progress as of 2026-07-05.
- Current conclusion:
  - LoRA baseline has partial safe reuse from existing
    `/data/gaoya/AAA_test_video/0623/test/v2v` results.
  - Official Wan2.2 baseline did not have byte-identical historical matches and
    still requires fresh generation in this A/B tree.
