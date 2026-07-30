#!/usr/bin/env bash
set -euo pipefail

# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_s_remaining_metric_tails_gpu3.sh

PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
BENCH="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/bench.py"
ALLOWLIST="/data/gaoya/agent-data/outputs/wan_dit_fulltoken_head_roles_50seeds/input_lists/test5_unique20.txt"
PREWARM="/data/gaoya/agent-data/outputs/wan_dit_head_role_dose_control/metric_prewarm_current/metrics"
PHYSICS="/data/gaoya/agent-data/outputs/wan_dit_s_motion_analysis/physics_metric_snapshots/20260729T173323Z"
PREWARM_RESULT="/data/gaoya/agent-data/outputs/wan_dit_head_role_dose_control/pilot/generation/physrvg/seed-003278/S_k08_r02_depthmatch_steps00_10/results/test5_unique20/physRVG_steps40_512x896_08_49f"
PHYSICS_RESULT="/data/gaoya/agent-data/outputs/wan_dit_head_role_depth_strata/s_only/generation/wan_lora/seed-003278/S_depth_late_B20_29_all_steps00_40/results"

run_one() {
  local run_root="$1"
  local task_id="$2"
  local metric="$3"
  local result_root="$4"
  local summary="${run_root}/task_summaries/${task_id}.json"
  local completed="${run_root}/completed_tasks.tsv"
  local lock="${run_root}/completed_tasks.lock"

  if awk -F $'\t' -v task="${task_id}" '$1 == task { found=1 } END { exit !found }' "${completed}"; then
    echo "[metric-tail] already complete: ${task_id}"
    return
  fi

  echo "[metric-tail] running: ${task_id}"
  TOKENIZERS_PARALLELISM=false \
  CUDA_VISIBLE_DEVICES=3 \
  PYTHONPATH="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/Code_data/Code_try0526" \
    "${PYTHON}" "${BENCH}" \
      --metric "${metric}" \
      --result-root "${result_root}" \
      --input-json-allowlist "${ALLOWLIST}" \
      --output-summary "${summary}"

  {
    flock 9
    if ! awk -F $'\t' -v task="${task_id}" '$1 == task { found=1 } END { exit !found }' "${completed}"; then
      printf '%s\t%s\t%s\t%s\n' \
        "${task_id}" "${metric}" "${result_root}" "tail_gpu3" >> "${completed}"
    fi
  } 9>"${lock}"
}

run_one "${PREWARM}" "gpu_common-00515" "vbench_temporal_flickering" "${PREWARM_RESULT}"
run_one "${PREWARM}" "gpu_common-00516" "vbench_motion_smoothness" "${PREWARM_RESULT}"
run_one "${PHYSICS}" "heavy-0024-cosmos_reason1" "cosmos_reason1" "${PHYSICS_RESULT}"

touch "${PREWARM}/run.complete"
touch "${PHYSICS}/run.complete"
echo "[metric-tail] all remaining tasks complete"
