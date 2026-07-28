#!/usr/bin/env bash
# Run:
#   bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_seed851_baseline_bench_tmux.sh

set -euo pipefail

ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
BATCH_ROOT=/data/gaoya/agent-data/outputs/wan_dit_common22_test5_seed851_baseline_bench
RUN_ROOT="${BATCH_ROOT}/run_20260728_dynamic"
ABLATION_RUN=/data/gaoya/agent-data/outputs/wan_dit_common22_test5_st_phased_seed851_bench/run_20260728_dynamic
SESSION="${SESSION:-wan_seed851_baseline_bench_gpu0123456_20260728}"
WORKER="${ROOT}/run_stc_bench_dynamic_worker.sh"

if [[ "${1:-}" == "--inside" ]]; then
  exec > >(tee -a "${BATCH_ROOT}/orchestrator.log") 2>&1
  if [[ "${BASELINE_START_IMMEDIATELY:-0}" == "1" ]]; then
    echo "[baseline-orchestrator] starting immediately on GPU0-4"
  else
    echo "[baseline-orchestrator] waiting for the ablation batch"
    until [[ -f "${ABLATION_RUN}/state/all_complete" ]]; do
      sleep 60
    done
  fi

  mkdir -p \
    "${RUN_ROOT}/queues" "${RUN_ROOT}/logs" "${RUN_ROOT}/state" \
    "${RUN_ROOT}/task_summaries"
  : > "${RUN_ROOT}/queues/common.tsv"
  : > "${RUN_ROOT}/queues/videophy.tsv"
  : > "${RUN_ROOT}/queues/cosmos.tsv"
  : > "${RUN_ROOT}/completed_tasks.tsv"
  : > "${RUN_ROOT}/failed_tasks.tsv"
  for kind in common videophy cosmos; do
    printf '1\n' > "${RUN_ROOT}/queues/${kind}.cursor"
    : > "${RUN_ROOT}/queues/${kind}.lock"
  done

  common_metrics=(
    physics_iq_with_context
    physics_iq_without_context
    pmf_with_context
    pmf_without_context
    wmreward
    vbench_subject_consistency
    vbench_background_consistency
    vbench_temporal_flickering
    vbench_motion_smoothness
    vbench_dynamic_degree
    vbench_aesthetic_quality
    vbench_imaging_quality
  )
  task_index=0
  for metric in "${common_metrics[@]}"; do
    for shard in $(seq 0 6); do
      printf 'common-%04d\t%s\t%s\t7\n' \
        "${task_index}" "${metric}" "${shard}" \
        >> "${RUN_ROOT}/queues/common.tsv"
      task_index=$((task_index + 1))
    done
  done
  for shard in $(seq 0 6); do
    printf 'videophy-%04d\tvideophy2\t%s\t7\n' \
      "${shard}" "${shard}" >> "${RUN_ROOT}/queues/videophy.tsv"
    printf 'cosmos-%04d\tcosmos_reason1\t%s\t7\n' \
      "${shard}" "${shard}" >> "${RUN_ROOT}/queues/cosmos.tsv"
  done

  for gpu in 0 1 2 3 4; do
    name="common_g${gpu}_0"
    tmux new-window -t "${SESSION}" -n "${name}" \
      "bash '${WORKER}' common '${gpu}' '${name}' '${RUN_ROOT}' '${BATCH_ROOT}' 36000"
  done
  until [[ "$(find "${RUN_ROOT}/state" -maxdepth 1 -name 'common_*.worker_complete' -type f | wc -l)" -ge 5 ]]; do
    sleep 30
  done

  for gpu in 0 2 4; do
    name="videophy_g${gpu}_0"
    tmux new-window -t "${SESSION}" -n "${name}" \
      "bash '${WORKER}' videophy '${gpu}' '${name}' '${RUN_ROOT}' '${BATCH_ROOT}' 30000"
  done
  for gpu in 1 3; do
    name="cosmos_g${gpu}_0"
    tmux new-window -t "${SESSION}" -n "${name}" \
      "bash '${WORKER}' cosmos '${gpu}' '${name}' '${RUN_ROOT}' '${BATCH_ROOT}' 28000"
  done
  until [[ "$(find "${RUN_ROOT}/state" -maxdepth 1 -name 'videophy_*.worker_complete' -type f | wc -l)" -ge 3 ]] \
    && [[ "$(find "${RUN_ROOT}/state" -maxdepth 1 -name 'cosmos_*.worker_complete' -type f | wc -l)" -ge 2 ]]; do
    sleep 30
  done

  "${PYTHON}" "${ROOT}/summarize_stc_bench_metrics.py" \
    --batch-root "${BATCH_ROOT}"
  touch "${RUN_ROOT}/state/all_complete"
  echo "[baseline-orchestrator] complete"
  exec bash
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

"${PYTHON}" "${ROOT}/build_seed851_baseline_bench_batch.py" \
  --output-root "${BATCH_ROOT}"
tmux new-session -d -s "${SESSION}" -n orchestrator \
  "bash '$0' --inside"
echo "tmux session: ${SESSION}"
echo "batch root: ${BATCH_ROOT}"
