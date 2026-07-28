#!/usr/bin/env bash
# Run:
#   bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_seed851_gt49f_bench_all_gpus_tmux.sh

set -euo pipefail

ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
BATCH_ROOT=/data/gaoya/agent-data/outputs/wan_dit_common22_test5_gt49f_896x512_bench
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-${BATCH_ROOT}/run_${RUN_TAG}_gpu0123567}"
SESSION="${SESSION:-wan_seed851_gt49f_896x512_bench_gpu0123567}"
WORKER="${ROOT}/run_stc_bench_dynamic_worker.sh"
GPUS=(0 1 2 3 5 6 7)
VIDEOPHY_GPUS=(0 2 5 6)
COSMOS_GPUS=(1 3 7)
NUM_SHARDS=7

if [[ " ${GPUS[*]} " == *" 4 "* ]]; then
  echo "GPU4 is forbidden for this workspace" >&2
  exit 2
fi

count_workers() {
  find "${RUN_ROOT}/state" -maxdepth 1 \
    -name "$1" -type f | wc -l
}

if [[ "${1:-}" == "--inside" ]]; then
  exec > >(tee -a "${RUN_ROOT}/orchestrator.log") 2>&1
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
    for shard in $(seq 0 $((NUM_SHARDS - 1))); do
      printf 'common-%04d\t%s\t%s\t%s\n' \
        "${task_index}" "${metric}" "${shard}" "${NUM_SHARDS}" \
        >> "${RUN_ROOT}/queues/common.tsv"
      task_index=$((task_index + 1))
    done
  done
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    printf 'videophy-%04d\tvideophy2\t%s\t%s\n' \
      "${shard}" "${shard}" "${NUM_SHARDS}" \
      >> "${RUN_ROOT}/queues/videophy.tsv"
    printf 'cosmos-%04d\tcosmos_reason1\t%s\t%s\n' \
      "${shard}" "${shard}" "${NUM_SHARDS}" \
      >> "${RUN_ROOT}/queues/cosmos.tsv"
  done

  echo "[gt49f] phase=common workers=${#GPUS[@]} tasks=${task_index}"
  for gpu in "${GPUS[@]}"; do
    name="common_g${gpu}"
    tmux new-window -t "${SESSION}" -n "${name}" \
      "bash '${WORKER}' common '${gpu}' '${name}' '${RUN_ROOT}' '${BATCH_ROOT}' 36000"
  done
  until [[ "$(count_workers 'common_*.worker_complete')" -ge "${#GPUS[@]}" ]]; do
    claimed="$(( $(<"${RUN_ROOT}/queues/common.cursor") - 1 ))"
    completed="$(grep -c '^common-' "${RUN_ROOT}/completed_tasks.tsv" || true)"
    failed="$(grep -c '^common-' "${RUN_ROOT}/failed_tasks.tsv" || true)"
    echo "[gt49f] phase=common claimed=${claimed}/${task_index} completed=${completed} failed=${failed}"
    sleep 30
  done

  echo "[gt49f] phase=heavy videophy=${#VIDEOPHY_GPUS[@]} cosmos=${#COSMOS_GPUS[@]}"
  for gpu in "${VIDEOPHY_GPUS[@]}"; do
    name="videophy_g${gpu}"
    tmux new-window -t "${SESSION}" -n "${name}" \
      "bash '${WORKER}' videophy '${gpu}' '${name}' '${RUN_ROOT}' '${BATCH_ROOT}' 30000"
  done
  for gpu in "${COSMOS_GPUS[@]}"; do
    name="cosmos_g${gpu}"
    tmux new-window -t "${SESSION}" -n "${name}" \
      "bash '${WORKER}' cosmos '${gpu}' '${name}' '${RUN_ROOT}' '${BATCH_ROOT}' 28000"
  done
  until [[ "$(count_workers 'videophy_*.worker_complete')" -ge "${#VIDEOPHY_GPUS[@]}" ]] \
    && [[ "$(count_workers 'cosmos_*.worker_complete')" -ge "${#COSMOS_GPUS[@]}" ]]; do
    videophy_done="$(grep -c '^videophy-' "${RUN_ROOT}/completed_tasks.tsv" || true)"
    cosmos_done="$(grep -c '^cosmos-' "${RUN_ROOT}/completed_tasks.tsv" || true)"
    failed="$(wc -l < "${RUN_ROOT}/failed_tasks.tsv")"
    echo "[gt49f] phase=heavy videophy=${videophy_done}/${NUM_SHARDS} cosmos=${cosmos_done}/${NUM_SHARDS} failed=${failed}"
    sleep 30
  done

  "${PYTHON}" "${ROOT}/summarize_stc_bench_metrics.py" \
    --batch-root "${BATCH_ROOT}"
  failed="$(wc -l < "${RUN_ROOT}/failed_tasks.tsv")"
  if [[ "${failed}" -eq 0 ]]; then
    touch "${RUN_ROOT}/state/all_complete"
    echo "[gt49f] all metrics complete"
  else
    touch "${RUN_ROOT}/state/complete_with_failures"
    echo "[gt49f] complete with ${failed} failed tasks"
  fi
  exec bash
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

"${PYTHON}" "${ROOT}/build_seed851_gt49f_bench_batch.py" \
  --output-root "${BATCH_ROOT}"
mkdir -p "${RUN_ROOT}"
tmux new-session -d -s "${SESSION}" -n orchestrator \
  "RUN_ROOT='${RUN_ROOT}' RUN_TAG='${RUN_TAG}' SESSION='${SESSION}' bash '$0' --inside"
echo "tmux session: ${SESSION}"
echo "batch root: ${BATCH_ROOT}"
echo "run root: ${RUN_ROOT}"
echo "GPUs: ${GPUS[*]} (GPU4 excluded)"
