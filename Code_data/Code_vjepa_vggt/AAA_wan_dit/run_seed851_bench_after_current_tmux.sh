#!/usr/bin/env bash
# Run:
#   bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_seed851_bench_after_current_tmux.sh

set -euo pipefail

ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
SOURCE_ROOT=/data/gaoya/agent-data/outputs/wan_dit_common22_test5_st_phased_seed851/generated
BATCH_ROOT=/data/gaoya/agent-data/outputs/wan_dit_common22_test5_st_phased_seed851_bench
RUN_ROOT="${BATCH_ROOT}/run_20260728_dynamic"
PREVIOUS_RUN=/data/gaoya/agent-data/outputs/wan_dit_stc_bench/run_20260728_dynamic
SESSION="${SESSION:-wan_seed851_bench_gpu0123456_20260728}"
WORKER="${ROOT}/run_stc_bench_dynamic_worker.sh"
COORDINATOR="${ROOT}/run_seed851_bench_coordinator.sh"
GPUS=(0 1 2 3 4 5 6)
VIDEOPHY_GPUS=(0 2 4 6)
COSMOS_GPUS=(1 3 5)
COMMON_WORKERS_PER_GPU=3
GPU6_COMMON_WORKERS=10
VIDEOPHY_WORKERS_PER_GPU=2
COSMOS_WORKERS_PER_GPU=1

queue_finished() {
  local kind="$1"
  local queued finished
  queued="$(wc -l < "${PREVIOUS_RUN}/queues/${kind}.tsv")"
  finished="$(
    awk -F'\t' -v prefix="${kind}-" \
      'index($1, prefix) == 1 {count++} END {print count + 0}' \
      "${PREVIOUS_RUN}/completed_tasks.tsv" \
      "${PREVIOUS_RUN}/failed_tasks.tsv"
  )"
  [[ "${finished}" -ge "${queued}" ]]
}

if [[ "${1:-}" == "--inside" ]]; then
  mkdir -p "${BATCH_ROOT}"
  exec > >(tee -a "${BATCH_ROOT}/waiter.log") 2>&1
  echo "[seed851-waiter] waiting for the current 503-case queues"
  until queue_finished common \
    && queue_finished videophy \
    && queue_finished cosmos; do
    common_cursor="$(( $(<"${PREVIOUS_RUN}/queues/common.cursor") - 1 ))"
    videophy_cursor="$(( $(<"${PREVIOUS_RUN}/queues/videophy.cursor") - 1 ))"
    cosmos_cursor="$(( $(<"${PREVIOUS_RUN}/queues/cosmos.cursor") - 1 ))"
    echo "[seed851-waiter] current claimed: common=${common_cursor}/147 videophy=${videophy_cursor}/14 cosmos=${cosmos_cursor}/7"
    sleep 60
  done

  echo "[seed851-waiter] previous queues complete; refreshing staged inputs"
  "${PYTHON}" "${ROOT}/build_seed851_bench_batch.py" \
    --source-root "${SOURCE_ROOT}" \
    --output-root "${BATCH_ROOT}"

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
    for shard in $(seq 0 13); do
      printf 'common-%04d\t%s\t%s\t14\n' \
        "${task_index}" "${metric}" "${shard}" \
        >> "${RUN_ROOT}/queues/common.tsv"
      task_index=$((task_index + 1))
    done
  done
  for shard in $(seq 0 13); do
    printf 'videophy-%04d\tvideophy2\t%s\t14\n' \
      "${shard}" "${shard}" >> "${RUN_ROOT}/queues/videophy.tsv"
  done
  for shard in $(seq 0 6); do
    printf 'cosmos-%04d\tcosmos_reason1\t%s\t7\n' \
      "${shard}" "${shard}" >> "${RUN_ROOT}/queues/cosmos.tsv"
  done

  common_workers=$(( (${#GPUS[@]} - 1) * COMMON_WORKERS_PER_GPU + GPU6_COMMON_WORKERS ))
  videophy_workers=$(( ${#VIDEOPHY_GPUS[@]} * VIDEOPHY_WORKERS_PER_GPU ))
  cosmos_workers=$(( ${#COSMOS_GPUS[@]} * COSMOS_WORKERS_PER_GPU ))
  tmux new-window -t "${SESSION}" -n coordinator \
    "bash '${COORDINATOR}' '${RUN_ROOT}' '${BATCH_ROOT}' '${common_workers}' '${videophy_workers}' '${cosmos_workers}'"

  for gpu in "${GPUS[@]}"; do
    common_worker_count="${COMMON_WORKERS_PER_GPU}"
    if [[ "${gpu}" -eq 6 ]]; then
      common_worker_count="${GPU6_COMMON_WORKERS}"
    fi
    for worker_index in $(seq 0 $((common_worker_count - 1))); do
      name="common_g${gpu}_${worker_index}"
      tmux new-window -t "${SESSION}" -n "${name}" \
        "bash '${WORKER}' common '${gpu}' '${name}' '${RUN_ROOT}' '${BATCH_ROOT}' 24500"
    done
  done

  for gpu in "${VIDEOPHY_GPUS[@]}"; do
    for worker_index in $(seq 0 $((VIDEOPHY_WORKERS_PER_GPU - 1))); do
      name="videophy_g${gpu}_${worker_index}"
      tmux new-window -t "${SESSION}" -n "${name}" \
        "bash '${WORKER}' videophy '${gpu}' '${name}' '${RUN_ROOT}' '${BATCH_ROOT}' 24500"
    done
  done
  for gpu in "${COSMOS_GPUS[@]}"; do
    name="cosmos_g${gpu}_0"
    tmux new-window -t "${SESSION}" -n "${name}" \
      "bash '${WORKER}' cosmos '${gpu}' '${name}' '${RUN_ROOT}' '${BATCH_ROOT}' 24500"
  done
  tmux select-window -t "${SESSION}:coordinator"
  echo "[seed851-waiter] workers started"
  exec bash
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

"${PYTHON}" "${ROOT}/build_seed851_bench_batch.py" \
  --source-root "${SOURCE_ROOT}" \
  --output-root "${BATCH_ROOT}"

tmux new-session -d -s "${SESSION}" -n waiter \
  "bash '$0' --inside"
echo "tmux session: ${SESSION}"
echo "batch root: ${BATCH_ROOT}"
echo "run root: ${RUN_ROOT}"
exit 0
