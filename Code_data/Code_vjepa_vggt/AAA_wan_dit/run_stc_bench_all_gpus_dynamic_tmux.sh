#!/usr/bin/env bash
# Run:
#   bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_stc_bench_all_gpus_dynamic_tmux.sh

set -euo pipefail

ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
BATCH_ROOT=/data/gaoya/agent-data/outputs/wan_dit_stc_bench
RUN_ROOT="${BATCH_ROOT}/run_20260728_dynamic"
REPORT_ROOT=/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/gallery/multiseed/benchmark-metrics
SESSION="${SESSION:-wan_stc_bench_gpu0123456_dynamic_20260728}"
WORKER="${ROOT}/run_stc_bench_dynamic_worker.sh"
COORDINATOR="${ROOT}/run_stc_bench_dynamic_coordinator.sh"
GPUS=(0 1 2 3 4 5 6)
VIDEOPHY_GPUS=(0 2 6)
COSMOS_GPUS=(1 3 4 5)
COMMON_WORKERS_PER_GPU=3
GPU6_COMMON_WORKERS=10
VIDEOPHY_WORKERS_PER_GPU=1
COSMOS_WORKERS_PER_GPU=1

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
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
  for shard in $(seq 0 20); do
    printf 'common-%04d\t%s\t%s\t21\n' \
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
tmux new-session -d -s "${SESSION}" -n coordinator \
  "bash '${COORDINATOR}' '${RUN_ROOT}' '${BATCH_ROOT}' '${REPORT_ROOT}' '${common_workers}' '${videophy_workers}' '${cosmos_workers}'"

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
echo "tmux session: ${SESSION}"
echo "run root: ${RUN_ROOT}"
echo "common workers: ${common_workers}"
echo "videophy workers: ${videophy_workers}"
echo "cosmos workers: ${cosmos_workers}"
