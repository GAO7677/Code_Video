#!/usr/bin/env bash
# Run:
#   bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_seed851_missing_metrics_all_gpus_tmux.sh

set -euo pipefail

ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
SOURCE_ROOT=/data/gaoya/agent-data/outputs/wan_dit_common22_test5_st_phased_seed851/generated
BATCH_ROOT=/data/gaoya/agent-data/outputs/wan_dit_common22_test5_st_phased_seed851_bench
RUN_ROOT="${BATCH_ROOT}/run_20260728_missing900_safe"
SESSION="${SESSION:-wan_seed851_missing_metrics_safe_gpu01234567_20260728}"
WORKER="${ROOT}/run_seed851_memory_guarded_metric_worker.sh"
MONITOR="${ROOT}/monitor_seed851_missing_metrics.sh"
GPUS=(0 1 2 3 4 5 6 7)

CPU_WORKERS_PER_GPU=1
COMMON_WORKERS_PER_GPU=2
VIDEOPHY_WORKERS_PER_GPU=1
COSMOS_WORKERS_PER_GPU=1
CPU_SHARDS=16
COMMON_SHARDS=16
VIDEOPHY_SHARDS=8
COSMOS_SHARDS=8
EXPECTED_CASES=900

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi
if [[ -e "${RUN_ROOT}" ]]; then
  echo "run root already exists: ${RUN_ROOT}" >&2
  exit 1
fi

"${PYTHON}" "${ROOT}/build_seed851_bench_batch.py" \
  --source-root "${SOURCE_ROOT}" \
  --output-root "${BATCH_ROOT}"

manifest_cases="$(
  "${PYTHON}" -c \
    "import json; print(json.load(open('${BATCH_ROOT}/batch_manifest.json'))['num_entries'])"
)"
if [[ "${manifest_cases}" -ne "${EXPECTED_CASES}" ]]; then
  echo "expected ${EXPECTED_CASES} staged cases, found ${manifest_cases}" >&2
  exit 1
fi

mkdir -p \
  "${RUN_ROOT}/queues" "${RUN_ROOT}/logs" "${RUN_ROOT}/state" \
  "${RUN_ROOT}/task_summaries"
: > "${RUN_ROOT}/completed_tasks.tsv"
: > "${RUN_ROOT}/failed_tasks.tsv"

for kind in cpu common videophy cosmos; do
  : > "${RUN_ROOT}/queues/${kind}.tsv"
  printf '1\n' > "${RUN_ROOT}/queues/${kind}.cursor"
  : > "${RUN_ROOT}/queues/${kind}.lock"
done

cpu_metrics=(
  physics_iq_with_context
  physics_iq_without_context
  pmf_with_context
  pmf_without_context
)
common_metrics=(
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
for metric in "${cpu_metrics[@]}"; do
  for shard in $(seq 0 $((CPU_SHARDS - 1))); do
    printf 'cpu-%04d\t%s\t%s\t%s\n' \
      "${task_index}" "${metric}" "${shard}" "${CPU_SHARDS}" \
      >> "${RUN_ROOT}/queues/cpu.tsv"
    task_index=$((task_index + 1))
  done
done

task_index=0
for metric in "${common_metrics[@]}"; do
  for shard in $(seq 0 $((COMMON_SHARDS - 1))); do
    printf 'common-%04d\t%s\t%s\t%s\n' \
      "${task_index}" "${metric}" "${shard}" "${COMMON_SHARDS}" \
      >> "${RUN_ROOT}/queues/common.tsv"
    task_index=$((task_index + 1))
  done
done

for shard in $(seq 0 $((VIDEOPHY_SHARDS - 1))); do
  printf 'videophy-%04d\tvideophy2\t%s\t%s\n' \
    "${shard}" "${shard}" "${VIDEOPHY_SHARDS}" \
    >> "${RUN_ROOT}/queues/videophy.tsv"
done
for shard in $(seq 0 $((COSMOS_SHARDS - 1))); do
  printf 'cosmos-%04d\tcosmos_reason1\t%s\t%s\n' \
    "${shard}" "${shard}" "${COSMOS_SHARDS}" \
    >> "${RUN_ROOT}/queues/cosmos.tsv"
done

cpu_workers=$(( ${#GPUS[@]} * CPU_WORKERS_PER_GPU ))
common_workers=$(( ${#GPUS[@]} * COMMON_WORKERS_PER_GPU ))
videophy_workers=$(( ${#GPUS[@]} * VIDEOPHY_WORKERS_PER_GPU ))
cosmos_workers=$(( ${#GPUS[@]} * COSMOS_WORKERS_PER_GPU ))

tmux new-session -d -s "${SESSION}" -n monitor \
  "bash '${MONITOR}' '${RUN_ROOT}' '${BATCH_ROOT}' '${cpu_workers}' '${common_workers}' '${videophy_workers}' '${cosmos_workers}' '${SESSION}'"

for gpu in "${GPUS[@]}"; do
  for worker_index in $(seq 0 $((CPU_WORKERS_PER_GPU - 1))); do
    name="cpu_g${gpu}_${worker_index}"
    tmux new-window -t "${SESSION}" -n "${name}" \
      "bash '${WORKER}' cpu '${gpu}' '${name}' '${RUN_ROOT}' '${BATCH_ROOT}' 49140"
  done
done

for gpu in "${GPUS[@]}"; do
  for worker_index in $(seq 0 $((COMMON_WORKERS_PER_GPU - 1))); do
    name="common_g${gpu}_${worker_index}"
    tmux new-window -t "${SESSION}" -n "${name}" \
      "while [[ \$(find '${RUN_ROOT}/state' -maxdepth 1 -type f -name 'cpu_*.worker_complete' | wc -l) -lt '${cpu_workers}' ]]; do sleep 30; done; exec bash '${WORKER}' common '${gpu}' '${name}' '${RUN_ROOT}' '${BATCH_ROOT}' 36000"
  done
done

for gpu in "${GPUS[@]}"; do
  for worker_index in $(seq 0 $((VIDEOPHY_WORKERS_PER_GPU - 1))); do
    name="vp_g${gpu}_${worker_index}"
    tmux new-window -t "${SESSION}" -n "${name}" \
      "while [[ \$(find '${RUN_ROOT}/state' -maxdepth 1 -type f -name 'common_*.worker_complete' | wc -l) -lt '${common_workers}' ]]; do sleep 30; done; exec bash '${WORKER}' videophy '${gpu}' '${name}' '${RUN_ROOT}' '${BATCH_ROOT}' 36000"
  done
done

for gpu in "${GPUS[@]}"; do
  name="cosmos_g${gpu}_0"
  tmux new-window -t "${SESSION}" -n "${name}" \
    "while [[ \$(find '${RUN_ROOT}/state' -maxdepth 1 -type f -name 'vp_*.worker_complete' | wc -l) -lt '${videophy_workers}' ]]; do sleep 30; done; exec bash '${WORKER}' cosmos '${gpu}' '${name}' '${RUN_ROOT}' '${BATCH_ROOT}' 36000"
done

tmux select-window -t "${SESSION}:monitor"
echo "tmux session: ${SESSION}"
echo "batch cases: ${manifest_cases}"
echo "run root: ${RUN_ROOT}"
echo "workers: cpu=${cpu_workers} common=${common_workers} videophy=${videophy_workers} cosmos=${cosmos_workers}"
