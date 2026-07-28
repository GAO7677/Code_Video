#!/usr/bin/env bash
# Run:
#   bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_seed851_missing_metrics_requested_layout_tmux.sh

set -euo pipefail

ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
SOURCE_ROOT=/data/gaoya/agent-data/outputs/wan_dit_common22_test5_st_phased_seed851/generated
BATCH_ROOT=/data/gaoya/agent-data/outputs/wan_dit_common22_test5_st_phased_seed851_bench
RUN_ROOT="${BATCH_ROOT}/run_20260728_missing900_requested_layout"
SESSION="${SESSION:-wan_seed851_missing_requested_layout_20260728}"
WORKER="${ROOT}/run_seed851_memory_guarded_metric_worker.sh"
MONITOR="${ROOT}/monitor_seed851_missing_metrics.sh"

PMF_GPUS=(0 1 2 3 5 6 7)
COMMON_GPUS=(0 1 2 3)
VBENCH_EXTRA_GPUS=(0 1 2 3 5 6 7)
VIDEOPHY_GPUS=(5)
COSMOS_GPUS=(6 7)
PMF_WORKERS_PER_GPU=5
COMMON_WORKERS_PER_GPU=3
VIDEOPHY_WORKERS_PER_GPU=2
COSMOS_WORKERS_PER_GPU=2
VBENCH_EXTRA_WORKERS_PER_GPU=1
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

task_index=0
for metric in pmf_with_context pmf_without_context; do
  for shard in $(seq 0 34); do
    printf 'cpu-%04d\t%s\t%s\t35\n' \
      "${task_index}" "${metric}" "${shard}" \
      >> "${RUN_ROOT}/queues/cpu.tsv"
    task_index=$((task_index + 1))
  done
done
for shard in $(seq 0 6); do
  printf 'cpu-%04d\tphysics_iq_without_context\t%s\t7\n' \
    "${task_index}" "${shard}" >> "${RUN_ROOT}/queues/cpu.tsv"
  task_index=$((task_index + 1))
done

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
for metric in "${common_metrics[@]}"; do
  for shard in $(seq 0 11); do
    printf 'common-%04d\t%s\t%s\t12\n' \
      "${task_index}" "${metric}" "${shard}" \
      >> "${RUN_ROOT}/queues/common.tsv"
    task_index=$((task_index + 1))
  done
done

for shard in 0 1; do
  printf 'videophy-%04d\tvideophy2\t%s\t2\n' \
    "${shard}" "${shard}" >> "${RUN_ROOT}/queues/videophy.tsv"
done
for shard in 0 1 2 3; do
  printf 'cosmos-%04d\tcosmos_reason1\t%s\t4\n' \
    "${shard}" "${shard}" >> "${RUN_ROOT}/queues/cosmos.tsv"
done

cpu_workers=$(( ${#PMF_GPUS[@]} * PMF_WORKERS_PER_GPU ))
common_workers=$(( ${#COMMON_GPUS[@]} * COMMON_WORKERS_PER_GPU + ${#VBENCH_EXTRA_GPUS[@]} * VBENCH_EXTRA_WORKERS_PER_GPU ))
videophy_workers=$(( ${#VIDEOPHY_GPUS[@]} * VIDEOPHY_WORKERS_PER_GPU ))
cosmos_workers=$(( ${#COSMOS_GPUS[@]} * COSMOS_WORKERS_PER_GPU ))

tmux new-session -d -s "${SESSION}" -n monitor \
  "MEMORY_STOP_GIB=128 MONITOR_INTERVAL_SECONDS=5 bash '${MONITOR}' '${RUN_ROOT}' '${BATCH_ROOT}' '${cpu_workers}' '${common_workers}' '${videophy_workers}' '${cosmos_workers}' '${SESSION}'"

for gpu in "${PMF_GPUS[@]}"; do
  for worker_index in $(seq 0 $((PMF_WORKERS_PER_GPU - 1))); do
    name="cpu_g${gpu}_${worker_index}"
    tmux new-window -t "${SESSION}" -n "${name}" \
      "METRIC_WORKER_THREADS=2 bash '${WORKER}' cpu '${gpu}' '${name}' '${RUN_ROOT}' '${BATCH_ROOT}' 49140"
  done
done
for gpu in "${COMMON_GPUS[@]}"; do
  for worker_index in $(seq 0 $((COMMON_WORKERS_PER_GPU - 1))); do
    name="common_g${gpu}_${worker_index}"
    tmux new-window -t "${SESSION}" -n "${name}" \
      "METRIC_WORKER_THREADS=2 bash '${WORKER}' common '${gpu}' '${name}' '${RUN_ROOT}' '${BATCH_ROOT}' 36000"
  done
done
for gpu in "${VBENCH_EXTRA_GPUS[@]}"; do
  name="common_vbench_g${gpu}_0"
  tmux new-window -t "${SESSION}" -n "${name}" \
    "METRIC_WORKER_THREADS=2 bash '${WORKER}' common '${gpu}' '${name}' '${RUN_ROOT}' '${BATCH_ROOT}' 40000"
done
for gpu in "${VIDEOPHY_GPUS[@]}"; do
  for worker_index in $(seq 0 $((VIDEOPHY_WORKERS_PER_GPU - 1))); do
    name="vp_g${gpu}_${worker_index}"
    tmux new-window -t "${SESSION}" -n "${name}" \
      "METRIC_WORKER_THREADS=2 bash '${WORKER}' videophy '${gpu}' '${name}' '${RUN_ROOT}' '${BATCH_ROOT}' 46000"
  done
done
for gpu in "${COSMOS_GPUS[@]}"; do
  name="cosmos_g${gpu}_0"
  tmux new-window -t "${SESSION}" -n "${name}" \
    "METRIC_WORKER_THREADS=2 bash '${WORKER}' cosmos '${gpu}' '${name}' '${RUN_ROOT}' '${BATCH_ROOT}' 40000"
done

tmux select-window -t "${SESSION}:monitor"
echo "tmux session: ${SESSION}"
echo "run root: ${RUN_ROOT}"
echo "layout: common=gpu0-3x3 videophy=gpu5x2 cosmos=gpu6-7x1 pmf=gpu0-3,5-7x5 gpu4=unused"
