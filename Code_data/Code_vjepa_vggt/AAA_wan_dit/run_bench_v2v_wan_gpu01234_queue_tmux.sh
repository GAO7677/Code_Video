#!/usr/bin/env bash
set -euo pipefail

# Run:
#   bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_bench_v2v_wan_gpu01234_queue_tmux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${SCRIPT_DIR}/run_bench_v2v_wan_queue_worker.sh"
PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
SUMMARY=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/summarize_benchmark_txt_metrics.py
VERIFY=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/verify_bench_physiq_metrics.py
BASELINE_LIST="${BASELINE_LIST:-/data/gaoya/AAA_test_video/0623/test/v2v_wan/leaf_folders.txt}"
INPUT_ALLOWLIST="${INPUT_ALLOWLIST:-/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt}"
SESSION="${SESSION:-bench_v2v_wan_32folders_gpu01234_20260724}"
RUN_ROOT="${RUN_ROOT:-/data/gaoya/AAA_test_video/0623/test/v2v_wan/_bench_runs/${SESSION}}"
GPUS=(0 1 2 3 4)
CPU_WORKERS_PER_GPU=8
GPU_WORKERS_PER_GPU=3

CPU_METRICS=(
  physics_iq_with_context
  physics_iq_without_context
  pmf_with_context
  pmf_without_context
)
GPU_METRICS=(
  wmreward
  vbench_subject_consistency
  vbench_background_consistency
  vbench_temporal_flickering
  vbench_motion_smoothness
  vbench_dynamic_degree
  vbench_aesthetic_quality
  vbench_imaging_quality
  videophy2
  cosmos_reason1
)

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi
if [[ ! -s "${BASELINE_LIST}" || ! -s "${INPUT_ALLOWLIST}" ]]; then
  echo "Missing baseline list or input allowlist" >&2
  exit 2
fi

mapfile -t RESULT_ROOTS < <(sed '/^[[:space:]]*$/d; /^[[:space:]]*#/d' "${BASELINE_LIST}")
if [[ "${#RESULT_ROOTS[@]}" -ne 32 ]]; then
  echo "Expected 32 result roots, got ${#RESULT_ROOTS[@]}" >&2
  exit 2
fi
for root in "${RESULT_ROOTS[@]}"; do
  if [[ ! -d "${root}" ]]; then
    echo "Missing result root: ${root}" >&2
    exit 2
  fi
done

mkdir -p "${RUN_ROOT}/queues" "${RUN_ROOT}/logs" "${RUN_ROOT}/state" "${RUN_ROOT}/task_summaries"
: > "${RUN_ROOT}/queues/cpu.tsv"
: > "${RUN_ROOT}/queues/gpu.tsv"
: > "${RUN_ROOT}/completed_tasks.tsv"
: > "${RUN_ROOT}/failed_tasks.tsv"
printf '1\n' > "${RUN_ROOT}/queues/cpu.cursor"
printf '1\n' > "${RUN_ROOT}/queues/gpu.cursor"
cp "${BASELINE_LIST}" "${RUN_ROOT}/leaf_folders.snapshot.txt"

task_index=0
for metric in "${CPU_METRICS[@]}"; do
  for root in "${RESULT_ROOTS[@]}"; do
    printf 'cpu-%04d\t%s\t%s\n' "${task_index}" "${metric}" "${root}" >> "${RUN_ROOT}/queues/cpu.tsv"
    task_index=$((task_index + 1))
  done
done

task_index=0
for metric in "${GPU_METRICS[@]}"; do
  for root in "${RESULT_ROOTS[@]}"; do
    printf 'gpu-%04d\t%s\t%s\n' "${task_index}" "${metric}" "${root}" >> "${RUN_ROOT}/queues/gpu.tsv"
    task_index=$((task_index + 1))
  done
done

total_workers=$(( ${#GPUS[@]} * (CPU_WORKERS_PER_GPU + GPU_WORKERS_PER_GPU) ))
tmux new-session -d -s "${SESSION}" -n coordinator \
  "while true; do done_count=\$(find '${RUN_ROOT}/state' -maxdepth 1 -name '*.complete' -type f | wc -l); cpu_cursor=\$(cat '${RUN_ROOT}/queues/cpu.cursor'); gpu_cursor=\$(cat '${RUN_ROOT}/queues/gpu.cursor'); printf '[coordinator] workers=%s/${total_workers} cpu_claimed=%s/128 gpu_claimed=%s/320 completed_tasks=%s failed_tasks=%s\\n' \"\$done_count\" \"\$((cpu_cursor - 1))\" \"\$((gpu_cursor - 1))\" \"\$(wc -l < '${RUN_ROOT}/completed_tasks.tsv')\" \"\$(wc -l < '${RUN_ROOT}/failed_tasks.tsv')\"; [ \"\$done_count\" -eq '${total_workers}' ] && break; sleep 30; done; '${PYTHON_BIN}' '${SUMMARY}' --input-txt '${BASELINE_LIST}' --output-csv '${RUN_ROOT}/metric_summary.csv' --input-json-allowlist '${INPUT_ALLOWLIST}'; '${PYTHON_BIN}' '${VERIFY}' --baseline-list '${BASELINE_LIST}' --output '${RUN_ROOT}/verification.json' --input-json-allowlist '${INPUT_ALLOWLIST}'; exec bash"

for gpu in "${GPUS[@]}"; do
  for worker_index in $(seq 0 $((CPU_WORKERS_PER_GPU - 1))); do
    name="g${gpu}_cpu${worker_index}"
    tmux new-window -t "${SESSION}" -n "${name}" \
      "bash '${WORKER}' '${gpu}' cpu '${name}' '${RUN_ROOT}' '${INPUT_ALLOWLIST}'"
  done
  for worker_index in $(seq 0 $((GPU_WORKERS_PER_GPU - 1))); do
    name="g${gpu}_gpu${worker_index}"
    tmux new-window -t "${SESSION}" -n "${name}" \
      "bash '${WORKER}' '${gpu}' gpu '${name}' '${RUN_ROOT}' '${INPUT_ALLOWLIST}'"
  done
done

tmux select-window -t "${SESSION}:coordinator"
echo "tmux session: ${SESSION}"
echo "run root: ${RUN_ROOT}"
echo "result roots: ${#RESULT_ROOTS[@]}"
echo "workers: ${total_workers} (CPU ${CPU_WORKERS_PER_GPU}/GPU x 5, GPU ${GPU_WORKERS_PER_GPU}/GPU x 5)"
echo "tasks: CPU 128, GPU 320"
