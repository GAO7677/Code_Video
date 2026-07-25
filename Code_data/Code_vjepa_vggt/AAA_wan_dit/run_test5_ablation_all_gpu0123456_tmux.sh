#!/usr/bin/env bash
set -euo pipefail

# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_test5_ablation_all_gpu0123456_tmux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${SESSION:-wan_dit_test5_ablation_63configs}"
SOURCE_LIST="${SOURCE_LIST:-/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt}"
OUTPUT_BASE="${OUTPUT_BASE:-/data/gaoya/agent-data/outputs/wan_dit_ablation/test5_first5}"
RUN_ROOT="${RUN_ROOT:-${OUTPUT_BASE}/_pipeline}"
INPUT_LIST="${RUN_ROOT}/input_first5_unique.txt"
GPUS=(0 1 2 3 4 5 6)
BLOCKS=(0 5 11 17 19 29)
GEN_WORKER="${SCRIPT_DIR}/run_test5_ablation_generation_worker.sh"
METRIC_WORKER="${SCRIPT_DIR}/run_test5_ablation_metric_wait_worker.sh"
COORDINATOR="${SCRIPT_DIR}/run_test5_ablation_pipeline_coordinator.sh"

CPU_WORKERS_PER_GPU=8
GPU_COMMON_WORKERS_PER_GPU=3
VIDEOPHY2_WORKERS_PER_GPU=2
COSMOS_WORKERS_PER_GPU=1
NUM_GEN_WORKERS="${#GPUS[@]}"
NUM_METRIC_WORKERS=$(( ${#GPUS[@]} * (
  CPU_WORKERS_PER_GPU + GPU_COMMON_WORKERS_PER_GPU +
  VIDEOPHY2_WORKERS_PER_GPU + COSMOS_WORKERS_PER_GPU
) ))

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi
if [[ ! -s "${SOURCE_LIST}" ]]; then
  echo "missing source list: ${SOURCE_LIST}" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}/generation/logs" "${RUN_ROOT}/generation/state" \
  "${RUN_ROOT}/generation/validations"
awk '!seen[$0]++ {print; if (++count == 5) exit}' "${SOURCE_LIST}" > "${INPUT_LIST}"
if [[ "$(wc -l < "${INPUT_LIST}")" -ne 5 ]]; then
  echo "failed to select five unique cases" >&2
  exit 2
fi

QUEUE="${RUN_ROOT}/generation/queue.tsv"
: > "${QUEUE}"
job_index=0
add_job() {
  printf 'gen-%03d\t%s\t%s\t%s\n' "$job_index" "$1" "$2" "$3" >> "${QUEUE}"
  job_index=$((job_index + 1))
}

add_job wan_lora baseline none
add_job xssc baseline none
for block in "${BLOCKS[@]}"; do
  add_job wan_lora whole_block "${block}"
  add_job wan_lora self_attn_zero "${block}"
  add_job xssc whole_block "${block}"
  add_job xssc self_attn_zero "${block}"
  add_job xssc object_cross_attn "${block}"
done
add_job physrvg baseline none
for block in "${BLOCKS[@]}"; do
  for mode in whole_block self_attn_zero text_cross_attn_zero ffn_zero lora_off; do
    add_job physrvg "${mode}" "${block}"
  done
done
if [[ "${job_index}" -ne 63 ]]; then
  echo "internal error: expected 63 jobs, got ${job_index}" >&2
  exit 2
fi

printf '1\n' > "${RUN_ROOT}/generation/cursor"
: > "${RUN_ROOT}/generation/completed.tsv"
: > "${RUN_ROOT}/generation/failed.tsv"
rm -f "${RUN_ROOT}/generation.failed" "${RUN_ROOT}/metrics.ready" \
  "${RUN_ROOT}/metrics.failed" "${RUN_ROOT}/pipeline.complete"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "dry_run=1"
  echo "output_base=${OUTPUT_BASE}"
  echo "run_root=${RUN_ROOT}"
  echo "generation_jobs=$(wc -l < "${QUEUE}") videos=315 workers=${NUM_GEN_WORKERS}"
  echo "metric_tasks=882 workers=${NUM_METRIC_WORKERS}"
  exit 0
fi

tmux new-session -d -s "${SESSION}" -n coordinator \
  "bash '${COORDINATOR}' '${RUN_ROOT}' '${OUTPUT_BASE}' '${INPUT_LIST}' '${NUM_GEN_WORKERS}' '${NUM_METRIC_WORKERS}'; exec bash"

for gpu in "${GPUS[@]}"; do
  worker_name="gen_g${gpu}"
  tmux new-window -t "${SESSION}" -n "${worker_name}" \
    "bash '${GEN_WORKER}' '${gpu}' '${worker_name}' '${RUN_ROOT}' '${OUTPUT_BASE}' '${INPUT_LIST}'; exec bash"
done

for gpu in "${GPUS[@]}"; do
  for index in $(seq 0 $((CPU_WORKERS_PER_GPU - 1))); do
    name="g${gpu}_cpu${index}"
    tmux new-window -t "${SESSION}" -n "${name}" \
      "bash '${METRIC_WORKER}' '${gpu}' cpu '${name}' '${RUN_ROOT}' '${INPUT_LIST}' '${RUN_ROOT}/metrics.ready'; exec bash"
  done
  for index in $(seq 0 $((GPU_COMMON_WORKERS_PER_GPU - 1))); do
    name="g${gpu}_gpu${index}"
    tmux new-window -t "${SESSION}" -n "${name}" \
      "bash '${METRIC_WORKER}' '${gpu}' gpu_common '${name}' '${RUN_ROOT}' '${INPUT_LIST}' '${RUN_ROOT}/metrics.ready'; exec bash"
  done
  for index in $(seq 0 $((VIDEOPHY2_WORKERS_PER_GPU - 1))); do
    name="g${gpu}_vp${index}"
    tmux new-window -t "${SESSION}" -n "${name}" \
      "bash '${METRIC_WORKER}' '${gpu}' videophy2 '${name}' '${RUN_ROOT}' '${INPUT_LIST}' '${RUN_ROOT}/metrics.ready'; exec bash"
  done
  name="g${gpu}_cosmos"
  tmux new-window -t "${SESSION}" -n "${name}" \
    "bash '${METRIC_WORKER}' '${gpu}' cosmos '${name}' '${RUN_ROOT}' '${INPUT_LIST}' '${RUN_ROOT}/metrics.ready'; exec bash"
done

tmux select-window -t "${SESSION}:coordinator"
echo "session=${SESSION}"
echo "output_base=${OUTPUT_BASE}"
echo "run_root=${RUN_ROOT}"
echo "generation_jobs=63 videos=315 workers=${NUM_GEN_WORKERS}"
echo "metric_tasks=882 workers=${NUM_METRIC_WORKERS}"
