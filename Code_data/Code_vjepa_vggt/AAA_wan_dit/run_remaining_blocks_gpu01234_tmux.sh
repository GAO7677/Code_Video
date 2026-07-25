#!/usr/bin/env bash
# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_remaining_blocks_gpu01234_tmux.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/remaining_blocks_experiment.env"
PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
MANAGER="${SCRIPT_DIR}/manage_remaining_block_pipeline.py"
GEN_WORKER="${SCRIPT_DIR}/run_remaining_blocks_generation_worker.sh"
METRIC_WORKER="${SCRIPT_DIR}/run_remaining_blocks_metric_wait_worker.sh"
COORDINATOR="${SCRIPT_DIR}/run_remaining_blocks_coordinator.sh"
if [[ ! -s "${CONFIG}" ]]; then
  echo "Missing experiment config: ${CONFIG}" >&2
  exit 2
fi
set -a
source "${CONFIG}"
set +a

RUN_ROOT="${OUTPUT_BASE}/_remaining_blocks_pipeline"
SESSION="${SESSION:-wan_dit_remaining_blocks_gpu01234}"
GPUS=(0 1 2 3 4)
CPU_WORKERS_PER_GPU=2
NUM_GEN="${#GPUS[@]}"
NUM_CPU=$((NUM_GEN * CPU_WORKERS_PER_GPU))
NUM_GPU="${#GPUS[@]}"
NUM_COSMOS="${#GPUS[@]}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi
if [[ -e "${RUN_ROOT}" ]]; then
  echo "run root already exists: ${RUN_ROOT}" >&2
  exit 1
fi
if [[ "$(sed '/^[[:space:]]*$/d; /^[[:space:]]*#/d' "${INPUT_LIST}" | wc -l)" -ne 67 ]]; then
  echo "expected exactly 67 input cases" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}/generation/logs" "${RUN_ROOT}/generation/state" \
  "${RUN_ROOT}/generation/validations" "${RUN_ROOT}/metrics/queues" \
  "${RUN_ROOT}/metrics/logs" "${RUN_ROOT}/metrics/state" \
  "${RUN_ROOT}/metrics/task_summaries"

"${PYTHON_BIN}" "${MANAGER}" build-generation \
  --output-base "${OUTPUT_BASE}" \
  --queue "${RUN_ROOT}/generation/queue.tsv" \
  --report "${RUN_ROOT}/generation_manifest.json"
printf '1\n' > "${RUN_ROOT}/generation/cursor"
: > "${RUN_ROOT}/generation/queue.lock"
: > "${RUN_ROOT}/generation/completed.tsv"
: > "${RUN_ROOT}/generation/failed.tsv"

tmux new-session -d -s "${SESSION}" -n coordinator \
  "bash '${COORDINATOR}' '${RUN_ROOT}' '${OUTPUT_BASE}' '${INPUT_LIST}' '${NUM_GEN}' '${NUM_CPU}' '${NUM_GPU}' '${NUM_COSMOS}'; exec bash"

for gpu in "${GPUS[@]}"; do
  name="gen_g${gpu}"
  tmux new-window -t "${SESSION}" -n "${name}" \
    "bash '${GEN_WORKER}' '${gpu}' '${name}' '${RUN_ROOT}' '${OUTPUT_BASE}' '${INPUT_LIST}'; exec bash"
done

for gpu in "${GPUS[@]}"; do
  for index in $(seq 0 $((CPU_WORKERS_PER_GPU - 1))); do
    name="cpu_g${gpu}_${index}"
    tmux new-window -t "${SESSION}" -n "${name}" \
      "bash '${METRIC_WORKER}' '${gpu}' cpu '${name}' '${RUN_ROOT}' '${INPUT_LIST}' '${RUN_ROOT}/cpu.ready' cpu; exec bash"
  done
  for stage in gpu_common videophy2 cosmos retry; do
    case "${stage}" in
      gpu_common) prefix=gpu; ready="${RUN_ROOT}/gpu_common.ready" ;;
      videophy2) prefix=vp; ready="${RUN_ROOT}/videophy2.ready" ;;
      cosmos) prefix=cosmos; ready="${RUN_ROOT}/cosmos.ready" ;;
      retry) prefix=retry; ready="${RUN_ROOT}/retry.ready" ;;
    esac
    name="${prefix}_g${gpu}"
    tmux new-window -t "${SESSION}" -n "${name}" \
      "bash '${METRIC_WORKER}' '${gpu}' '${stage}' '${name}' '${RUN_ROOT}' '${INPUT_LIST}' '${ready}' '${prefix}'; exec bash"
  done
done

tmux select-window -t "${SESSION}:coordinator"
echo "session=${SESSION}"
echo "run_root=${RUN_ROOT}"
echo "generation_jobs=240, expected_videos=16080, generation_workers=${NUM_GEN}"
echo "metric_roots=303, metric_types=15, metric_tasks=4545"
echo "stages=generation -> cpu -> gpu_common -> videophy2 -> cosmos -> retry -> verify -> plot"
