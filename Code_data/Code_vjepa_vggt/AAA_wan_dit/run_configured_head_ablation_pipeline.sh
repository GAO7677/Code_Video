#!/usr/bin/env bash
set -euo pipefail

# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_configured_head_ablation_pipeline.sh \
#   /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/config_head_ablation_all_blocks_test5.sh

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 CONFIG" >&2
  exit 2
fi

CONFIG="$(realpath "$1")"
# shellcheck source=/dev/null
source "${CONFIG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GEN_WORKER="${SCRIPT_DIR}/run_configured_head_ablation_generation_worker.sh"
METRIC_WORKER="${SCRIPT_DIR}/run_test5_ablation_metric_wait_worker.sh"
COORDINATOR="${SCRIPT_DIR}/run_configured_head_ablation_coordinator.sh"
INPUT_LIST="${RUN_ROOT}/input_unique.txt"

expand_integer_spec() {
  local spec="$1" token start stop value
  spec="${spec//,/ }"
  for token in ${spec}; do
    if [[ "${token}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      start="${BASH_REMATCH[1]}"
      stop="${BASH_REMATCH[2]}"
      for ((value=start; value<=stop; value++)); do
        printf '%s\n' "${value}"
      done
    elif [[ "${token}" =~ ^[0-9]+$ ]]; then
      printf '%s\n' "${token}"
    else
      echo "invalid integer/range token: ${token}" >&2
      return 2
    fi
  done
}

read -r -a model_array <<< "${MODELS}"
read -r -a gpu_array <<< "${GPUS}"
mapfile -t block_array < <(expand_integer_spec "${BLOCKS}")
mapfile -t head_array < <(expand_integer_spec "${HEADS}")
if [[ "${#model_array[@]}" -eq 0 || "${#gpu_array[@]}" -eq 0 \
      || "${#block_array[@]}" -eq 0 || "${#head_array[@]}" -eq 0 ]]; then
  echo "models, GPUs, blocks, and heads must all be non-empty" >&2
  exit 2
fi
for block in "${block_array[@]}"; do
  (( block >= 0 && block < 30 )) || { echo "invalid block ${block}" >&2; exit 2; }
done
for head in "${head_array[@]}"; do
  (( head >= 0 && head < 24 )) || { echo "invalid head ${head}" >&2; exit 2; }
done
for model in "${model_array[@]}"; do
  [[ "${model}" =~ ^(wan_lora|xssc|physrvg)$ ]] || {
    echo "invalid model ${model}" >&2
    exit 2
  }
done
[[ -s "${SOURCE_LIST}" ]] || { echo "missing source list ${SOURCE_LIST}" >&2; exit 2; }
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

if [[ -e "${RUN_ROOT}/run_initialized" && "${RESUME}" != "1" ]]; then
  echo "run already initialized; set RESUME=1 in ${CONFIG}" >&2
  exit 1
fi
mkdir -p "${RUN_ROOT}/generation/logs" "${RUN_ROOT}/generation/state" \
  "${RUN_ROOT}/generation/task_state" "${RUN_ROOT}/generation/validations"
if [[ "${DEDUPLICATE_INPUTS}" == "1" ]]; then
  awk 'NF && $0 !~ /^[[:space:]]*#/ && !seen[$0]++ {print}' \
    "${SOURCE_LIST}" > "${INPUT_LIST}"
else
  awk 'NF && $0 !~ /^[[:space:]]*#/ {print}' "${SOURCE_LIST}" > "${INPUT_LIST}"
fi
actual_cases="$(wc -l < "${INPUT_LIST}")"
if [[ "${actual_cases}" -ne "${EXPECTED_CASES}" ]]; then
  echo "expected ${EXPECTED_CASES} unique cases, got ${actual_cases}" >&2
  exit 2
fi
stem_count="$(while IFS= read -r path; do basename "${path}" .json; done \
  < "${INPUT_LIST}" | sort -u | wc -l)"
if [[ "${stem_count}" -ne "${EXPECTED_CASES}" ]]; then
  echo "input JSON stems are not unique: ${stem_count}/${EXPECTED_CASES}" >&2
  exit 2
fi

QUEUE="${RUN_ROOT}/generation/queue.tsv"
: > "${QUEUE}"
job_index=0
for model in "${model_array[@]}"; do
  for block in "${block_array[@]}"; do
    for head in "${head_array[@]}"; do
      printf 'head-%04d\t%s\t%s\t%s\n' \
        "${job_index}" "${model}" "${block}" "${head}" >> "${QUEUE}"
      job_index=$((job_index + 1))
    done
  done
done
EXPECTED_JOBS=$(( ${#model_array[@]} * ${#block_array[@]} * ${#head_array[@]} ))
[[ "${job_index}" -eq "${EXPECTED_JOBS}" ]]

printf '1\n' > "${RUN_ROOT}/generation/cursor"
if [[ "${RESUME}" != "1" ]]; then
  : > "${RUN_ROOT}/generation/completed.tsv"
  : > "${RUN_ROOT}/generation/failed.tsv"
  : > "${RUN_ROOT}/run_initialized"
fi
find "${RUN_ROOT}/generation/state" -name '*.worker.complete' -type f -delete
rm -f "${RUN_ROOT}/generation.failed" "${RUN_ROOT}/metrics.ready" \
  "${RUN_ROOT}/metrics.failed" "${RUN_ROOT}/pipeline.complete"
cp "${CONFIG}" "${RUN_ROOT}/config_snapshot.sh"
ACTIVE_CONFIG="${RUN_ROOT}/config_snapshot.sh"

NUM_GEN_WORKERS=$(( ${#gpu_array[@]} * GEN_WORKERS_PER_GPU ))
NUM_METRIC_WORKERS=$(( ${#gpu_array[@]} * (
  CPU_WORKERS_PER_GPU + GPU_COMMON_WORKERS_PER_GPU +
  VIDEOPHY2_WORKERS_PER_GPU + COSMOS_WORKERS_PER_GPU
) ))
read -r -a cpu_metrics <<< "${CPU_METRICS}"
read -r -a gpu_metrics <<< "${GPU_COMMON_METRICS}"
read -r -a vp_metrics <<< "${VIDEOPHY2_METRICS}"
read -r -a cosmos_metrics <<< "${COSMOS_METRICS}"
METRICS_PER_CONFIG=$(( ${#cpu_metrics[@]} + ${#gpu_metrics[@]} +
  ${#vp_metrics[@]} + ${#cosmos_metrics[@]} ))
EXPECTED_VIDEOS=$((EXPECTED_JOBS * EXPECTED_CASES))
EXPECTED_METRIC_TASKS=$((EXPECTED_JOBS * METRICS_PER_CONFIG))

cat > "${RUN_ROOT}/run_summary.txt" <<EOF
session=${SESSION}
models=${MODELS}
blocks=${BLOCKS}
heads=${HEADS}
gpus=${GPUS}
cases=${EXPECTED_CASES}
generation_configs=${EXPECTED_JOBS}
expected_videos=${EXPECTED_VIDEOS}
metrics_per_config=${METRICS_PER_CONFIG}
expected_metric_tasks=${EXPECTED_METRIC_TASKS}
generation_workers=${NUM_GEN_WORKERS}
metric_workers=${NUM_METRIC_WORKERS}
output_base=${OUTPUT_BASE}
EOF

if [[ "${DRY_RUN}" == "1" ]]; then
  cat "${RUN_ROOT}/run_summary.txt"
  exit 0
fi

tmux new-session -d -s "${SESSION}" -n coordinator \
  "bash '${COORDINATOR}' '${ACTIVE_CONFIG}' '${EXPECTED_JOBS}' '${NUM_GEN_WORKERS}' '${NUM_METRIC_WORKERS}'; exec bash"

worker_index=0
for gpu in "${gpu_array[@]}"; do
  for ((slot=0; slot<GEN_WORKERS_PER_GPU; slot++)); do
    name="gen_g${gpu}_${slot}"
    tmux new-window -t "${SESSION}" -n "${name}" \
      "bash '${GEN_WORKER}' '${ACTIVE_CONFIG}' '${gpu}' '${name}'; exec bash"
    worker_index=$((worker_index + 1))
  done
done
[[ "${worker_index}" -eq "${NUM_GEN_WORKERS}" ]]

launch_metric_workers() {
  local gpu="$1" count="$2" kind="$3" prefix="$4" index name
  for ((index=0; index<count; index++)); do
    name="g${gpu}_${prefix}${index}"
    tmux new-window -t "${SESSION}" -n "${name}" \
      "bash '${METRIC_WORKER}' '${gpu}' '${kind}' '${name}' '${RUN_ROOT}' '${INPUT_LIST}' '${RUN_ROOT}/metrics.ready'; exec bash"
  done
}
for gpu in "${gpu_array[@]}"; do
  launch_metric_workers "${gpu}" "${CPU_WORKERS_PER_GPU}" cpu cpu
  launch_metric_workers "${gpu}" "${GPU_COMMON_WORKERS_PER_GPU}" gpu_common gpu
  launch_metric_workers "${gpu}" "${VIDEOPHY2_WORKERS_PER_GPU}" videophy2 vp
  launch_metric_workers "${gpu}" "${COSMOS_WORKERS_PER_GPU}" cosmos cosmos
done

tmux select-window -t "${SESSION}:coordinator"
cat "${RUN_ROOT}/run_summary.txt"
echo "tmux_session=${SESSION}"
