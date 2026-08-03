#!/usr/bin/env bash
set -euo pipefail

# Run:
#   bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_recompute_aaaevalphysiq_5metrics_tmux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPU_WORKER="${SCRIPT_DIR}/run_recompute_aaaevalphysiq_cpu_worker.sh"
VIDEO_WORKER="${SCRIPT_DIR}/run_recompute_videophy2_generated_only_worker.sh"
COORDINATOR="${SCRIPT_DIR}/run_recompute_aaaevalphysiq_coordinator.sh"

ROOTS_FILE="${ROOTS_FILE:-/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/AAAevalphysiq.txt}"
INPUT_ALLOWLIST="${INPUT_ALLOWLIST:-/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt}"
EXPECTED_ROOTS="${EXPECTED_ROOTS:-38}"
EXPECTED_CASES="${EXPECTED_CASES:-67}"
CPU_WORKERS="${CPU_WORKERS:-12}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
GPU_MAX_USED_MIB="${GPU_MAX_USED_MIB:-22000}"
GPU_IDS_TEXT="${GPU_IDS:-2 5}"
read -r -a GPUS <<< "${GPU_IDS_TEXT}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SESSION="${SESSION:-aaaevalphysiq_recompute_5metrics_${STAMP}}"
RUN_ROOT="${RUN_ROOT:-/data/gaoya/agent-data/outputs/aaaevalphysiq_metric_recompute/${SESSION}}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/data/gaoya/agent-data/outputs/aaaevalphysiq_metric_recompute/artifacts/${SESSION}}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi
if [[ -e "${RUN_ROOT}" ]]; then
  echo "run root already exists: ${RUN_ROOT}" >&2
  exit 1
fi
if [[ ! -s "${ROOTS_FILE}" || ! -s "${INPUT_ALLOWLIST}" ]]; then
  echo "Missing roots file or input allowlist" >&2
  exit 2
fi

mapfile -t ROOTS < <(sed '/^[[:space:]]*$/d; /^[[:space:]]*#/d' "${ROOTS_FILE}")
mapfile -t INPUTS < <(sed '/^[[:space:]]*$/d; /^[[:space:]]*#/d' "${INPUT_ALLOWLIST}")
if [[ "${#ROOTS[@]}" -ne "${EXPECTED_ROOTS}" ]]; then
  echo "Expected ${EXPECTED_ROOTS} roots, got ${#ROOTS[@]}" >&2
  exit 2
fi
if [[ "${#INPUTS[@]}" -ne "${EXPECTED_CASES}" ]]; then
  echo "Expected ${EXPECTED_CASES} allowlisted cases, got ${#INPUTS[@]}" >&2
  exit 2
fi
if [[ "$(printf '%s\n' "${ROOTS[@]}" | sort -u | wc -l)" -ne "${EXPECTED_ROOTS}" ]]; then
  echo "Duplicate result roots" >&2
  exit 2
fi
for root in "${ROOTS[@]}"; do
  [[ -d "${root}" ]] || { echo "Missing result root: ${root}" >&2; exit 2; }
done
for gpu in "${GPUS[@]}"; do
  [[ "${gpu}" != "4" ]] || { echo "GPU4 is prohibited by workspace policy" >&2; exit 2; }
done

mkdir -p "${RUN_ROOT}/queues" "${RUN_ROOT}/logs" "${RUN_ROOT}/state" \
  "${RUN_ROOT}/task_summaries" "${ARTIFACT_ROOT}"
: > "${RUN_ROOT}/queues/cpu.tsv"
: > "${RUN_ROOT}/queues/videophy2.tsv"
: > "${RUN_ROOT}/completed_tasks.tsv"
: > "${RUN_ROOT}/failed_tasks.tsv"
printf '1\n' > "${RUN_ROOT}/queues/cpu.cursor"
printf '1\n' > "${RUN_ROOT}/queues/videophy2.cursor"
cp "${ROOTS_FILE}" "${RUN_ROOT}/result_roots.snapshot.txt"
cp "${INPUT_ALLOWLIST}" "${RUN_ROOT}/input_allowlist.snapshot.txt"

task_index=0
for metric in \
  physics_iq_with_context physics_iq_without_context \
  pmf_with_context pmf_without_context; do
  for root in "${ROOTS[@]}"; do
    printf 'cpu-%04d\t%s\t%s\n' "${task_index}" "${metric}" "${root}" \
      >> "${RUN_ROOT}/queues/cpu.tsv"
    task_index=$((task_index + 1))
  done
done
for index in "${!ROOTS[@]}"; do
  printf 'videophy2-%04d\t%s\t0\n' "${index}" "${ROOTS[$index]}" \
    >> "${RUN_ROOT}/queues/videophy2.tsv"
done

touch "${RUN_ROOT}/start.ready"
VIDEO_WORKERS=$(( ${#GPUS[@]} * WORKERS_PER_GPU ))

tmux new-session -d -s "${SESSION}" -n coordinator \
  "bash '${COORDINATOR}' '${RUN_ROOT}' '${RUN_ROOT}/result_roots.snapshot.txt' '${RUN_ROOT}/input_allowlist.snapshot.txt' '${CPU_WORKERS}' '${VIDEO_WORKERS}' '${EXPECTED_CASES}' '${ARTIFACT_ROOT}'"

for worker_index in $(seq 0 $((CPU_WORKERS - 1))); do
  name="cpu${worker_index}"
  tmux new-window -t "${SESSION}" -n "${name}" \
    "bash '${CPU_WORKER}' '${name}' '${RUN_ROOT}' '${RUN_ROOT}/input_allowlist.snapshot.txt' '${RUN_ROOT}/start.ready' '${ARTIFACT_ROOT}'"
done

for gpu in "${GPUS[@]}"; do
  for worker_index in $(seq 0 $((WORKERS_PER_GPU - 1))); do
    name="g${gpu}_vp${worker_index}"
    tmux new-window -t "${SESSION}" -n "${name}" \
      "bash '${VIDEO_WORKER}' '${gpu}' '${name}' '${RUN_ROOT}' '${RUN_ROOT}/input_allowlist.snapshot.txt' '${EXPECTED_CASES}' '${GPU_MAX_USED_MIB}' '${RUN_ROOT}/start.ready' '${RUN_ROOT}/unused_prior_state' 0"
  done
done

tmux select-window -t "${SESSION}:coordinator"
echo "tmux session: ${SESSION}"
echo "run root: ${RUN_ROOT}"
echo "artifacts: ${ARTIFACT_ROOT}"
echo "CPU queue: 152 tasks (${CPU_WORKERS} workers)"
echo "VideoPhy2 queue: 38 tasks (${VIDEO_WORKERS} workers on GPUs ${GPU_IDS_TEXT})"
