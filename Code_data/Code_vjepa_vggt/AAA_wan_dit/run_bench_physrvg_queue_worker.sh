#!/usr/bin/env bash
set -uo pipefail

if [[ "$#" -ne 7 ]]; then
  echo "Usage: $0 GPU_ID KIND WORKER_NAME RUN_ROOT INPUT_ALLOWLIST EXPECTED_CASES GPU_READY_MAX_USED_MIB" >&2
  exit 2
fi

GPU_ID="$1"
KIND="$2"
WORKER_NAME="$3"
RUN_ROOT="$4"
INPUT_ALLOWLIST="$5"
EXPECTED_CASES="$6"
GPU_READY_MAX_USED_MIB="$7"

PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
BENCH_PY=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/bench.py
QUEUE="${RUN_ROOT}/queues/${KIND}.tsv"
CURSOR="${RUN_ROOT}/queues/${KIND}.cursor"
LOCK="${RUN_ROOT}/queues/${KIND}.lock"
LOG="${RUN_ROOT}/logs/${WORKER_NAME}.log"
STATE_DIR="${RUN_ROOT}/state"
SUMMARY_DIR="${RUN_ROOT}/task_summaries"

mkdir -p "$(dirname "${LOG}")" "${STATE_DIR}" "${SUMMARY_DIR}"
exec > >(tee -a "${LOG}") 2>&1

claim_task() {
  local line_number task
  exec 9>"${LOCK}"
  flock 9
  line_number="$(<"${CURSOR}")"
  task="$(sed -n "${line_number}p" "${QUEUE}")"
  if [[ -n "${task}" ]]; then
    printf '%s\n' "$((line_number + 1))" > "${CURSOR}"
  fi
  flock -u 9
  exec 9>&-
  printf '%s' "${task}"
}

wait_for_complete_result_root() {
  local result_root="$1" count=0
  while true; do
    if [[ -d "${result_root}" ]]; then
      count="$(find "${result_root}" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
    else
      count=0
    fi
    if [[ "${count}" -ge "${EXPECTED_CASES}" ]]; then
      return 0
    fi
    echo "[queue-worker] waiting for videos: ${count}/${EXPECTED_CASES} root=${result_root}"
    sleep 60
  done
}

wait_for_gpu_capacity() {
  local used=""
  while true; do
    used="$(nvidia-smi -i "${GPU_ID}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
    if [[ "${used}" =~ ^[0-9]+$ ]] && [[ "${used}" -le "${GPU_READY_MAX_USED_MIB}" ]]; then
      return 0
    fi
    echo "[queue-worker] waiting for GPU${GPU_ID}: used=${used:-unknown} MiB, threshold=${GPU_READY_MAX_USED_MIB} MiB"
    sleep 60
  done
}

echo "[queue-worker] start worker=${WORKER_NAME} kind=${KIND} gpu=${GPU_ID}"
num_done=0
num_failed=0

while true; do
  task="$(claim_task)"
  if [[ -z "${task}" ]]; then
    break
  fi

  IFS=$'\t' read -r task_id metric result_root <<< "${task}"
  summary_path="${SUMMARY_DIR}/${task_id}.json"
  echo "[queue-worker] task=${task_id} metric=${metric} root=${result_root}"

  wait_for_complete_result_root "${result_root}"
  if [[ "${KIND}" == "gpu" ]]; then
    wait_for_gpu_capacity
  fi

  extra_args=()
  if [[ "${metric}" == "wmreward" ]]; then
    extra_args+=(--wmreward-reset-interval 1000000)
  fi

  set +e
  TOKENIZERS_PARALLELISM=false \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/Code_data/Code_try0526 \
  "${PYTHON_BIN}" "${BENCH_PY}" \
    --metric "${metric}" \
    --result-root "${result_root}" \
    --input-json-allowlist "${INPUT_ALLOWLIST}" \
    --output-summary "${summary_path}" \
    "${extra_args[@]}"
  status=$?
  set -e

  if [[ "${status}" -eq 0 ]]; then
    num_done=$((num_done + 1))
    printf '%s\t%s\t%s\t%s\n' "${task_id}" "${metric}" "${result_root}" "${WORKER_NAME}" \
      >> "${RUN_ROOT}/completed_tasks.tsv"
  else
    num_failed=$((num_failed + 1))
    printf '%s\t%s\t%s\t%s\t%s\n' "${task_id}" "${metric}" "${result_root}" "${WORKER_NAME}" "${status}" \
      >> "${RUN_ROOT}/failed_tasks.tsv"
  fi
done

printf 'worker=%s\nkind=%s\ngpu=%s\ndone=%s\nfailed=%s\nfinished_utc=%s\n' \
  "${WORKER_NAME}" "${KIND}" "${GPU_ID}" "${num_done}" "${num_failed}" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATE_DIR}/${WORKER_NAME}.complete"
echo "[queue-worker] finish worker=${WORKER_NAME} done=${num_done} failed=${num_failed}"
