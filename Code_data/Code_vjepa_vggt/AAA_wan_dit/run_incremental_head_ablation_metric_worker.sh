#!/usr/bin/env bash
set -uo pipefail

if [[ "$#" -ne 6 ]]; then
  echo "Usage: $0 GPU_ID KIND WORKER_NAME RUN_ROOT INPUT_ALLOWLIST MAX_GPU_USED_MIB" >&2
  exit 2
fi

GPU_ID="$1"
KIND="$2"
WORKER_NAME="$3"
RUN_ROOT="$4"
INPUT_ALLOWLIST="$5"
MAX_GPU_USED_MIB="$6"

PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
BENCH_PY=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/bench.py
QUEUE="${RUN_ROOT}/queues/${KIND}.tsv"
CURSOR="${RUN_ROOT}/queues/${KIND}.cursor"
QUEUE_LOCK="${RUN_ROOT}/queues/${KIND}.lock"
GPU_LOCK="${RUN_ROOT}/gpu${GPU_ID}.metric.lock"
LOG="${RUN_ROOT}/logs/${WORKER_NAME}.log"
STATE_DIR="${RUN_ROOT}/state"
SUMMARY_DIR="${RUN_ROOT}/task_summaries"

mkdir -p "$(dirname "${LOG}")" "${STATE_DIR}" "${SUMMARY_DIR}" \
  "${RUN_ROOT}/queues"
touch "${QUEUE}" "${QUEUE_LOCK}" "${RUN_ROOT}/completed_tasks.tsv" \
  "${RUN_ROOT}/failed_tasks.tsv"
[[ -s "${CURSOR}" ]] || printf '1\n' > "${CURSOR}"
exec > >(tee -a "${LOG}") 2>&1

claim_task() {
  local line_number task
  exec 9>"${QUEUE_LOCK}"
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

append_retry() {
  local line="$1"
  exec 9>"${QUEUE_LOCK}"
  flock 9
  printf '%s\n' "${line}" >> "${QUEUE}"
  flock -u 9
  exec 9>&-
}

gpu_used_mib() {
  nvidia-smi -i "${GPU_ID}" --query-gpu=memory.used \
    --format=csv,noheader,nounits | head -n 1
}

run_metric() {
  local metric="$1"
  local result_root="$2"
  local summary_path="$3"
  local status
  local -a extra_args=()
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
  if [[ "${status}" -ne 0 ]]; then
    return "${status}"
  fi
  "${PYTHON_BIN}" - "${summary_path}" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
status = payload.get("metric_status", {})
if (
    int(status.get("num_cases", -1)) != 20
    or int(status.get("num_success", -1)) != 20
    or int(status.get("num_failed", -1)) != 0
):
    raise SystemExit(f"incomplete metric summary: {status}")
PY
}

echo "[incremental-worker] start worker=${WORKER_NAME} kind=${KIND} gpu=${GPU_ID}"
num_done=0
num_failed=0

while true; do
  task="$(claim_task)"
  if [[ -z "${task}" ]]; then
    if [[ -f "${RUN_ROOT}/enqueue.complete" ]]; then
      break
    fi
    sleep 20
    continue
  fi

  IFS=$'\t' read -r task_id metric result_root attempt <<< "${task}"
  summary_path="${SUMMARY_DIR}/${task_id}.json"
  echo "[incremental-worker] task=${task_id} metric=${metric} attempt=${attempt}"

  if [[ "${KIND}" == "cpu" ]]; then
    run_metric "${metric}" "${result_root}" "${summary_path}"
    status=$?
  else
    status=1
    while true; do
      used="$(gpu_used_mib)"
      if [[ "${used}" -gt "${MAX_GPU_USED_MIB}" ]]; then
        echo "[incremental-worker] GPU${GPU_ID} used=${used}MiB > ${MAX_GPU_USED_MIB}; wait"
        sleep 30
        continue
      fi
      exec 8>"${GPU_LOCK}"
      flock 8
      used="$(gpu_used_mib)"
      if [[ "${used}" -gt "${MAX_GPU_USED_MIB}" ]]; then
        flock -u 8
        exec 8>&-
        sleep 30
        continue
      fi
      run_metric "${metric}" "${result_root}" "${summary_path}"
      status=$?
      flock -u 8
      exec 8>&-
      break
    done
  fi

  if [[ "${status}" -eq 0 ]]; then
    num_done=$((num_done + 1))
    printf '%s\t%s\t%s\t%s\n' "${task_id}" "${metric}" "${result_root}" \
      "${WORKER_NAME}" >> "${RUN_ROOT}/completed_tasks.tsv"
  else
    num_failed=$((num_failed + 1))
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${task_id}" "${metric}" \
      "${result_root}" "${WORKER_NAME}" "${status}" "${attempt}" \
      >> "${RUN_ROOT}/failed_tasks.tsv"
    if [[ "${attempt}" -lt 2 ]]; then
      retry=$((attempt + 1))
      append_retry "${task_id}-retry${retry}"$'\t'"${metric}"$'\t'"${result_root}"$'\t'"${retry}"
    fi
  fi
done

printf 'worker=%s\nkind=%s\ngpu=%s\ndone=%s\nfailed_attempts=%s\nfinished_utc=%s\n' \
  "${WORKER_NAME}" "${KIND}" "${GPU_ID}" "${num_done}" "${num_failed}" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATE_DIR}/${WORKER_NAME}.complete"
echo "[incremental-worker] finish worker=${WORKER_NAME} done=${num_done} failed=${num_failed}"
