#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 6 ]]; then
  echo "Usage: $0 GPU_ID KIND WORKER_NAME RUN_ROOT INPUT_LIST READY_FILE" >&2
  exit 2
fi

GPU_ID="$1"
KIND="$2"
WORKER_NAME="$3"
RUN_ROOT="$4"
INPUT_LIST="$5"
READY_FILE="$6"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

while [[ ! -f "${READY_FILE}" ]]; do
  if [[ -f "${RUN_ROOT}/generation.failed" ]]; then
    echo "[metric-wait] generation failed; worker=${WORKER_NAME} exits"
    exit 1
  fi
  sleep 20
done

if [[ "${KIND}" != "cpu" ]]; then
  min_free_mib="${METRIC_MIN_FREE_MIB:-40000}"
  queue="${RUN_ROOT}/metrics/queues/${KIND}.tsv"
  cursor="${RUN_ROOT}/metrics/queues/${KIND}.cursor"
  while true; do
    next_line="$(<"${cursor}")"
    total_lines="$(wc -l < "${queue}")"
    if (( next_line > total_lines )); then
      state_dir="${RUN_ROOT}/metrics/state"
      mkdir -p "${state_dir}"
      printf 'worker=%s\nkind=%s\ngpu=%s\ndone=0\nfailed=0\nfinished_utc=%s\n' \
        "${WORKER_NAME}" "${KIND}" "${GPU_ID}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        > "${state_dir}/${WORKER_NAME}.complete"
      echo "[metric-wait] queue already drained; worker=${WORKER_NAME} exits"
      exit 0
    fi
    free_mib="$(nvidia-smi --id="${GPU_ID}" --query-gpu=memory.free --format=csv,noheader,nounits)"
    if (( free_mib >= min_free_mib )); then
      break
    fi
    echo "[metric-wait] GPU${GPU_ID} free=${free_mib}MiB; waiting for ${min_free_mib}MiB"
    sleep 20
  done
fi

exec bash "${SCRIPT_DIR}/run_bench_v2v_wan_queue_worker.sh" \
  "${GPU_ID}" "${KIND}" "${WORKER_NAME}" "${RUN_ROOT}/metrics" "${INPUT_LIST}"
