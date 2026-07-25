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

exec bash "${SCRIPT_DIR}/run_bench_v2v_wan_queue_worker.sh" \
  "${GPU_ID}" "${KIND}" "${WORKER_NAME}" "${RUN_ROOT}/metrics" "${INPUT_LIST}"
