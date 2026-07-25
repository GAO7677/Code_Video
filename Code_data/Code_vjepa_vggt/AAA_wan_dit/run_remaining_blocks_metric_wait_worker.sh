#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 7 ]]; then
  echo "Usage: $0 GPU_ID KIND WORKER_NAME RUN_ROOT INPUT_LIST READY_FILE STATE_PREFIX" >&2
  exit 2
fi

GPU_ID="$1"
KIND="$2"
WORKER_NAME="$3"
RUN_ROOT="$4"
INPUT_LIST="$5"
READY_FILE="$6"
STATE_PREFIX="$7"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

while [[ ! -f "${READY_FILE}" ]]; do
  [[ -f "${RUN_ROOT}/pipeline.failed" ]] && exit 1
  sleep 20
done

bash "${SCRIPT_DIR}/run_remaining_blocks_queue_worker.sh" \
  "${GPU_ID}" "${KIND}" "${WORKER_NAME}" "${RUN_ROOT}/metrics" "${INPUT_LIST}"

touch "${RUN_ROOT}/metrics/state/${STATE_PREFIX}_${WORKER_NAME}.stage_complete"
