#!/usr/bin/env bash
set -uo pipefail

CONFIG="$(realpath "$1")"
# shellcheck source=/dev/null
source "${CONFIG}"

resume_workers() {
  if [[ -s "${PRIORITY_RUN_ROOT}/paused_worker_pids.txt" ]]; then
    while read -r pid; do
      if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        kill -CONT "${pid}"
        echo "[priority-coordinator] resumed pid=${pid}"
      fi
    done < "${PRIORITY_RUN_ROOT}/paused_worker_pids.txt"
  fi
}
trap resume_workers EXIT

while true; do
  complete="$(find "${PRIORITY_RUN_ROOT}/task_state" -maxdepth 1 \
    -name 'priority-*.complete' -type f | wc -l)"
  failed="$(find "${PRIORITY_RUN_ROOT}/task_state" -maxdepth 1 \
    -name '*.failed' -type f | wc -l)"
  workers="$(find "${PRIORITY_RUN_ROOT}/task_state" -maxdepth 1 \
    -name '*.worker.complete' -type f | wc -l)"
  echo "[priority-coordinator] tasks=${complete}/${EXPECTED_PRIORITY_TASKS} failed=${failed} workers=${workers}/2"
  if [[ "${complete}" -eq "${EXPECTED_PRIORITY_TASKS}" ]]; then
    touch "${PRIORITY_RUN_ROOT}/priority.complete"
    python3 "${GALLERY_SCRIPT}" --root "${OUTPUT_BASE}" --build-only
    exit 0
  fi
  if [[ "${workers}" -eq 2 ]]; then
    touch "${PRIORITY_RUN_ROOT}/priority.failed"
    exit 1
  fi
  sleep 10
done
