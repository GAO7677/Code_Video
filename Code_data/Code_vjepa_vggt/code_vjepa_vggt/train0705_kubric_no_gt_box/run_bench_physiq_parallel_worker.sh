#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 5 ]]; then
  echo "Usage: $0 GPU_ID WORKER_NAME METRICS RUN_ROOT BASELINE_LIST" >&2
  exit 2
fi

GPU_ID="$1"
WORKER_NAME="$2"
METRICS="$3"
RUN_ROOT="$4"
BASELINE_LIST="$5"
BENCH_SH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/bench.sh
LOG_DIR="${RUN_ROOT}/logs"
STATE_DIR="${RUN_ROOT}/state"
LOG_PATH="${LOG_DIR}/${WORKER_NAME}.log"

mkdir -p "${LOG_DIR}" "${STATE_DIR}"
rm -f "${STATE_DIR}/${WORKER_NAME}.complete" "${STATE_DIR}/${WORKER_NAME}.failed"
exec > >(tee -a "${LOG_PATH}") 2>&1

echo "[parallel-bench-worker] start name=${WORKER_NAME} gpu=${GPU_ID} metrics=${METRICS}"
set +e
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
BENCH_CUDA_VISIBLE_DEVICES="${GPU_ID}" \
BENCH_METRICS="${METRICS}" \
bash "${BENCH_SH}" "${BASELINE_LIST}"
status=$?
set -e

if [[ "${status}" -eq 0 ]]; then
  printf 'name=%s\ngpu=%s\nmetrics=%s\nstatus=0\nfinished_utc=%s\n' \
    "${WORKER_NAME}" "${GPU_ID}" "${METRICS}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${STATE_DIR}/${WORKER_NAME}.complete"
  echo "[parallel-bench-worker] success name=${WORKER_NAME}"
else
  printf 'name=%s\ngpu=%s\nmetrics=%s\nstatus=%s\nfinished_utc=%s\n' \
    "${WORKER_NAME}" "${GPU_ID}" "${METRICS}" "${status}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > "${STATE_DIR}/${WORKER_NAME}.failed"
  echo "[parallel-bench-worker] failed name=${WORKER_NAME} status=${status}" >&2
fi
exit "${status}"
