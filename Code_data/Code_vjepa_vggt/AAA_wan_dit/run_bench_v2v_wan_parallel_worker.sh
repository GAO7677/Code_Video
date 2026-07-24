#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 7 ]]; then
  echo "Usage: $0 GPU_ID WORKER_NAME METRICS RUN_ROOT BASELINE_LIST INPUT_ALLOWLIST WAIT_FOR_GPU" >&2
  exit 2
fi

GPU_ID="$1"
WORKER_NAME="$2"
METRICS="$3"
RUN_ROOT="$4"
BASELINE_LIST="$5"
INPUT_ALLOWLIST="$6"
WAIT_FOR_GPU="$7"

BENCH_SH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/bench.sh
LOG_DIR="${RUN_ROOT}/logs"
STATE_DIR="${RUN_ROOT}/state"
LOG_PATH="${LOG_DIR}/${WORKER_NAME}.log"

mkdir -p "${LOG_DIR}" "${STATE_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

if [[ "${WAIT_FOR_GPU}" == "1" ]]; then
  while true; do
    used="$(nvidia-smi -i "${GPU_ID}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    if [[ -n "${used}" && "${used}" -lt 2048 ]]; then
      break
    fi
    echo "[parallel-bench-worker] wait gpu=${GPU_ID} memory_used_mib=${used:-unknown}"
    sleep 60
  done
fi

echo "[parallel-bench-worker] start name=${WORKER_NAME} gpu=${GPU_ID} metrics=${METRICS}"
set +e
TOKENIZERS_PARALLELISM=false \
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
BENCH_CUDA_VISIBLE_DEVICES="${GPU_ID}" \
BENCH_RUN_METRICS=1 \
BENCH_METRICS="${METRICS}" \
BENCH_INPUT_JSON_ALLOWLIST="${INPUT_ALLOWLIST}" \
BENCH_RESULT_DIR="${RUN_ROOT}/worker_summaries/${WORKER_NAME}" \
bash "${BENCH_SH}" "${BASELINE_LIST}"
status=$?
set -e

state_suffix=complete
if [[ "${status}" -ne 0 ]]; then
  state_suffix=failed
fi
printf 'name=%s\ngpu=%s\nmetrics=%s\nstatus=%s\nfinished_utc=%s\n' \
  "${WORKER_NAME}" "${GPU_ID}" "${METRICS}" "${status}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "${STATE_DIR}/${WORKER_NAME}.${state_suffix}"

if [[ "${status}" -eq 0 ]]; then
  echo "[parallel-bench-worker] success name=${WORKER_NAME}"
else
  echo "[parallel-bench-worker] failed name=${WORKER_NAME} status=${status}" >&2
fi
exit "${status}"
