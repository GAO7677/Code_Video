#!/usr/bin/env bash
set -uo pipefail

if [[ "$#" -lt 5 ]]; then
  echo "Usage: $0 WORKER_NAME GPU_ID BATCH_ROOT RUN_ROOT METRIC..." >&2
  exit 2
fi

WORKER_NAME="$1"
GPU_ID="$2"
BATCH_ROOT="$3"
RUN_ROOT="$4"
shift 4
METRICS=("$@")

PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
BENCH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_phys_papers_compare/bench.py
BASELINE_LIST="${BATCH_ROOT}/result_roots.txt"
LOG="${RUN_ROOT}/logs/${WORKER_NAME}.log"
STATUS_ROOT="${RUN_ROOT}/status"

mkdir -p "${RUN_ROOT}/logs" "${STATUS_ROOT}"
exec > >(tee -a "${LOG}") 2>&1

for metric in "${METRICS[@]}"; do
  while true; do
    used="$(nvidia-smi -i "${GPU_ID}" --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"
    [[ "${used}" -le 2048 ]] && break
    echo "[stc-bench-worker] wait GPU${GPU_ID}: used=${used}MiB metric=${metric}"
    sleep 30
  done
  echo "[stc-bench-worker] start worker=${WORKER_NAME} gpu=${GPU_ID} metric=${metric}"
  set +e
  TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${PYTHON}" "${BENCH}" \
      --metric "${metric}" \
      --baseline-list "${BASELINE_LIST}" \
      --python-bin "${PYTHON}" \
      --wmreward-reset-interval 1000000
  status=$?
  set -e
  if [[ "${status}" -eq 0 ]]; then
    touch "${STATUS_ROOT}/${metric}.complete"
  else
    printf '%s\n' "${status}" > "${STATUS_ROOT}/${metric}.failed"
  fi
  echo "[stc-bench-worker] finish metric=${metric} status=${status}"
done

touch "${STATUS_ROOT}/${WORKER_NAME}.worker_complete"
exec bash
