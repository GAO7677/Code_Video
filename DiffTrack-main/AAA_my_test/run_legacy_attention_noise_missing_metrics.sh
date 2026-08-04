#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 GPU_ID METRIC [METRIC ...]" >&2
  exit 2
fi

GPU_ID="$1"
shift
METRICS=("$@")

PROJECT_ROOT="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt"
TRY0526_ROOT="/home/gaoya/Code_Video/Code_data/Code_try0526"
PYTHON_BIN="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
BENCH_PY="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_phys_papers_compare/bench.py"
BENCH_ROOT="/data/gaoya/agent-data/outputs/attention_probability_noise_metrics_test5"
METHODS_FILE="${BENCH_ROOT}/bench_legacy_baseline_lora_methods.txt"
ALLOWLIST="${BENCH_ROOT}/input_json_allowlist.txt"
STATUS_DIR="${BENCH_ROOT}/status"
LOG_DIR="${BENCH_ROOT}/logs/legacy_missing"

mkdir -p "${STATUS_DIR}" "${LOG_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${PROJECT_ROOT}:${TRY0526_ROOT}:${PYTHONPATH:-}"
export PYTHON_BIN

for metric in "${METRICS[@]}"; do
  rm -f "${STATUS_DIR}/${metric}.failed" "${STATUS_DIR}/${metric}.done"
  printf 'gpu=%s\nstarted=%s\n' "${GPU_ID}" "$(date -u +%FT%TZ)" \
    > "${STATUS_DIR}/${metric}.running"
  if "${PYTHON_BIN}" "${BENCH_PY}" \
      --metric "${metric}" \
      --baseline-list "${METHODS_FILE}" \
      --input-json-allowlist "${ALLOWLIST}" \
      >> "${LOG_DIR}/${metric}.log" 2>&1; then
    rm -f "${STATUS_DIR}/${metric}.running"
    printf 'gpu=%s\ncompleted=%s\n' "${GPU_ID}" "$(date -u +%FT%TZ)" \
      > "${STATUS_DIR}/${metric}.done"
  else
    code=$?
    rm -f "${STATUS_DIR}/${metric}.running"
    printf 'gpu=%s\nfailed=%s\nexit_code=%s\n' \
      "${GPU_ID}" "$(date -u +%FT%TZ)" "${code}" \
      > "${STATUS_DIR}/${metric}.failed"
    exit "${code}"
  fi
done

echo "Legacy missing metrics complete on GPU ${GPU_ID}: ${METRICS[*]}"
