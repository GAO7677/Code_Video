#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 GPU_ID METRIC [METRIC ...]" >&2
  exit 2
fi

GPU_ID="$1"
shift
METRICS=("$@")

BENCH_ROOT="/data/gaoya/agent-data/outputs/attention_probability_noise_metrics_test5"
BENCH_DIR="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box"
METHODS_FILE="${BENCH_ROOT}/bench_methods.txt"
ALLOWLIST="${BENCH_ROOT}/input_json_allowlist.txt"
STATUS_DIR="${BENCH_ROOT}/status"
LOG_DIR="${BENCH_ROOT}/logs"

mkdir -p "${STATUS_DIR}" "${LOG_DIR}" "${BENCH_ROOT}/summaries/gpu${GPU_ID}"
while [[ ! -f "${BENCH_ROOT}/PREPARED" ]]; do
  if [[ -f "${BENCH_ROOT}/PREPARE_FAILED" ]]; then
    echo "Preparation failed; GPU ${GPU_ID} metric worker is stopping." >&2
    exit 1
  fi
  sleep 30
done

cd "${BENCH_DIR}"
export BENCH_RUN_METRICS=1
export BENCH_CUDA_VISIBLE_DEVICES="${GPU_ID}"
export BENCH_INPUT_JSON_ALLOWLIST="${ALLOWLIST}"
export BENCH_RESULT_DIR="${BENCH_ROOT}/summaries/gpu${GPU_ID}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

for metric in "${METRICS[@]}"; do
  rm -f "${STATUS_DIR}/${metric}.failed"
  printf 'gpu=%s\nstarted=%s\n' "${GPU_ID}" "$(date -u +%FT%TZ)" \
    > "${STATUS_DIR}/${metric}.running"
  export BENCH_METRICS_RAW="${metric}"
  if bash bench.sh "${METHODS_FILE}" \
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

echo "GPU ${GPU_ID} metric shard complete: ${METRICS[*]}"
