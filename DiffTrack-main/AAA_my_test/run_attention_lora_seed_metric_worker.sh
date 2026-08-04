#!/usr/bin/env bash
set -u

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 GPU_ID" >&2
  exit 2
fi

GPU="$1"
ROOT="/data/gaoya/agent-data/outputs/attention_lora_seed_sweep_metrics_case001460"
BENCH="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/bench.sh"
METHODS="${ROOT}/bench_methods_gpu${GPU}.txt"
ALLOWLIST="${ROOT}/input_json_allowlist.txt"
LOG="${ROOT}/logs/gpu${GPU}.log"
ROUND=0

mkdir -p "${ROOT}/logs" "${ROOT}/status" "${ROOT}/summaries/gpu${GPU}"
while true; do
  while [[ ! -s "${METHODS}" || ! -f "${ROOT}/PREPARED" ]]; do sleep 15; done
  ROUND=$((ROUND + 1))
  printf 'gpu=%s\nround=%s\nstarted=%s\n' "${GPU}" "${ROUND}" "$(date -u +%FT%TZ)" \
    > "${ROOT}/status/gpu${GPU}.running"
  if BENCH_RUN_METRICS=1 \
     BENCH_CUDA_VISIBLE_DEVICES="${GPU}" \
     BENCH_INPUT_JSON_ALLOWLIST="${ALLOWLIST}" \
     BENCH_RESULT_DIR="${ROOT}/summaries/gpu${GPU}" \
     CUDA_VISIBLE_DEVICES="${GPU}" \
     bash "${BENCH}" "${METHODS}" >> "${LOG}" 2>&1; then
    printf 'gpu=%s\nround=%s\ncompleted=%s\n' "${GPU}" "${ROUND}" "$(date -u +%FT%TZ)" \
      > "${ROOT}/status/gpu${GPU}.round_complete"
    rm -f "${ROOT}/status/gpu${GPU}.failed"
    sleep 60
  else
    code=$?
    printf 'gpu=%s\nround=%s\nexit_code=%s\nfailed=%s\n' \
      "${GPU}" "${ROUND}" "${code}" "$(date -u +%FT%TZ)" \
      > "${ROOT}/status/gpu${GPU}.failed"
    sleep 300
  fi
done
