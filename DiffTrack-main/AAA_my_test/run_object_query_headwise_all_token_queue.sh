#!/usr/bin/env bash
set -euo pipefail
GPU="${1:?usage: $0 GPU PROFILE...}"
shift
RUNNER="/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/run_attention_lora_headwise_pck_overlay_gpu.sh"
ROOT="/data/gaoya/agent-data/outputs/object_query_attention_overlay_headwise_pck_case001460_seed090094"
mkdir -p "${ROOT}/logs"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for profile in "$@"; do
  log="${ROOT}/logs/all_token_gpu${GPU}_${profile}.log"
  while ! bash "${RUNNER}" "${GPU}" "${profile}" >> "${log}" 2>&1; do
    printf '[%s] retry profile=%s gpu=%s\n' "$(date -u +%FT%TZ)" "${profile}" "${GPU}" >> "${log}"
    sleep 120
  done
done
