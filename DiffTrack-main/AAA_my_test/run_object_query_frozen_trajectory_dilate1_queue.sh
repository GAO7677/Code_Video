#!/usr/bin/env bash
set -u -o pipefail

GPU="${1:?usage: $0 GPU SEED...}"; shift
ROOT="/data/gaoya/agent-data/outputs/attention_lora_object_query_frozen_trajectory_case001460"
mkdir -p "${ROOT}/logs"
for seed in "$@"; do
  sid=$(printf '%06d' "${seed}")
  while [[ ! -f "${ROOT}/seeds/seed_${sid}/single_p95_p99_complete" ]]; do
    printf '%s gpu=%s seed=%s waiting_for_single\n' "$(date -u +%FT%TZ)" "${GPU}" "${seed}"
    sleep 30
  done
done
cd /home/gaoya/Code_Video/DiffTrack-main
for seed in "$@"; do
  [[ -f "${ROOT}/seeds/seed_$(printf '%06d' "${seed}")/dilate1_p95_p99_complete" ]] && continue
  log="${ROOT}/logs/dilate1_gpu${GPU}_seed${seed}.log"; success=0
  for attempt in 1 2 3; do
    printf '%s gpu=%s seed=%s attempt=%s start_dilate1\n' "$(date -u +%FT%TZ)" "${GPU}" "${seed}" "${attempt}" | tee -a "${log}"
    if ./AAA_my_test/run_object_query_frozen_trajectory_dilate1_seed_gpu.sh "${GPU}" "${seed}" 2>&1 | tee -a "${log}"; then success=1; break; fi
    sleep 60
  done
  [[ "${success}" == 1 ]] || exit 1
done
