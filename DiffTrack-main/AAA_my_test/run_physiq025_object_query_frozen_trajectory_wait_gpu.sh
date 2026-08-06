#!/usr/bin/env bash
set -euo pipefail

SEED="${1:?usage: $0 SEED}"
DIFFTRACK="/home/gaoya/Code_Video/DiffTrack-main"
ROOT="/data/gaoya/agent-data/outputs/attention_lora_object_query_frozen_trajectory_physiq025_2seed"
GPU_CANDIDATES=(0 1 2 3 5 6 7)
mkdir -p "${ROOT}/logs" "${ROOT}/locks"

while true; do
  for GPU in "${GPU_CANDIDATES[@]}"; do
    exec {LOCK_FD}>"${ROOT}/locks/gpu_${GPU}.lock"
    if ! flock -n "${LOCK_FD}"; then
      exec {LOCK_FD}>&-
      continue
    fi
    USED=$(nvidia-smi --id="${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if [[ "${USED}" =~ ^[0-9]+$ ]] && (( USED <= 4000 )); then
      printf 'seed=%s selected_gpu=%s memory_used_mib=%s started=%s\n' \
        "${SEED}" "${GPU}" "${USED}" "$(date -u +%FT%TZ)" \
        | tee -a "${ROOT}/logs/scheduler.log"
      cd "${DIFFTRACK}"
      exec ./AAA_my_test/run_physiq025_object_query_frozen_trajectory_seed_gpu.sh \
        "${GPU}" "${SEED}"
    fi
    flock -u "${LOCK_FD}"
    exec {LOCK_FD}>&-
  done
  printf 'seed=%s waiting_for_gpu=%s checked=%s\n' \
    "${SEED}" "${GPU_CANDIDATES[*]}" "$(date -u +%FT%TZ)"
  sleep 20
done
