#!/usr/bin/env bash
# Wait for GPU1 to be genuinely idle, then run the two-checkpoint comparison.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLL_SECONDS=20
STABLE_POLLS=3
MAX_USED_MIB=2000
stable=0

while true; do
  used="$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
  if [[ "${used}" =~ ^[0-9]+$ ]] && (( used <= MAX_USED_MIB )); then
    stable=$((stable + 1))
    echo "[$(date -u +%FT%TZ)] GPU1 used=${used} MiB; idle ${stable}/${STABLE_POLLS}"
    if (( stable >= STABLE_POLLS )); then
      break
    fi
  else
    stable=0
    echo "[$(date -u +%FT%TZ)] GPU1 used=${used} MiB; waiting"
  fi
  sleep "${POLL_SECONDS}"
done

echo "[$(date -u +%FT%TZ)] GPU1 admitted; starting step-500 then step-1000"
exec bash "${PROJECT_DIR}/run_full_sa_no_object_xssc_train_cases_gpu1.sh"
