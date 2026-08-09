#!/usr/bin/env bash
set -euo pipefail

worker_id="${1:?usage: $0 WORKER_ID NUM_WORKERS PHYSICAL_GPU}"
num_workers="${2:?usage: $0 WORKER_ID NUM_WORKERS PHYSICAL_GPU}"
physical_gpu="${3:?usage: $0 WORKER_ID NUM_WORKERS PHYSICAL_GPU}"

if [[ "${physical_gpu}" == "4" ]]; then
  echo "GPU 4 is prohibited in this workspace." >&2
  exit 2
fi

runner=/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/run_legacy_object_ablation_other10_6seed_worker.sh
while true; do
  used_mib="$(
    nvidia-smi --id="${physical_gpu}" --query-gpu=memory.used \
      --format=csv,noheader,nounits | tr -d '[:space:]'
  )"
  if [[ "${used_mib}" =~ ^[0-9]+$ ]] && (( used_mib < 1024 )); then
    echo "[$(date -u +%FT%TZ)] GPU${physical_gpu} is free (${used_mib} MiB); starting worker ${worker_id}/${num_workers}"
    exec "${runner}" "${worker_id}" "${num_workers}" "${physical_gpu}"
  fi
  echo "[$(date -u +%FT%TZ)] waiting for GPU${physical_gpu}; memory.used=${used_mib:-unknown} MiB"
  sleep 60
done
