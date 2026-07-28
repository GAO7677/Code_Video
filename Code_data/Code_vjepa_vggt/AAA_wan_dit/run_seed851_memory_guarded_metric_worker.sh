#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 6 ]]; then
  echo "Usage: $0 KIND GPU_ID WORKER_ID RUN_ROOT BATCH_ROOT MAX_USED_MIB" >&2
  exit 2
fi

WORKER=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_stc_bench_dynamic_worker.sh
MIN_AVAILABLE_KIB=$((96 * 1024 * 1024))

worker_threads="${METRIC_WORKER_THREADS:-4}"
export OMP_NUM_THREADS="${worker_threads}"
export MKL_NUM_THREADS="${worker_threads}"
export OPENBLAS_NUM_THREADS="${worker_threads}"
export NUMEXPR_NUM_THREADS="${worker_threads}"
export MALLOC_ARENA_MAX=2
export TOKENIZERS_PARALLELISM=false

while true; do
  available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
  [[ "${available_kib}" -ge "${MIN_AVAILABLE_KIB}" ]] && break
  echo "[memory-guarded-worker] waiting: MemAvailable=${available_kib} KiB"
  sleep 30
done

exec bash "${WORKER}" "$@"
