#!/usr/bin/env bash
set -uo pipefail

if [[ "$#" -ne 6 ]]; then
  echo "Usage: $0 GPU_ID SHARD_INDEX NUM_SHARDS BATCH_ROOT RUN_ROOT MAX_USED_MIB" >&2
  exit 2
fi

GPU_ID="$1"
SHARD_INDEX="$2"
NUM_SHARDS="$3"
BATCH_ROOT="$4"
RUN_ROOT="$5"
MAX_USED_MIB="$6"

PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
BENCH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_phys_papers_compare/bench.py
BASELINE_LIST="${BATCH_ROOT}/result_roots.txt"
WORKER_NAME="gpu${GPU_ID}_shard${SHARD_INDEX}"
LOG="${RUN_ROOT}/logs/${WORKER_NAME}.log"
STATUS_ROOT="${RUN_ROOT}/status"
SUMMARY_ROOT="${RUN_ROOT}/task_summaries"
METRICS=(
  vbench_subject_consistency
  vbench_background_consistency
  vbench_temporal_flickering
  vbench_motion_smoothness
  vbench_dynamic_degree
  vbench_aesthetic_quality
  vbench_imaging_quality
  videophy2
  cosmos_reason1
)

mkdir -p "${RUN_ROOT}/logs" "${STATUS_ROOT}" "${SUMMARY_ROOT}"
exec > >(tee -a "${LOG}") 2>&1

gpu_used_mib() {
  nvidia-smi -i "${GPU_ID}" --query-gpu=memory.used \
    --format=csv,noheader,nounits | head -1
}

for metric in "${METRICS[@]}"; do
  marker="${STATUS_ROOT}/${metric}.shard${SHARD_INDEX}"
  if [[ -f "${marker}.complete" ]]; then
    continue
  fi
  while true; do
    used="$(gpu_used_mib)"
    if [[ "${used}" -le "${MAX_USED_MIB}" ]]; then
      break
    fi
    echo "[parallel-shard] wait gpu=${GPU_ID} used=${used}MiB metric=${metric}"
    sleep 30
  done

  summary="${SUMMARY_ROOT}/${metric}.shard${SHARD_INDEX}.json"
  echo "[parallel-shard] start gpu=${GPU_ID} shard=${SHARD_INDEX}/${NUM_SHARDS} metric=${metric}"
  set +e
  TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${PYTHON}" "${BENCH}" \
      --metric "${metric}" \
      --baseline-list "${BASELINE_LIST}" \
      --python-bin "${PYTHON}" \
      --num-shards "${NUM_SHARDS}" \
      --shard-index "${SHARD_INDEX}" \
      --output-summary "${summary}"
  status=$?
  set -e

  if [[ "${status}" -eq 0 ]]; then
    set +e
    "${PYTHON}" - "${summary}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
status = payload.get("metric_status") or {}
num_cases = int(status.get("num_cases", -1))
num_success = int(status.get("num_success", -1))
num_failed = int(status.get("num_failed", -1))
completed = int(status.get("completed", -1))
if num_cases <= 0 or num_success != num_cases or num_failed != 0 or completed != num_cases:
    raise SystemExit(
        f"incomplete summary: cases={num_cases} success={num_success} "
        f"failed={num_failed} completed={completed}"
    )
PY
    status=$?
    set -e
  fi

  if [[ "${status}" -eq 0 ]]; then
    touch "${marker}.complete"
  else
    printf '%s\n' "${status}" > "${marker}.failed"
  fi
  echo "[parallel-shard] finish gpu=${GPU_ID} shard=${SHARD_INDEX} metric=${metric} status=${status}"
done

touch "${STATUS_ROOT}/${WORKER_NAME}.worker_complete"
exec bash
