#!/usr/bin/env bash
set -uo pipefail

if [[ "$#" -ne 6 ]]; then
  echo "Usage: $0 KIND GPU_ID WORKER_ID RUN_ROOT BATCH_ROOT MAX_USED_MIB" >&2
  exit 2
fi

KIND="$1"
GPU_ID="$2"
WORKER_ID="$3"
RUN_ROOT="$4"
BATCH_ROOT="$5"
MAX_USED_MIB="$6"

PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
BENCH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_phys_papers_compare/bench.py
BASELINE_LIST="${BATCH_ROOT}/result_roots.txt"
QUEUE="${RUN_ROOT}/queues/${KIND}.tsv"
CURSOR="${RUN_ROOT}/queues/${KIND}.cursor"
LOCK="${RUN_ROOT}/queues/${KIND}.lock"
LOG="${RUN_ROOT}/logs/${WORKER_ID}.log"
STATE="${RUN_ROOT}/state"
SUMMARIES="${RUN_ROOT}/task_summaries"

mkdir -p "${RUN_ROOT}/logs" "${STATE}" "${SUMMARIES}" "${RUN_ROOT}/queues"
touch "${LOCK}"
exec > >(tee -a "${LOG}") 2>&1

gpu_used_mib() {
  nvidia-smi -i "${GPU_ID}" --query-gpu=memory.used \
    --format=csv,noheader,nounits | head -1
}

claim_task() {
  local line_number task
  exec 9>"${LOCK}"
  flock 9
  line_number="$(<"${CURSOR}")"
  task="$(sed -n "${line_number}p" "${QUEUE}")"
  if [[ -n "${task}" ]]; then
    printf '%s\n' "$((line_number + 1))" > "${CURSOR}"
  fi
  flock -u 9
  exec 9>&-
  printf '%s' "${task}"
}

echo "[dynamic-worker] start kind=${KIND} gpu=${GPU_ID} worker=${WORKER_ID}"
while true; do
  while true; do
    used="$(gpu_used_mib)"
    [[ "${used}" -le "${MAX_USED_MIB}" ]] && break
    echo "[dynamic-worker] wait gpu=${GPU_ID} used=${used}MiB threshold=${MAX_USED_MIB}"
    sleep 30
  done

  task="$(claim_task)"
  [[ -z "${task}" ]] && break
  IFS=$'\t' read -r task_id metric shard_index num_shards <<< "${task}"
  summary="${SUMMARIES}/${task_id}.json"
  echo "[dynamic-worker] task=${task_id} metric=${metric} shard=${shard_index}/${num_shards}"

  set +e
  TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    "${PYTHON}" "${BENCH}" \
      --metric "${metric}" \
      --baseline-list "${BASELINE_LIST}" \
      --python-bin "${PYTHON}" \
      --num-shards "${num_shards}" \
      --shard-index "${shard_index}" \
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
    printf '%s\t%s\t%s\t%s\n' \
      "${task_id}" "${metric}" "${shard_index}" "${GPU_ID}" \
      >> "${RUN_ROOT}/completed_tasks.tsv"
  else
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "${task_id}" "${metric}" "${shard_index}" "${GPU_ID}" "${status}" \
      >> "${RUN_ROOT}/failed_tasks.tsv"
  fi
done

touch "${STATE}/${WORKER_ID}.worker_complete"
echo "[dynamic-worker] complete kind=${KIND} gpu=${GPU_ID} worker=${WORKER_ID}"
exec bash
