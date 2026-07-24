#!/usr/bin/env bash
set -uo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "Usage: $0 WORKER_NAME RUN_ROOT INPUT_ALLOWLIST START_GATE" >&2
  exit 2
fi

WORKER_NAME="$1"
RUN_ROOT="$2"
INPUT_ALLOWLIST="$3"
START_GATE="$4"

PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
BENCH_PY=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/bench.py
QUEUE="${RUN_ROOT}/queues/pmf.tsv"
CURSOR="${RUN_ROOT}/queues/pmf.cursor"
LOCK="${RUN_ROOT}/queues/pmf.lock"
LOG="${RUN_ROOT}/logs/${WORKER_NAME}.log"
STATE_DIR="${RUN_ROOT}/state"
SUMMARY_DIR="${RUN_ROOT}/task_summaries"

mkdir -p "$(dirname "${LOG}")" "${STATE_DIR}" "${SUMMARY_DIR}"
exec > >(tee -a "${LOG}") 2>&1

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

echo "[pmf-worker] waiting for start gate: ${START_GATE}"
while [[ ! -f "${START_GATE}" ]]; do
  sleep 60
done

echo "[pmf-worker] start worker=${WORKER_NAME}"
num_done=0
num_failed=0
while true; do
  task="$(claim_task)"
  if [[ -z "${task}" ]]; then
    break
  fi

  IFS=$'\t' read -r task_id metric result_root <<< "${task}"
  summary_path="${SUMMARY_DIR}/${task_id}.json"
  echo "[pmf-worker] task=${task_id} metric=${metric} root=${result_root}"

  set +e
  TOKENIZERS_PARALLELISM=false \
  OMP_NUM_THREADS=2 \
  MKL_NUM_THREADS=2 \
  PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/Code_data/Code_try0526 \
  "${PYTHON_BIN}" "${BENCH_PY}" \
    --metric "${metric}" \
    --result-root "${result_root}" \
    --input-json-allowlist "${INPUT_ALLOWLIST}" \
    --output-summary "${summary_path}" \
    --overwrite
  status=$?
  set -e

  if [[ "${status}" -eq 0 ]]; then
    num_done=$((num_done + 1))
    printf '%s\t%s\t%s\t%s\n' "${task_id}" "${metric}" "${result_root}" "${WORKER_NAME}" \
      >> "${RUN_ROOT}/completed_tasks.tsv"
  else
    num_failed=$((num_failed + 1))
    printf '%s\t%s\t%s\t%s\t%s\n' "${task_id}" "${metric}" "${result_root}" "${WORKER_NAME}" "${status}" \
      >> "${RUN_ROOT}/failed_tasks.tsv"
  fi
done

printf 'worker=%s\ndone=%s\nfailed=%s\nfinished_utc=%s\n' \
  "${WORKER_NAME}" "${num_done}" "${num_failed}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "${STATE_DIR}/${WORKER_NAME}.complete"
echo "[pmf-worker] finish worker=${WORKER_NAME} done=${num_done} failed=${num_failed}"
