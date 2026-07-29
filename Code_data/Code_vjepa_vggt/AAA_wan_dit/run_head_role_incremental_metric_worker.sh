#!/usr/bin/env bash
set -uo pipefail

if [[ "$#" -ne 7 ]]; then
  echo "Usage: $0 GPU KIND WORKER RUN_ROOT INPUT_LIST MIN_FREE_MIB COOLDOWN_SEC" >&2
  exit 2
fi

GPU_ID="$1"
KIND="$2"
WORKER_NAME="$3"
RUN_ROOT="$4"
INPUT_ALLOWLIST="$5"
MIN_FREE_MIB="$6"
COOLDOWN_SEC="$7"
PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
BENCH_PY=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/bench.py
QUEUE="${RUN_ROOT}/queues/${KIND}.tsv"
CURSOR="${RUN_ROOT}/queues/${KIND}.cursor"
LOCK="${RUN_ROOT}/queues/${KIND}.lock"
LOG="${RUN_ROOT}/logs/${WORKER_NAME}.log"
STATE_DIR="${RUN_ROOT}/state"
SUMMARY_DIR="${RUN_ROOT}/task_summaries"
STOP="${RUN_ROOT}/stop"

mkdir -p "$(dirname "${LOG}")" "${STATE_DIR}" "${SUMMARY_DIR}"
exec > >(tee -a "${LOG}") 2>&1

num_done=0
num_failed=0
write_state() {
  printf 'worker=%s\nkind=%s\ngpu=%s\ndone=%s\nfailed=%s\nfinished_utc=%s\n' \
    "${WORKER_NAME}" "${KIND}" "${GPU_ID}" "${num_done}" "${num_failed}" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${STATE_DIR}/${WORKER_NAME}.complete"
}
trap write_state EXIT

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

wait_for_memory() {
  [[ "${KIND}" == "cpu" ]] && return 0
  while [[ ! -f "${STOP}" ]]; do
    free="$(nvidia-smi --id="${GPU_ID}" --query-gpu=memory.free --format=csv,noheader,nounits)"
    if (( free >= MIN_FREE_MIB )); then
      return 0
    fi
    echo "[incremental-metric] GPU${GPU_ID} free=${free}MiB; waiting"
    sleep 20
  done
  return 1
}

echo "[incremental-metric] start worker=${WORKER_NAME} kind=${KIND} gpu=${GPU_ID}"
while [[ ! -f "${STOP}" ]]; do
  wait_for_memory || break
  task="$(claim_task)"
  [[ -z "${task}" ]] && break
  IFS=$'\t' read -r task_id metric result_root <<< "${task}"
  summary_path="${SUMMARY_DIR}/${task_id}.json"
  extra_args=()
  [[ "${metric}" == "wmreward" ]] && extra_args+=(--wmreward-reset-interval 1000000)
  echo "[incremental-metric] task=${task_id} metric=${metric}"
  set +e
  TOKENIZERS_PARALLELISM=false \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/Code_data/Code_try0526 \
  "${PYTHON_BIN}" "${BENCH_PY}" \
    --metric "${metric}" \
    --result-root "${result_root}" \
    --input-json-allowlist "${INPUT_ALLOWLIST}" \
    --output-summary "${summary_path}" \
    "${extra_args[@]}"
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
  if (( COOLDOWN_SEC > 0 )); then
    for ((second=0; second<COOLDOWN_SEC; second+=5)); do
      [[ -f "${STOP}" ]] && break
      sleep 5
    done
  fi
done
echo "[incremental-metric] finish worker=${WORKER_NAME} done=${num_done} failed=${num_failed}"
