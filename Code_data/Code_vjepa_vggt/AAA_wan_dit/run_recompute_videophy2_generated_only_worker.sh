#!/usr/bin/env bash
set -uo pipefail

if [[ "$#" -ne 9 ]]; then
  echo "Usage: $0 GPU_ID WORKER_NAME RUN_ROOT INPUT_ALLOWLIST EXPECTED_CASES GPU_MAX_USED_MIB START_GATE PRIOR_STATE_DIR PRIOR_WORKERS" >&2
  exit 2
fi

GPU_ID="$1"
WORKER_NAME="$2"
RUN_ROOT="$3"
INPUT_ALLOWLIST="$4"
EXPECTED_CASES="$5"
GPU_MAX_USED_MIB="$6"
START_GATE="$7"
PRIOR_STATE_DIR="$8"
PRIOR_WORKERS="$9"

PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
BENCH_PY=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/bench.py
QUEUE="${RUN_ROOT}/queues/videophy2.tsv"
CURSOR="${RUN_ROOT}/queues/videophy2.cursor"
LOCK="${RUN_ROOT}/queues/videophy2.lock"
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

wait_for_complete_result_root() {
  local result_root="$1" count=0
  while true; do
    count="$(find "${result_root}" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | wc -l)"
    if [[ "${count}" -ge "${EXPECTED_CASES}" ]]; then
      return 0
    fi
    echo "[videophy2-worker] waiting for videos: ${count}/${EXPECTED_CASES} root=${result_root}"
    sleep 60
  done
}

wait_for_gpu_capacity() {
  local used=""
  while true; do
    used="$(nvidia-smi -i "${GPU_ID}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
    if [[ "${used}" =~ ^[0-9]+$ ]] && [[ "${used}" -le "${GPU_MAX_USED_MIB}" ]]; then
      return 0
    fi
    echo "[videophy2-worker] waiting for GPU${GPU_ID}: used=${used:-unknown} MiB threshold=${GPU_MAX_USED_MIB} MiB"
    sleep 60
  done
}

echo "[videophy2-worker] waiting for start gate: ${START_GATE}"
while [[ ! -f "${START_GATE}" ]]; do
  sleep 30
done

num_done=0
num_failed=0
while true; do
  task="$(claim_task)"
  if [[ -z "${task}" ]]; then
    break
  fi
  IFS=$'\t' read -r task_id result_root require_prior_complete <<< "${task}"
  summary_path="${SUMMARY_DIR}/${task_id}.json"
  echo "[videophy2-worker] task=${task_id} gpu=${GPU_ID} root=${result_root}"

  wait_for_complete_result_root "${result_root}"
  if [[ "${require_prior_complete}" == "1" ]]; then
    while true; do
      prior_done="$(find "${PRIOR_STATE_DIR}" -maxdepth 1 -type f -name 'g*_gpu*.complete' 2>/dev/null | wc -l)"
      if [[ "${prior_done}" -ge "${PRIOR_WORKERS}" ]]; then
        break
      fi
      echo "[videophy2-worker] waiting for prior PhyRVG GPU workers: ${prior_done}/${PRIOR_WORKERS}"
      sleep 60
    done
  fi
  wait_for_gpu_capacity

  set +e
  TOKENIZERS_PARALLELISM=false \
  PYTHONNOUSERSITE=1 \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/Code_data/Code_try0526 \
  "${PYTHON_BIN}" "${BENCH_PY}" \
    --metric videophy2 \
    --videophy2-task generated_only_sa_pc_joint \
    --result-root "${result_root}" \
    --input-json-allowlist "${INPUT_ALLOWLIST}" \
    --output-summary "${summary_path}" \
    --overwrite
  status=$?
  set -e

  if [[ "${status}" -eq 0 ]]; then
    num_done=$((num_done + 1))
    printf '%s\t%s\t%s\t%s\n' "${task_id}" "${result_root}" "${WORKER_NAME}" "${GPU_ID}" \
      >> "${RUN_ROOT}/completed_tasks.tsv"
  else
    num_failed=$((num_failed + 1))
    printf '%s\t%s\t%s\t%s\t%s\n' "${task_id}" "${result_root}" "${WORKER_NAME}" "${GPU_ID}" "${status}" \
      >> "${RUN_ROOT}/failed_tasks.tsv"
  fi
done

printf 'worker=%s\ngpu=%s\ndone=%s\nfailed=%s\nfinished_utc=%s\n' \
  "${WORKER_NAME}" "${GPU_ID}" "${num_done}" "${num_failed}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "${STATE_DIR}/${WORKER_NAME}.complete"
echo "[videophy2-worker] finish worker=${WORKER_NAME} done=${num_done} failed=${num_failed}"
