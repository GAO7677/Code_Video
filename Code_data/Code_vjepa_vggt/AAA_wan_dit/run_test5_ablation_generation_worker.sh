#!/usr/bin/env bash
set -uo pipefail

if [[ "$#" -ne 5 ]]; then
  echo "Usage: $0 GPU_ID WORKER_NAME RUN_ROOT OUTPUT_BASE INPUT_LIST" >&2
  exit 2
fi

GPU_ID="$1"
WORKER_NAME="$2"
RUN_ROOT="$3"
OUTPUT_BASE="$4"
INPUT_LIST="$5"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUEUE="${RUN_ROOT}/generation/queue.tsv"
CURSOR="${RUN_ROOT}/generation/cursor"
LOCK="${RUN_ROOT}/generation/queue.lock"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
VERIFY="${SCRIPT_DIR}/verify_test5_ablation_outputs.py"
WORKER_LOG="${RUN_ROOT}/generation/logs/${WORKER_NAME}.log"

mkdir -p "${RUN_ROOT}/generation/logs" "${RUN_ROOT}/generation/state" \
  "${RUN_ROOT}/generation/validations"
exec > >(tee -a "${WORKER_LOG}") 2>&1

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

tag_for() {
  local mode="$1"
  local block="$2"
  if [[ "${mode}" == "baseline" ]]; then
    printf 'baseline'
  else
    printf '%s_block%02d' "${mode}" "$((10#${block}))"
  fi
}

echo "[generation-worker] start worker=${WORKER_NAME} gpu=${GPU_ID}"
num_done=0
num_failed=0

while true; do
  task="$(claim_task)"
  if [[ -z "${task}" ]]; then
    break
  fi
  IFS=$'\t' read -r task_id model mode block <<< "${task}"
  tag="$(tag_for "${mode}" "${block}")"
  task_log="${RUN_ROOT}/generation/logs/${task_id}.log"
  validation="${RUN_ROOT}/generation/validations/${task_id}.json"
  case "${model}" in
    wan_lora|xssc)
      config_root="${OUTPUT_BASE}/${model}/${tag}"
      command=(
        env INPUT_LIST="${INPUT_LIST}" OUTPUT_BASE="${OUTPUT_BASE}"
        bash "${SCRIPT_DIR}/run_physiciq_one.sh"
        "${model}" "${mode}" "${block}" "${GPU_ID}"
      )
      ;;
    physrvg)
      config_root="${OUTPUT_BASE}/PhyRVG/${tag}"
      command=(
        env INPUT_LIST="${INPUT_LIST}" OUTPUT_BASE="${OUTPUT_BASE}/PhyRVG"
        bash "${SCRIPT_DIR}/run_physrvg_physiciq_one.sh"
        "${mode}" "${block}" "${GPU_ID}"
      )
      ;;
    *)
      echo "[generation-worker] invalid model=${model}" | tee -a "${task_log}"
      continue
      ;;
  esac

  echo "[generation-worker] task=${task_id} model=${model} mode=${mode} block=${block}"
  {
    echo "task_id=${task_id}"
    echo "worker=${WORKER_NAME}"
    echo "gpu=${GPU_ID}"
    printf 'command='
    printf '%q ' "${command[@]}"
    printf '\n'
  } > "${task_log}"

  set +e
  "${command[@]}" >> "${task_log}" 2>&1
  status=$?
  if [[ "${status}" -eq 0 ]]; then
    "${PYTHON}" "${VERIFY}" \
      --config-root "${config_root}" \
      --input-list "${INPUT_LIST}" \
      --model "${model}" \
      --mode "${mode}" \
      --block "${block}" \
      --output "${validation}" >> "${task_log}" 2>&1
    status=$?
  fi
  set -e

  exec 8>"${RUN_ROOT}/generation/results.lock"
  flock 8
  if [[ "${status}" -eq 0 ]]; then
    num_done=$((num_done + 1))
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "${task_id}" "${model}" "${mode}" "${block}" "${WORKER_NAME}" \
      >> "${RUN_ROOT}/generation/completed.tsv"
  else
    num_failed=$((num_failed + 1))
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${task_id}" "${model}" "${mode}" "${block}" "${WORKER_NAME}" "${status}" \
      >> "${RUN_ROOT}/generation/failed.tsv"
  fi
  flock -u 8
  exec 8>&-
done

printf 'worker=%s\ngpu=%s\ndone=%s\nfailed=%s\nfinished_utc=%s\n' \
  "${WORKER_NAME}" "${GPU_ID}" "${num_done}" "${num_failed}" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "${RUN_ROOT}/generation/state/${WORKER_NAME}.complete"
echo "[generation-worker] finish worker=${WORKER_NAME} done=${num_done} failed=${num_failed}"
