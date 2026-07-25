#!/usr/bin/env bash
set -euo pipefail

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
QUEUE="${RUN_ROOT}/queue.tsv"
CURSOR="${RUN_ROOT}/cursor"
LOCK="${RUN_ROOT}/queue.lock"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
VERIFY="${SCRIPT_DIR}/verify_test5_ablation_outputs.py"
WORKER_LOG="${RUN_ROOT}/logs/${WORKER_NAME}.log"

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/validations"
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

echo "[head-worker] start worker=${WORKER_NAME} gpu=${GPU_ID}"
while true; do
  task="$(claim_task)"
  if [[ -z "${task}" ]]; then
    break
  fi
  IFS=$'\t' read -r task_id model head <<< "${task}"
  printf -v head_padded "%02d" "$((10#${head}))"
  tag="self_attn_head_zero_block17_head${head_padded}"
  task_log="${RUN_ROOT}/logs/${task_id}.log"
  validation="${RUN_ROOT}/validations/${task_id}.json"

  if [[ "${model}" == "physrvg" ]]; then
    config_root="${OUTPUT_BASE}/PhyRVG/${tag}"
    command=(
      env INPUT_LIST="${INPUT_LIST}" OUTPUT_BASE="${OUTPUT_BASE}/PhyRVG"
      bash "${SCRIPT_DIR}/run_physrvg_physiciq_one.sh"
      self_attn_head_zero 17 "${GPU_ID}" "${head}"
    )
  else
    config_root="${OUTPUT_BASE}/${model}/${tag}"
    command=(
      env INPUT_LIST="${INPUT_LIST}" OUTPUT_BASE="${OUTPUT_BASE}"
      bash "${SCRIPT_DIR}/run_physiciq_one.sh"
      "${model}" self_attn_head_zero 17 "${GPU_ID}" "${head}"
    )
  fi

  {
    echo "task_id=${task_id}"
    echo "model=${model}"
    echo "head=${head}"
    echo "gpu=${GPU_ID}"
    printf 'command='
    printf '%q ' "${command[@]}"
    printf '\n'
  } > "${task_log}"
  echo "[head-worker] task=${task_id} model=${model} head=${head}"

  status=0
  "${command[@]}" >> "${task_log}" 2>&1 || status=$?
  if [[ "${status}" -eq 0 ]]; then
    "${PYTHON}" "${VERIFY}" \
      --config-root "${config_root}" \
      --input-list "${INPUT_LIST}" \
      --model "${model}" \
      --mode self_attn_head_zero \
      --block 17 \
      --head "${head}" \
      --output "${validation}" >> "${task_log}" 2>&1 || status=$?
  fi

  exec 8>"${RUN_ROOT}/results.lock"
  flock 8
  if [[ "${status}" -eq 0 ]]; then
    printf '%s\t%s\t%s\t%s\n' \
      "${task_id}" "${model}" "${head}" "${WORKER_NAME}" \
      >> "${RUN_ROOT}/completed.tsv"
  else
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "${task_id}" "${model}" "${head}" "${WORKER_NAME}" "${status}" \
      >> "${RUN_ROOT}/failed.tsv"
  fi
  flock -u 8
  exec 8>&-
done
echo "[head-worker] finish worker=${WORKER_NAME}"
