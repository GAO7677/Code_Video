#!/usr/bin/env bash
set -euo pipefail

# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_test5_head_ablation_all_gpu0123456.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_LIST="${SOURCE_LIST:-/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt}"
OUTPUT_BASE="${OUTPUT_BASE:-/data/gaoya/agent-data/outputs/wan_dit_head_ablation/test5_first5}"
RUN_ROOT="${RUN_ROOT:-${OUTPUT_BASE}/_run}"
INPUT_LIST="${RUN_ROOT}/input_first5_unique.txt"
GPUS_TEXT="${GPUS_TEXT:-0 1 2 3 4 5 6}"
HEADS=(1 2 3 18)
MODELS=(wan_lora xssc physrvg)
WORKER="${SCRIPT_DIR}/run_test5_head_ablation_worker.sh"

read -r -a GPUS <<< "${GPUS_TEXT}"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "GPUS_TEXT selected no GPUs" >&2
  exit 2
fi
if [[ ! -s "${SOURCE_LIST}" ]]; then
  echo "missing source list: ${SOURCE_LIST}" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/validations"
awk '!seen[$0]++ {print; if (++count == 5) exit}' "${SOURCE_LIST}" > "${INPUT_LIST}"
if [[ "$(wc -l < "${INPUT_LIST}")" -ne 5 ]]; then
  echo "failed to select five unique cases" >&2
  exit 2
fi

: > "${RUN_ROOT}/queue.tsv"
task_index=0
for model in "${MODELS[@]}"; do
  for head in "${HEADS[@]}"; do
    printf 'head-%03d\t%s\t%s\n' "${task_index}" "${model}" "${head}" \
      >> "${RUN_ROOT}/queue.tsv"
    task_index=$((task_index + 1))
  done
done
if [[ "${task_index}" -ne 12 ]]; then
  echo "internal queue error: expected 12 tasks, got ${task_index}" >&2
  exit 2
fi

printf '1\n' > "${RUN_ROOT}/cursor"
: > "${RUN_ROOT}/completed.tsv"
: > "${RUN_ROOT}/failed.tsv"

pids=()
for gpu in "${GPUS[@]}"; do
  worker_name="head_g${gpu}"
  bash "${WORKER}" "${gpu}" "${worker_name}" \
    "${RUN_ROOT}" "${OUTPUT_BASE}" "${INPUT_LIST}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

completed="$(wc -l < "${RUN_ROOT}/completed.tsv")"
failed="$(wc -l < "${RUN_ROOT}/failed.tsv")"
validations="$(find "${RUN_ROOT}/validations" -maxdepth 1 -type f -name '*.json' | wc -l)"
videos="$(find "${OUTPUT_BASE}" -type f -name '*.mp4' | wc -l)"
printf 'completed=%s\nfailed=%s\nvalidations=%s\nvideos=%s\n' \
  "${completed}" "${failed}" "${validations}" "${videos}" \
  | tee "${RUN_ROOT}/summary.txt"

if [[ "${status}" -ne 0 || "${completed}" -ne 12 || "${failed}" -ne 0 \
      || "${validations}" -ne 12 || "${videos}" -ne 60 ]]; then
  echo "head ablation batch did not complete cleanly" >&2
  exit 1
fi
