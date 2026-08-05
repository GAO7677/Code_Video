#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 GPU_ID SHARD_ID" >&2
  exit 2
fi

GPU="$1"
SHARD_ID="$2"
NUM_SHARDS=2
HERE="/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test"
RUNNER="${HERE}/run_attention_lora_seed_sweep_gpu.sh"
PREPARE_BENCH="${HERE}/prepare_attention_lora_seed_sweep_benchmark.py"
QUEUE="${HERE}/attention_lora_test5_20case_10seed_queue.tsv"
ROOT="/data/gaoya/agent-data/outputs/attention_lora_seed_sweep_test5_20case_10seed"
METRIC_ROOT="/data/gaoya/agent-data/outputs/attention_lora_seed_sweep_metrics_test5_20case_10seed"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
BENCH="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/bench.sh"
IDLE_MEMORY_MIB="${ATTENTION_TEST5_IDLE_MEMORY_MIB:-2500}"
IDLE_POLLS="${ATTENTION_TEST5_IDLE_POLLS:-4}"
IDLE_INTERVAL="${ATTENTION_TEST5_IDLE_INTERVAL:-30}"

if [[ "${GPU}" != "2" && "${GPU}" != "3" ]]; then
  echo "This experiment is restricted to GPU2/3; got GPU${GPU}" >&2
  exit 2
fi
if [[ "${SHARD_ID}" != "0" && "${SHARD_ID}" != "1" ]]; then
  echo "SHARD_ID must be 0 or 1" >&2
  exit 2
fi

mkdir -p "${ROOT}/logs" "${METRIC_ROOT}/logs"
exec > >(tee -a "${ROOT}/logs/gpu${GPU}_shard${SHARD_ID}.log") 2>&1

gpu_memory_used() {
  nvidia-smi --id="${GPU}" --query-gpu=memory.used --format=csv,noheader,nounits \
    | tr -dc '0-9'
}

wait_for_gpu() {
  local stable=0 used
  while (( stable < IDLE_POLLS )); do
    used="$(gpu_memory_used)"
    if [[ -n "${used}" ]] && (( used < IDLE_MEMORY_MIB )); then
      stable=$((stable + 1))
      echo "[$(date -Is)] GPU${GPU} idle check ${stable}/${IDLE_POLLS}: ${used} MiB"
    else
      stable=0
      echo "[$(date -Is)] GPU${GPU} busy: ${used:-unknown} MiB; waiting"
    fi
    sleep "${IDLE_INTERVAL}"
  done
  echo "[$(date -Is)] GPU${GPU} remained idle; claiming shard ${SHARD_ID}"
}

run_generation_case() {
  local case_key="$1" input_json="$2"
  local case_root="${ROOT}/cases/${case_key}"
  local marker="${case_root}/logs/generation_shard${SHARD_ID}.complete"
  [[ -f "${marker}" ]] && return
  wait_for_gpu
  while ! ATTENTION_SEED_SWEEP_ROOT="${case_root}" \
    ATTENTION_SEED_SWEEP_CASE_LIST="${case_root}/case_list.txt" \
    ATTENTION_SEED_SWEEP_CASE_KEY="${case_key}" \
    ATTENTION_SEED_SWEEP_PILOT_SEED=-1 \
    ATTENTION_SEED_SWEEP_SHARD_ID="${SHARD_ID}" \
    ATTENTION_SEED_SWEEP_COMPLETE_NAME="generation_shard${SHARD_ID}.complete" \
      bash "${RUNNER}" "${GPU}" "${NUM_SHARDS}"; do
    echo "[$(date -Is)] generation failed for ${case_key}; retry in 300s"
    sleep 300
    wait_for_gpu
  done
  echo "[$(date -Is)] generation shard complete: ${case_key}"
}

generation_all_complete() {
  local case_key input_json
  while IFS=$'\t' read -r case_key input_json; do
    [[ -f "${ROOT}/cases/${case_key}/logs/generation_shard0.complete" ]] || return 1
    [[ -f "${ROOT}/cases/${case_key}/logs/generation_shard1.complete" ]] || return 1
  done < "${QUEUE}"
  return 0
}

run_metrics_case() {
  local case_key="$1" input_json="$2"
  local source_root="${ROOT}/cases/${case_key}"
  local bench_root="${METRIC_ROOT}/cases/${case_key}"
  local marker="${bench_root}/METRICS_COMPLETE"
  [[ -f "${marker}" ]] && return
  mkdir -p "${bench_root}/logs" "${bench_root}/summaries"
  ATTENTION_SEED_SWEEP_SOURCE_ROOT="${source_root}" \
  ATTENTION_SEED_SWEEP_BENCH_ROOT="${bench_root}" \
  ATTENTION_SEED_SWEEP_INPUT_JSON="${input_json}" \
  ATTENTION_SEED_SWEEP_CASE_KEY="${case_key}" \
    "${PYTHON}" "${PREPARE_BENCH}" >> "${bench_root}/logs/prepare.log" 2>&1
  wait_for_gpu
  while ! BENCH_RUN_METRICS=1 \
    BENCH_CUDA_VISIBLE_DEVICES="${GPU}" \
    BENCH_INPUT_JSON_ALLOWLIST="${bench_root}/input_json_allowlist.txt" \
    BENCH_RESULT_DIR="${bench_root}/summaries" \
    CUDA_VISIBLE_DEVICES="${GPU}" \
      bash "${BENCH}" "${bench_root}/bench_methods.txt" \
      >> "${bench_root}/logs/metrics.log" 2>&1; do
    echo "[$(date -Is)] metrics failed for ${case_key}; retry in 300s"
    sleep 300
    wait_for_gpu
  done
  printf 'case=%s\ncompleted=%s\n' "${case_key}" "$(date -u +%FT%TZ)" > "${marker}"
  echo "[$(date -Is)] metrics complete: ${case_key}"
}

echo "[$(date -Is)] start generation queue on GPU${GPU}, shard ${SHARD_ID}/${NUM_SHARDS}"
while IFS=$'\t' read -r case_key input_json; do
  run_generation_case "${case_key}" "${input_json}"
done < "${QUEUE}"
printf 'gpu=%s\nshard=%s\ncompleted=%s\n' \
  "${GPU}" "${SHARD_ID}" "$(date -u +%FT%TZ)" \
  > "${ROOT}/GENERATION_SHARD${SHARD_ID}_COMPLETE"

echo "[$(date -Is)] waiting for both generation shards"
until generation_all_complete; do sleep 60; done
[[ -f "${ROOT}/GENERATION_COMPLETE" ]] || printf 'completed=%s\n' \
  "$(date -u +%FT%TZ)" > "${ROOT}/GENERATION_COMPLETE"

case_index=0
while IFS=$'\t' read -r case_key input_json; do
  if (( case_index % NUM_SHARDS == SHARD_ID )); then
    run_metrics_case "${case_key}" "${input_json}"
  fi
  case_index=$((case_index + 1))
done < "${QUEUE}"
printf 'gpu=%s\nshard=%s\ncompleted=%s\n' \
  "${GPU}" "${SHARD_ID}" "$(date -u +%FT%TZ)" \
  > "${METRIC_ROOT}/METRICS_SHARD${SHARD_ID}_COMPLETE"
echo "[$(date -Is)] all assigned generation and metric work complete"
