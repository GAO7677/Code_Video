#!/usr/bin/env bash
set -u

ROOT="/data/gaoya/agent-data/outputs/attention_lora_object_query_group_mean_case001460"
STATUS="${ROOT}/monitor_status.log"
mkdir -p "${ROOT}"
while true; do
  {
    printf '\n=== %s ===\n' "$(date -u +%FT%TZ)"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader \
      | awk -F, '$1 ~ /^[01235]$/'
    ps -eo pid,etimes,cmd \
      | rg 'run_object_query_group_mean_seed_gpu|run_attention_lora_seed_sweep_worker' \
      | rg -v 'rg ' || true
    for seed in 90094 35075 21890 49530 47326 32466; do
      for stage in all_steps steps00_09; do
        marker="${ROOT}/seeds/seed_$(printf '%06d' "${seed}")/${stage}/complete"
        [[ -f "${marker}" ]] && state=complete || state=pending
        printf 'seed=%s stage=%s state=%s\n' "${seed}" "${stage}" "${state}"
      done
    done
  } >> "${STATUS}" 2>&1
  sleep 30
done
