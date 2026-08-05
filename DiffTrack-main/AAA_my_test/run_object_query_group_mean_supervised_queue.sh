#!/usr/bin/env bash
set -u -o pipefail

GPU="${1:?usage: $0 GPU SEED...}"
shift
SEEDS=("$@")
DIFFTRACK="/home/gaoya/Code_Video/DiffTrack-main"
ROOT="/data/gaoya/agent-data/outputs/attention_lora_object_query_group_mean_case001460"
WAIT_ROOT="/data/gaoya/agent-data/outputs/attention_lora_object_query_top100_mean_case001460"
mkdir -p "${ROOT}/logs"

for seed in "${SEEDS[@]}"; do
  marker="${WAIT_ROOT}/seeds/seed_$(printf '%06d' "${seed}")/complete"
  while [[ ! -f "${marker}" ]]; do
    printf '%s gpu=%s waiting_for_top100_seed=%s\n' "$(date -u +%FT%TZ)" "${GPU}" "${seed}"
    sleep 30
  done
done

cd "${DIFFTRACK}"
for seed in "${SEEDS[@]}"; do
  for stage in all_steps steps00_09; do
    complete="${ROOT}/seeds/seed_$(printf '%06d' "${seed}")/${stage}/complete"
    [[ -f "${complete}" ]] && continue
    log="${ROOT}/logs/gpu${GPU}_seed${seed}_${stage}.log"
    success=0
    for attempt in 1 2 3; do
      printf '%s gpu=%s seed=%s stage=%s attempt=%s start\n' \
        "$(date -u +%FT%TZ)" "${GPU}" "${seed}" "${stage}" "${attempt}" | tee -a "${log}"
      if ./AAA_my_test/run_object_query_group_mean_seed_gpu.sh \
          "${GPU}" "${seed}" "${stage}" 2>&1 | tee -a "${log}"; then
        success=1
        break
      fi
      printf '%s gpu=%s seed=%s stage=%s attempt=%s failed\n' \
        "$(date -u +%FT%TZ)" "${GPU}" "${seed}" "${stage}" "${attempt}" | tee -a "${log}"
      sleep 60
    done
    if [[ "${success}" != 1 ]]; then
      printf '%s gpu=%s seed=%s stage=%s exhausted_retries\n' \
        "$(date -u +%FT%TZ)" "${GPU}" "${seed}" "${stage}" | tee -a "${log}"
      exit 1
    fi
  done
done
