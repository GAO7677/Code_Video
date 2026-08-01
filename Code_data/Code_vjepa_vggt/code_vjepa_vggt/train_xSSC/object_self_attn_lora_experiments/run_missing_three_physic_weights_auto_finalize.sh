#!/usr/bin/env bash
# Run missing PhysicIQ catch-up and keep metrics/dashboard in sync until all selected targets are complete.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
CONFIG="${CONFIG:-${SCRIPT_DIR}/xssc_lora_three_train_watch_config_with_t_head.json}"
GPUS="${GPUS:-5,7}"
METHODS="${METHODS:-slot_dedup_merge,t_head70}"
STEPS="${STEPS:-500,1000,1500,3000,3500}"
POLL_SECONDS="${POLL_SECONDS:-300}"
METRICS_POLL_SECONDS="${METRICS_POLL_SECONDS:-180}"
DO_CATCHUP="${DO_CATCHUP:-1}"

if [[ ! -s "${CONFIG}" ]]; then
  echo "Missing config: ${CONFIG}" >&2
  exit 2
fi

method_array=($(echo "${METHODS}" | tr ',' ' '))
step_array=($(echo "${STEPS}" | tr ',' ' '))

metric_count=$(${PYTHON} - "$CONFIG" <<'PY'
import json
import sys
cfg = json.load(open(sys.argv[1], 'r', encoding='utf-8'))
print(len(cfg['metrics']['cpu']) + len(cfg['metrics']['gpu']))
PY
)

metric_count=${metric_count}

wait_for_all_inference() {
  while true; do
    all_done=1
    for method in "${method_array[@]}"; do
      for step in "${step_array[@]}"; do
        if [[ ! -f "/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_three_run_watch/state/physiciq/inference/${method}/step-$(printf '%06d' "${step}").json" ]]; then
          all_done=0
          break 2
        fi
      done
    done

    if [[ ${all_done} -eq 1 ]]; then
      echo "[finalize] all inference manifests exist for methods=${METHODS} steps=${STEPS}"
      return 0
    fi

    echo "[finalize] waiting inference manifests ..."
    sleep "${POLL_SECONDS}"
  done
}

run_metrics_once() {
  ${PYTHON} "${SCRIPT_DIR}/xssc_lora_physiciq_parallel_metrics.py" \
    --config "${CONFIG}" \
    --kind cpu \
    --methods "${METHODS}" \
    --steps "${STEPS}" \
    --skip-locked \
    --refresh-plots

  ${PYTHON} "${SCRIPT_DIR}/xssc_lora_physiciq_parallel_metrics.py" \
    --config "${CONFIG}" \
    --kind gpu \
    --gpus "${GPUS}" \
    --methods "${METHODS}" \
    --steps "${STEPS}" \
    --workers-per-gpu 1 \
    --skip-locked \
    --refresh-plots

  ${PYTHON} "${SCRIPT_DIR}/build_xssc_lora_checkpoint_dashboard.py" \
    --config "${CONFIG}"
}

is_metrics_complete() {
  for method in "${method_array[@]}"; do
    for step in "${step_array[@]}"; do
      metric_root="/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_three_run_watch/state/physiciq/metrics/${method}/step-$(printf '%06d' "${step}")"
      if [[ ! -d "${metric_root}" ]]; then
        return 1
      fi
      existing=0
      for f in ${metric_root}/*.json; do
        [[ -f "${f}" ]] && existing=$((existing+1))
      done
      if [[ "${existing}" -lt "${metric_count}" ]]; then
        return 1
      fi
    done
  done
  return 0
}

echo "[finalize] starting inference catchup: config=${CONFIG}" 
if [[ "${DO_CATCHUP}" == "1" ]]; then
  ${PYTHON} "${SCRIPT_DIR}/xssc_lora_physiciq_sharded_catchup.py" \
    --config "${CONFIG}" \
    --gpus "${GPUS}" \
    --methods "${METHODS}" \
    --steps "${STEPS}" \
    --force-missing
else
  echo "[finalize] DO_CATCHUP=0, skip inference stage and only finalize metrics"
fi

wait_for_all_inference

echo "[finalize] start metric finalize loop"
while true; do
  run_metrics_once
  if is_metrics_complete; then
    echo "[finalize] all metrics complete"
    break
  fi
  echo "[finalize] metrics pending, wait ${METRICS_POLL_SECONDS}s and retry"
  sleep "${METRICS_POLL_SECONDS}"
 done

echo "[finalize] done"
