#!/usr/bin/env bash
# Run: bash run_watch_full_sa_no_object_vjepa_loss.sh

set -u -o pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
CONFIG="${CONFIG:-${ROOT}/xssc_lora_three_train_watch_config_with_t_head.json}"
METHOD="${METHOD:-full_sa_no_object_vjepa_loss}"
GPUS="${GPUS:-5}"
POLL_SECONDS="${POLL_SECONDS:-60}"
GPU_WORKERS_PER_GPU="${GPU_WORKERS_PER_GPU:-2}"
METRIC_GPUS="${METRIC_GPUS:-6}"
METRIC_GPU_WORKERS_PER_GPU="${METRIC_GPU_WORKERS_PER_GPU:-1}"
METRIC_GPU_LOCK="${METRIC_GPU_LOCK:-/data/gaoya/agent-data/locks/physrvg_gpu6_metrics.lock}"
LOG_ROOT="/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_three_run_watch/logs/vjepa_loss_method_watch"

if [[ ",${GPUS}," == *,4,* ]]; then
  echo "GPU4 is prohibited by workspace rules." >&2
  exit 2
fi
if [[ ",${METRIC_GPUS}," == *,4,* ]]; then
  echo "GPU4 is prohibited by workspace rules." >&2
  exit 2
fi
mkdir -p "${LOG_ROOT}"
mkdir -p "$(dirname "${METRIC_GPU_LOCK}")"
export PYTHONNOUSERSITE=1

run_stage() {
  local name="$1"
  shift
  echo "[$(date -u +%FT%TZ)] start ${name}"
  "$@" 2>&1 | tee -a "${LOG_ROOT}/${name}.log"
  local status=${PIPESTATUS[0]}
  echo "[$(date -u +%FT%TZ)] finish ${name} status=${status}"
  return "${status}"
}

run_metric_stage() {
  local name="$1"
  shift
  echo "[$(date -u +%FT%TZ)] wait metric lock ${name} gpu=${METRIC_GPUS}"
  flock -x "${METRIC_GPU_LOCK}" bash -c '
    echo "[$(date -u +%FT%TZ)] start metric stage $1"
    shift
    "$@"
  ' _ "${name}" "$@" 2>&1 | tee -a "${LOG_ROOT}/${name}.log"
  local status=${PIPESTATUS[0]}
  echo "[$(date -u +%FT%TZ)] finish ${name} status=${status}"
  return "${status}"
}

while true; do
  run_stage generation \
    "${PYTHON}" "${ROOT}/run_missing_checkpoint_generation.py" \
      --config "${CONFIG}" --gpus "${GPUS}" --methods "${METHOD}" || true

  run_stage test5_cpu_metrics \
    "${PYTHON}" "${ROOT}/xssc_lora_checkpoint_filtered_cpu_metrics.py" \
      --config "${CONFIG}" --methods "${METHOD}" --steps "500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000,10500,11000,11500,12000,12500,13000,13500,14000,14500,15000,15500,16000,16500,17000,17500,18000,18500,19000,19500,20000" --workers 4 --refresh || true

  run_metric_stage test5_gpu_metrics \
    "${PYTHON}" "${ROOT}/xssc_lora_checkpoint_parallel_metrics.py" \
      --config "${CONFIG}" --gpus "${METRIC_GPUS}" --methods "${METHOD}" \
      --workers-per-gpu "${METRIC_GPU_WORKERS_PER_GPU}" --refresh || true

  run_stage physiciq_cpu_metrics \
    "${PYTHON}" "${ROOT}/xssc_lora_physiciq_parallel_metrics.py" \
      --config "${CONFIG}" --kind cpu --cpu-workers 4 --methods "${METHOD}" \
      --skip-locked --refresh-plots || true

  run_metric_stage physiciq_gpu_metrics \
    "${PYTHON}" "${ROOT}/xssc_lora_physiciq_parallel_metrics.py" \
      --config "${CONFIG}" --kind gpu --gpus "${METRIC_GPUS}" \
      --workers-per-gpu "${METRIC_GPU_WORKERS_PER_GPU}" --methods "${METHOD}" \
      --skip-locked --refresh-plots || true

  run_stage dashboard \
    "${PYTHON}" "${ROOT}/build_xssc_lora_checkpoint_dashboard.py" \
      --config "${CONFIG}" || true

  echo "[$(date -u +%FT%TZ)] sleep ${POLL_SECONDS}s"
  sleep "${POLL_SECONDS}"
done
