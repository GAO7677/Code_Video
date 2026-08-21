#!/usr/bin/env bash
# Watch the w0.3 Rect384x672 PhysRVG run and feed both 8844 test suites.
set -u -o pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
CONFIG="${CONFIG:-${ROOT}/xssc_lora_three_train_watch_config_with_t_head.json}"
METHOD="${METHOD:-full_sa_physrvg_vjepa_rect384x672_0717_w0p3_b4gacc1}"
GPUS="${GPUS:-0,1,2,3,5,6,7}"
GPU_WORKERS_PER_GPU="${GPU_WORKERS_PER_GPU:-2}"
POLL_SECONDS="${POLL_SECONDS:-60}"
LOG_ROOT="${LOG_ROOT:-/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_three_run_watch/logs/physrvg_vjepa_w0p3_b4gacc1_8844}"

if [[ ",${GPUS}," == *,4,* ]]; then
  echo "GPU4 is prohibited by workspace rules." >&2
  exit 2
fi
mkdir -p "${LOG_ROOT}"
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

while true; do
  # This stage discovers newly completed checkpoints and runs both test suites.
  run_stage generation \
    "${PYTHON}" "${ROOT}/run_missing_checkpoint_generation.py" \
      --config "${CONFIG}" \
      --gpus "${GPUS}" \
      --methods "${METHOD}" || true

  run_stage test5_cpu_metrics \
    "${PYTHON}" "${ROOT}/xssc_lora_checkpoint_watch.py" \
      --config "${CONFIG}" \
      --methods "${METHOD}" \
      --mode metrics \
      --kind cpu \
      --once || true

  run_stage test5_gpu_metrics \
    "${PYTHON}" "${ROOT}/xssc_lora_checkpoint_watch.py" \
      --config "${CONFIG}" \
      --methods "${METHOD}" \
      --mode metrics \
      --kind gpu \
      --gpus "${GPUS}" \
      --gpu-metric-workers-per-gpu "${GPU_WORKERS_PER_GPU}" \
      --once || true

  run_stage physiciq_cpu_metrics \
    "${PYTHON}" "${ROOT}/xssc_lora_physiciq_watch.py" \
      --config "${CONFIG}" \
      --methods "${METHOD}" \
      --mode metrics \
      --kind cpu \
      --once || true

  run_stage physiciq_gpu_metrics \
    "${PYTHON}" "${ROOT}/xssc_lora_physiciq_watch.py" \
      --config "${CONFIG}" \
      --methods "${METHOD}" \
      --mode metrics \
      --kind gpu \
      --gpus "${GPUS}" \
      --gpu-metric-workers-per-gpu "${GPU_WORKERS_PER_GPU}" \
      --once || true

  run_stage dashboard \
    "${PYTHON}" "${ROOT}/build_xssc_lora_checkpoint_dashboard.py" \
      --config "${CONFIG}" || true

  echo "[$(date -u +%FT%TZ)] sleep ${POLL_SECONDS}s"
  sleep "${POLL_SECONDS}"
done
