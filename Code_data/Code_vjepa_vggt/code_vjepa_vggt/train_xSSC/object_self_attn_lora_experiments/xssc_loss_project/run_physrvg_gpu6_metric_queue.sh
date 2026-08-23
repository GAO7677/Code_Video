#!/usr/bin/env bash
# Serialize the active PhysRVG/8844 GPU metric watchers on GPU6.
# Generation and CPU-only metric workers are deliberately outside this queue.
set -u -o pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
CONFIG="${CONFIG:-${ROOT}/xssc_lora_three_train_watch_config_with_t_head.json}"
GPU="6"
GPU_READY_MAX_USED_MIB="${GPU_READY_MAX_USED_MIB:-8000}"
GPU_WORKERS_PER_GPU="${GPU_WORKERS_PER_GPU:-1}"
LOCK="${METRIC_GPU_LOCK:-/data/gaoya/agent-data/locks/physrvg_gpu6_metrics.lock}"
LOG_ROOT="${LOG_ROOT:-/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_three_run_watch/logs/physrvg_gpu6_metric_queue}"
POLL_SECONDS="${POLL_SECONDS:-60}"

CHECKPOINT_METHODS="${CHECKPOINT_METHODS:-full_sa_physrvg_vjepa_rect384x672_0717_w0p3_b4gacc1,full_sa_physrvg_phyco_kubric_0717_b4gacc1,full_sa_physrvg_vjepa_utonia_scene_hardmask_v1_b2gacc2}"
PHYSICIQ_METHODS="${PHYSICIQ_METHODS:-full_sa_physrvg_no_vjepa_0717_b2g2,full_sa_physrvg_vjepa_rect384x672_0717_b2g2,full_sa_physrvg_vjepa_rect384x672_0717_w0p3_b4gacc1,full_sa_physrvg_phyco_kubric_0717_b4gacc1,full_sa_physrvg_vjepa_utonia_scene_hardmask_v1_b2gacc2,t_head100_lora_pck32_no_object,full_sa_object_xssc_loss,full_sa_physrvg_vjepa_loss_0613_b2g2}"
PARALLEL_METHODS="${PARALLEL_METHODS:-full_sa_no_object_vjepa_loss}"

mkdir -p "${LOG_ROOT}" "$(dirname "${LOCK}")"
export PYTHONNOUSERSITE=1

run_locked_once() {
  local name="$1"
  shift
  local log="${LOG_ROOT}/${name}.log"
  {
    echo "[$(date -u +%FT%TZ)] waiting for GPU6 metric lock"
    flock -x "${LOCK}" bash -c '
      stage="$1"
      echo "[$(date -u +%FT%TZ)] start ${stage}"
      shift
      "$@"
      status=$?
      echo "[$(date -u +%FT%TZ)] finish ${stage} status=${status}"
      exit "${status}"
    ' _ "${name}" "$@"
  } >>"${log}" 2>&1
  return $?
}

while true; do
  run_locked_once checkpoint_gpu \
    "${PYTHON}" "${ROOT}/xssc_lora_checkpoint_watch.py" \
      --config "${CONFIG}" --methods "${CHECKPOINT_METHODS}" \
      --mode metrics --kind gpu --gpus "${GPU}" \
      --gpu-ready-max-used-mib "${GPU_READY_MAX_USED_MIB}" \
      --gpu-metric-workers-per-gpu "${GPU_WORKERS_PER_GPU}" --once || true

  run_locked_once physiciq_gpu \
    "${PYTHON}" "${ROOT}/xssc_lora_physiciq_watch.py" \
      --config "${CONFIG}" --methods "${PHYSICIQ_METHODS}" \
      --mode metrics --kind gpu --gpus "${GPU}" \
      --gpu-ready-max-used-mib "${GPU_READY_MAX_USED_MIB}" \
      --gpu-metric-workers-per-gpu "${GPU_WORKERS_PER_GPU}" --once || true

  run_locked_once parallel_gpu \
    "${PYTHON}" "${ROOT}/xssc_lora_checkpoint_parallel_metrics.py" \
      --config "${CONFIG}" --gpus "${GPU}" --methods "${PARALLEL_METHODS}" \
      --workers-per-gpu "${GPU_WORKERS_PER_GPU}" --refresh || true

  sleep "${POLL_SECONDS}"
done
