#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
DATA_DIR="${DATA_DIR:-/data/gaoya/dataset}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-/data/gaoya/agent-data/checkpoints/xssc_vjepa2_1_video_noncausal_movi_c_10f_transfer16000_clip2_steps50000/rsfq2_c-movi_c-vjepa2_1_vitl16_256-video-10f-slot512-transfer16000-clip2/42/step-025000.pth}"
LOG_DIR="${LOG_DIR:-/data/gaoya/agent-data/outputs/xssc_stage1_causal_state_from25000_gpu0/logs}"
CONFIG="${ROOT}/upstream/config-randsfq/rsfq2_c-movi_c-vjepa2_1_vitl16_256-video-24f-slot512-prefix-causal-from25000-gpu0.py"
PROBE="${ROOT}/stage1_causal_state_probe/probe_training_capacity.py"
LAUNCHER="${ROOT}/run_train_stage1_movic_24f_prefix_causal_from25000_gpu0.sh"
FREE_MEMORY_MIB="${FREE_MEMORY_MIB:-46000}"
MAX_RESERVED_GIB="${MAX_RESERVED_GIB:-42}"

mkdir -p "${LOG_DIR}"
WAIT_LOG="${LOG_DIR}/wait_gpu0.log"
PROBE_LOG="${LOG_DIR}/capacity_probe.log"
TRAIN_LOG="${LOG_DIR}/train.log"

echo "[$(date -Is)] waiting for GPU0 free_memory>=${FREE_MEMORY_MIB} MiB" | tee -a "${WAIT_LOG}"
while true; do
  free_mib="$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  if [[ "${free_mib}" =~ ^[0-9]+$ ]] && (( free_mib >= FREE_MEMORY_MIB )); then
    # Require two consecutive idle readings to avoid racing a finishing job.
    sleep 10
    free_mib2="$(nvidia-smi --id=0 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    if [[ "${free_mib2}" =~ ^[0-9]+$ ]] && (( free_mib2 >= FREE_MEMORY_MIB )); then
      break
    fi
  fi
  echo "[$(date -Is)] GPU0 free=${free_mib} MiB; still waiting" | tee -a "${WAIT_LOG}"
  sleep 30
done

echo "[$(date -Is)] GPU0 is free; probing microbatch" | tee -a "${WAIT_LOG}"
: > "${PROBE_LOG}"
chosen_batch=""
for candidate in 32 24 16 12 8 6 4 3 2 1; do
  echo "[$(date -Is)] probe batch=${candidate}" | tee -a "${PROBE_LOG}"
  set +e
  CUDA_VISIBLE_DEVICES=0 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  STAGE1_BATCH_SIZE_T="${candidate}" \
  PYTHONPATH="${ROOT}/upstream:/home/gaoya/Code_Video/vjepa2-main${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" "${PROBE}" \
      --cfg-file "${CONFIG}" \
      --checkpoint "${SOURCE_CHECKPOINT}" \
      --data-dir "${DATA_DIR}" \
      --batch-size "${candidate}" \
      --max-reserved-gib "${MAX_RESERVED_GIB}" \
      >> "${PROBE_LOG}" 2>&1
  probe_rc=$?
  set -e
  if (( probe_rc == 0 )); then
    chosen_batch="${candidate}"
    break
  fi
  echo "[$(date -Is)] batch=${candidate} rejected rc=${probe_rc}" | tee -a "${PROBE_LOG}"
done

if [[ -z "${chosen_batch}" ]]; then
  echo "[$(date -Is)] ERROR: no safe microbatch found" | tee -a "${WAIT_LOG}"
  exit 3
fi
accumulation=$((384 / chosen_batch))
selection_file="${LOG_DIR}/selected_capacity.env"
{
  echo "STAGE1_BATCH_SIZE_T=${chosen_batch}"
  echo "GRADIENT_ACCUMULATION_STEPS=${accumulation}"
  echo "EFFECTIVE_GLOBAL_BATCH_SIZE=384"
  echo "SELECTED_AT=$(date -Is)"
} > "${selection_file}"
echo "[$(date -Is)] selected batch=${chosen_batch}, accumulation=${accumulation}; starting training" | tee -a "${WAIT_LOG}"

env \
  STAGE1_BATCH_SIZE_T="${chosen_batch}" \
  SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT}" \
  DATA_DIR="${DATA_DIR}" \
  "${LAUNCHER}" 2>&1 | tee -a "${TRAIN_LOG}"
