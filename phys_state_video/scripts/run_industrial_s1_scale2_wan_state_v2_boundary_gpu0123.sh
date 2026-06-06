#!/usr/bin/env bash
set -euo pipefail

ENV_ROOT="/data/gaoya/home_miniconda3/envs/wan-cu128"
ENV_PY="${ENV_ROOT}/bin/python"
TORCHRUN_BIN="${ENV_ROOT}/bin/torchrun"
PROJECT_ROOT="/home/gaoya/Code_Video/phys_state_video"
EPISODE_ROOT="/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/episodes_v1/industrial_s1_scale2_256x144_s8_f16_n6"
WAN_CKPT_DIR="/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B"
BASE_RUN_ROOT="/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v2/industrial_s1_scale2_wan_state_v2_ti2vprefix_gpu0123_20260605"
RESUME_CKPT="${BASE_RUN_ROOT}/checkpoints/predictor_v2_last.joint_finetune.best.pt"
RUN_ROOT="/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v2/industrial_s1_scale2_wan_state_v2_boundary_gpu0123_20260606"
CKPT_DIR="${RUN_ROOT}/checkpoints"
LOG_DIR="${RUN_ROOT}/logs"
CFG_DIR="${RUN_ROOT}/configs"
PID_FILE="${RUN_ROOT}/train.pid"

TRAIN_DIR="${EPISODE_ROOT}/train"
VAL_DIR="${EPISODE_ROOT}/val"

OUTPUT_CKPT="${CKPT_DIR}/predictor_v2_boundary.pt"
LOG_FILE="${LOG_DIR}/predictor_train.log"
LAUNCH_LOG="${LOG_DIR}/launcher.log"

MASTER_PORT="${MASTER_PORT:-29626}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EPOCHS_FUTURE="${EPOCHS_FUTURE:-4}"
EPOCHS_JOINT="${EPOCHS_JOINT:-6}"
LR="${LR:-5e-5}"

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${CKPT_DIR}" "${LOG_DIR}" "${CFG_DIR}"

if [[ ! -f "${RESUME_CKPT}" ]]; then
  echo "resume checkpoint is missing: ${RESUME_CKPT}" >&2
  exit 1
fi

train_count=$(find "${TRAIN_DIR}" -maxdepth 1 -type f -name '*.npz' | wc -l)
val_count=$(find "${VAL_DIR}" -maxdepth 1 -type f -name '*.npz' | wc -l)

cat > "${CFG_DIR}/run_config.env" <<EOF
ENV_PY=${ENV_PY}
TORCHRUN_BIN=${TORCHRUN_BIN}
PROJECT_ROOT=${PROJECT_ROOT}
EPISODE_ROOT=${EPISODE_ROOT}
WAN_CKPT_DIR=${WAN_CKPT_DIR}
BASE_RUN_ROOT=${BASE_RUN_ROOT}
RESUME_CKPT=${RESUME_CKPT}
RUN_ROOT=${RUN_ROOT}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
MASTER_PORT=${MASTER_PORT}
NPROC_PER_NODE=${NPROC_PER_NODE}
BATCH_SIZE=${BATCH_SIZE}
NUM_WORKERS=${NUM_WORKERS}
EPOCHS_FUTURE=${EPOCHS_FUTURE}
EPOCHS_JOINT=${EPOCHS_JOINT}
LR=${LR}
EOF

echo "[stage0] counts train=${train_count} val=${val_count}" | tee "${LAUNCH_LOG}"
echo "[stage1] resume predictor from ${RESUME_CKPT}" | tee -a "${LAUNCH_LOG}"

if [[ -f "${PID_FILE}" ]]; then
  existing_pid="$(cat "${PID_FILE}" || true)"
  if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    echo "training already running with pid=${existing_pid}" | tee -a "${LAUNCH_LOG}"
    exit 0
  fi
fi

nohup "${TORCHRUN_BIN}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT}" \
  "${PROJECT_ROOT}/scripts/train_predictor_wan_state_v2.py" \
  --data "${TRAIN_DIR}" \
  --val-data "${VAL_DIR}" \
  --output "${OUTPUT_CKPT}" \
  --device cuda \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --epochs-context 0 \
  --epochs-future "${EPOCHS_FUTURE}" \
  --epochs-joint "${EPOCHS_JOINT}" \
  --wan-ckpt-dir "${WAN_CKPT_DIR}" \
  --wan-repo-root "/home/gaoya/Code_Video/Wan2.2-main" \
  --wan-task ti2v-5B \
  --resume "${RESUME_CKPT}" \
  --boundary-continuity-scale 0.5 \
  --boundary-head-scale 1.0 \
  --boundary-rollout-scale 0.5 \
  --boundary-rollout-steps 3 \
  --boundary-rollout-decay 0.5 \
  --boundary-curvature-scale 0.1 \
  --adapter-align-scale 0.0 \
  --selection-metric boundary_focus \
  > "${LOG_FILE}" 2>&1 &

echo $! > "${PID_FILE}"
echo "[launched] pid=$(cat "${PID_FILE}") log=${LOG_FILE}" | tee -a "${LAUNCH_LOG}"
