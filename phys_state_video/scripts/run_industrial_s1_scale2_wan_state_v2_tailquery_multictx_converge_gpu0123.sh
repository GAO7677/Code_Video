#!/usr/bin/env bash
set -euo pipefail

ENV_ROOT="/data/gaoya/home_miniconda3/envs/wan-cu128"
ENV_PY="${ENV_ROOT}/bin/python"
TORCHRUN_BIN="${ENV_ROOT}/bin/torchrun"
PROJECT_ROOT="/home/gaoya/Code_Video/phys_state_video"
EPISODE_ROOT="/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/episodes_v1/industrial_s1_scale2_256x144_s8_f16_n6"
WAN_CKPT_DIR="/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B"
BASE_RUN_ROOT="/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v2/industrial_s1_scale2_wan_state_v2_tailquery_multictx_gpu0123_20260606"
RESUME_CKPT="${BASE_RUN_ROOT}/checkpoints/predictor_v2_tailquery_multictx.joint_finetune.best.pt"
RUN_ROOT="/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v2/industrial_s1_scale2_wan_state_v2_tailquery_multictx_converge_gpu0123_20260606"
CKPT_DIR="${RUN_ROOT}/checkpoints"
LOG_DIR="${RUN_ROOT}/logs"
CFG_DIR="${RUN_ROOT}/configs"

TRAIN_DIR="${EPISODE_ROOT}/train"
VAL_DIR="${EPISODE_ROOT}/val"

OUTPUT_CKPT="${CKPT_DIR}/predictor_v2_tailquery_multictx_converge.pt"
LOG_FILE="${LOG_DIR}/predictor_train.log"

MASTER_PORT="${MASTER_PORT:-29658}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS_JOINT="${EPOCHS_JOINT:-30}"
LR="${LR:-1e-5}"
CONTEXT_RATIO_LIST="${CONTEXT_RATIO_LIST:-1.0,0.75,0.5,0.25}"
MIN_CONTEXT_FRAMES="${MIN_CONTEXT_FRAMES:-3}"
MAX_CONTEXT_FRAMES="${MAX_CONTEXT_FRAMES:-12}"
MIN_FUTURE_FRAMES="${MIN_FUTURE_FRAMES:-8}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-5}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0001}"
WANDB_PROJECT="${WANDB_PROJECT:-phys_state_video}"
WANDB_ENTITY="${WANDB_ENTITY:-875222004-gy}"
WANDB_GROUP="${WANDB_GROUP:-wan_state_v2_tailquery_multictx}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-tailquery_multictx_converge_gpu0123_20260606}"
WANDB_TAGS="${WANDB_TAGS:-tailquery,multictx,converge,gpu0123,ti2v-5B}"

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1

mkdir -p "${CKPT_DIR}" "${LOG_DIR}" "${CFG_DIR}" "${RUN_ROOT}/wandb"

if [[ ! -f "${RESUME_CKPT}" ]]; then
  echo "resume checkpoint is missing: ${RESUME_CKPT}" >&2
  exit 1
fi

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
EPOCHS_JOINT=${EPOCHS_JOINT}
LR=${LR}
CONTEXT_RATIO_LIST=${CONTEXT_RATIO_LIST}
MIN_CONTEXT_FRAMES=${MIN_CONTEXT_FRAMES}
MAX_CONTEXT_FRAMES=${MAX_CONTEXT_FRAMES}
MIN_FUTURE_FRAMES=${MIN_FUTURE_FRAMES}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE}
EARLY_STOP_MIN_DELTA=${EARLY_STOP_MIN_DELTA}
WANDB_PROJECT=${WANDB_PROJECT}
WANDB_ENTITY=${WANDB_ENTITY}
WANDB_GROUP=${WANDB_GROUP}
WANDB_RUN_NAME=${WANDB_RUN_NAME}
WANDB_TAGS=${WANDB_TAGS}
EOF

exec "${TORCHRUN_BIN}" \
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
  --epochs-future 0 \
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
  --min-context-frames "${MIN_CONTEXT_FRAMES}" \
  --max-context-frames "${MAX_CONTEXT_FRAMES}" \
  --min-future-frames "${MIN_FUTURE_FRAMES}" \
  --context-ratio-list "${CONTEXT_RATIO_LIST}" \
  --early-stop-patience "${EARLY_STOP_PATIENCE}" \
  --early-stop-min-delta "${EARLY_STOP_MIN_DELTA}" \
  --wandb \
  --wandb-project "${WANDB_PROJECT}" \
  --wandb-entity "${WANDB_ENTITY}" \
  --wandb-group "${WANDB_GROUP}" \
  --wandb-run-name "${WANDB_RUN_NAME}" \
  --wandb-tags "${WANDB_TAGS}" \
  --wandb-dir "${RUN_ROOT}/wandb" \
  2>&1 | tee "${LOG_FILE}"
