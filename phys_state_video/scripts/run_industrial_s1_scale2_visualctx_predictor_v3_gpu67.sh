#!/usr/bin/env bash
set -euo pipefail

ENV_PY="/data/gaoya/miniconda3/envs/vjepa2/bin/python"
PROJECT_ROOT="/home/gaoya/Code_Video/phys_state_video"
EPISODE_ROOT="/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/episodes_v1/industrial_s1_scale2_256x144_s8_f16_n6"
RUN_ROOT="/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v1/industrial_s1_scale2_visualctx_predictor_v3_gpu67"
CKPT_DIR="${RUN_ROOT}/checkpoints"
LOG_DIR="${RUN_ROOT}/logs"
CFG_DIR="${RUN_ROOT}/configs"

TRAIN_DIR="${EPISODE_ROOT}/train"
VAL_DIR="${EPISODE_ROOT}/val"
TEST_DIR="${EPISODE_ROOT}/test"

PREDICTOR_LAST="${CKPT_DIR}/predictor_last.pt"
PREDICTOR_BEST="${CKPT_DIR}/predictor_best.pt"

RUN_GROUP="industrial_s1_scale2_visualctx_predictor_v3_gpu67"
PREDICTOR_RUN_NAME="industrial_s1_scale2_visualctx_predictor_v3_gpu67"

PREDICTOR_EPOCHS=40
PREDICTOR_BATCH=640
PREDICTOR_SAVE_EVERY=5

export CUDA_VISIBLE_DEVICES=6,7
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB__SERVICE_WAIT=300

mkdir -p "${CKPT_DIR}" "${LOG_DIR}" "${CFG_DIR}"

train_count=$(find "${TRAIN_DIR}" -maxdepth 1 -type f -name '*.npz' | wc -l)
val_count=$(find "${VAL_DIR}" -maxdepth 1 -type f -name '*.npz' | wc -l)
test_count=$(find "${TEST_DIR}" -maxdepth 1 -type f -name '*.npz' | wc -l)

if [[ "${train_count}" -lt 3000 || "${val_count}" -lt 400 || "${test_count}" -lt 400 ]]; then
  echo "episode counts are incomplete: train=${train_count} val=${val_count} test=${test_count}" >&2
  exit 1
fi

cat > "${CFG_DIR}/run_config.env" <<EOF
ENV_PY=${ENV_PY}
PROJECT_ROOT=${PROJECT_ROOT}
EPISODE_ROOT=${EPISODE_ROOT}
RUN_ROOT=${RUN_ROOT}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
PREDICTOR_EPOCHS=${PREDICTOR_EPOCHS}
PREDICTOR_BATCH=${PREDICTOR_BATCH}
RUN_GROUP=${RUN_GROUP}
PREDICTOR_RUN_NAME=${PREDICTOR_RUN_NAME}
EOF

echo "[stage0] counts train=${train_count} val=${val_count} test=${test_count}" | tee "${LOG_DIR}/launcher.log"
echo "[stage1] train visual-context predictor v3 on gpu67" | tee -a "${LOG_DIR}/launcher.log"

stdbuf -oL -eL "${ENV_PY}" "${PROJECT_ROOT}/scripts/train_predictor_visual_v3.py" \
  --data "${TRAIN_DIR}" \
  --val-data "${VAL_DIR}" \
  --output "${PREDICTOR_LAST}" \
  --best-output "${PREDICTOR_BEST}" \
  --epochs "${PREDICTOR_EPOCHS}" \
  --batch-size "${PREDICTOR_BATCH}" \
  --save-every "${PREDICTOR_SAVE_EVERY}" \
  --lr 1e-3 \
  --device cuda:0 \
  --gpu-ids 0,1 \
  --num-workers 16 \
  --prefetch-factor 4 \
  --wandb-project phys-state-video \
  --wandb-group "${RUN_GROUP}" \
  --wandb-run-name "${PREDICTOR_RUN_NAME}" \
  --wandb-mode online \
  2>&1 | tee "${LOG_DIR}/predictor_train.log"

if [[ ! -f "${PREDICTOR_BEST}" ]]; then
  echo "predictor best checkpoint is missing: ${PREDICTOR_BEST}" >&2
  exit 1
fi

echo "[done] visual-context predictor v3 finished" | tee -a "${LOG_DIR}/launcher.log"
