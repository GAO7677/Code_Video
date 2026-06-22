#!/usr/bin/env bash
set -euo pipefail

PROMPT="${PROMPT:-A ball flew in from the left, knocking a wooden block that was stationary on the ground far away.}"
BASE_SEED="${BASE_SEED:-20250622}"
STEPS_LIST="${STEPS_LIST:-5 15 25}"

ROOT_DIR="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa2step"
OUT_DIR="${ROOT_DIR}/tmp/wan_runs_step_sweep"
mkdir -p "${OUT_DIR}"

WAN21_REPO="/home/gaoya/Code_Video/WAN_2p2/Wan2.1-main"
WAN22_REPO="/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main"

WAN22_CKPT="/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B"
WAN21_T2V_CKPT="/data/gaoya/ckpt/Wan-AI-Wan2.1-T2V-1.3B"
WAN21_VACE_CKPT="/data/gaoya/ckpt/Wan-AI-Wan2.1-VACE-1.3B"

source /home/gaoya/miniconda3/etc/profile.d/conda.sh
conda activate wan

echo "[info] prompt: ${PROMPT}"
echo "[info] seed: ${BASE_SEED}"
echo "[info] steps: ${STEPS_LIST}"
echo "[info] outputs: ${OUT_DIR}"

for STEPS in ${STEPS_LIST}; do
  echo "[info] running ti2v-5B with ${STEPS} steps"
  pushd "${WAN22_REPO}" >/dev/null
  python generate.py \
    --task ti2v-5B \
    --size 1280*704 \
    --ckpt_dir "${WAN22_CKPT}" \
    --offload_model True \
    --convert_model_dtype \
    --t5_cpu \
    --sample_steps "${STEPS}" \
    --base_seed "${BASE_SEED}" \
    --prompt "${PROMPT}" \
    --save_file "${OUT_DIR}/wan22_ti2v_5b_steps${STEPS}_seed${BASE_SEED}.mp4" \
    2>&1 | tee "${OUT_DIR}/wan22_ti2v_5b_steps${STEPS}_seed${BASE_SEED}.log"
  popd >/dev/null

  echo "[info] running t2v-1.3B with ${STEPS} steps"
  pushd "${WAN21_REPO}" >/dev/null
  python generate.py \
    --task t2v-1.3B \
    --size 832*480 \
    --ckpt_dir "${WAN21_T2V_CKPT}" \
    --offload_model True \
    --t5_cpu \
    --sample_steps "${STEPS}" \
    --sample_shift 8 \
    --sample_guide_scale 6 \
    --base_seed "${BASE_SEED}" \
    --prompt "${PROMPT}" \
    --save_file "${OUT_DIR}/wan21_t2v_1p3b_steps${STEPS}_seed${BASE_SEED}.mp4" \
    2>&1 | tee "${OUT_DIR}/wan21_t2v_1p3b_steps${STEPS}_seed${BASE_SEED}.log"
  popd >/dev/null

  echo "[info] running vace-1.3B with ${STEPS} steps"
  pushd "${WAN21_REPO}" >/dev/null
  python generate.py \
    --task vace-1.3B \
    --size 832*480 \
    --ckpt_dir "${WAN21_VACE_CKPT}" \
    --offload_model True \
    --t5_cpu \
    --sample_steps "${STEPS}" \
    --sample_shift 16 \
    --sample_guide_scale 5 \
    --base_seed "${BASE_SEED}" \
    --prompt "${PROMPT}" \
    --save_file "${OUT_DIR}/wan21_vace_1p3b_steps${STEPS}_seed${BASE_SEED}.mp4" \
    2>&1 | tee "${OUT_DIR}/wan21_vace_1p3b_steps${STEPS}_seed${BASE_SEED}.log"
  popd >/dev/null
done

echo "[done] completed all step sweep runs"
