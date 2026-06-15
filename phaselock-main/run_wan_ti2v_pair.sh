#!/usr/bin/env bash
set -euo pipefail
# /home/gaoya/Code_Video/phaselock-main/run_wan_ti2v_pair.sh

# Edit only these two variables for a new run.
INPUT_VIDEO="/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/raw_v1/industrial_s1_scale2_merged_h264_batch1500/val/F5_drop_support/sample_000335/context_video.mp4"
PROMPT="A sphere rolls after landing on the platform and leaves the support surface, testing support switching."

GPU_ID=7
SEED=42
SAMPLE_STEPS=20
FEW_STEPS=2
FRAME_NUM=49
SIZE="1280*704"

REPO_ROOT="/home/gaoya/Code_Video/phaselock-main"
WAN_ENV_PYTHON="/data/gaoya/miniconda3/envs/wan/bin/python"
WAN_CKPT_DIR="/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B"
PHASELOCK_SCRIPT="${REPO_ROOT}/code/scripts/wan_ti2v_phaselock.py"
BASELINE_SCRIPT="${REPO_ROOT}/Wan2.2-main/generate.py"
OUTPUT_DIR="/data/gaoya/AAA_test_video/0529/vjepa_vggt/tmp/AAAphaselock"

mkdir -p "${OUTPUT_DIR}"

if [ ! -f "${INPUT_VIDEO}" ]; then
  echo "Error: input video not found: ${INPUT_VIDEO}" >&2
  exit 1
fi

if [ $(((FRAME_NUM - 1) % 4)) -ne 0 ]; then
  echo "Error: FRAME_NUM must satisfy 4n+1 for Wan TI2V, got ${FRAME_NUM}." >&2
  echo "Examples of valid values: 49, 81, 121." >&2
  exit 1
fi

BASENAME="$(basename "${INPUT_VIDEO}")"
BASENAME="${BASENAME%.*}"
FIRST_FRAME_PATH="${OUTPUT_DIR}/${BASENAME}_first_frame.png"
PHASELOCK_OUTPUT="${OUTPUT_DIR}/${BASENAME}_phaselock.mp4"
BASELINE_OUTPUT="${OUTPUT_DIR}/${BASENAME}_baseline.mp4"

echo "Extracting first frame from: ${INPUT_VIDEO}"
"${WAN_ENV_PYTHON}" - <<'PY' "${INPUT_VIDEO}" "${FIRST_FRAME_PATH}"
import sys
import imageio.v2 as imageio

video_path = sys.argv[1]
frame_path = sys.argv[2]
reader = imageio.get_reader(video_path)
frame = reader.get_data(0)
reader.close()
imageio.imwrite(frame_path, frame)
print(frame_path)
PY

echo "Running PhaseLock TI2V..."
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${WAN_ENV_PYTHON}" "${PHASELOCK_SCRIPT}" \
  --ckpt_dir "${WAN_CKPT_DIR}" \
  --size "${SIZE}" \
  --image "${FIRST_FRAME_PATH}" \
  --prompt "${PROMPT}" \
  --output "${PHASELOCK_OUTPUT}" \
  --few_steps "${FEW_STEPS}" \
  --full_steps "${SAMPLE_STEPS}" \
  --frame_num "${FRAME_NUM}" \
  --seed "${SEED}" \
  --offload_model \
  --t5_cpu \
  --convert_model_dtype \
  --device_id 0

echo "Running baseline TI2V..."
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${WAN_ENV_PYTHON}" "${BASELINE_SCRIPT}" \
  --task ti2v-5B \
  --size "${SIZE}" \
  --ckpt_dir "${WAN_CKPT_DIR}" \
  --offload_model True \
  --convert_model_dtype \
  --t5_cpu \
  --image "${FIRST_FRAME_PATH}" \
  --prompt "${PROMPT}" \
  --sample_steps "${SAMPLE_STEPS}" \
  --frame_num "${FRAME_NUM}" \
  --base_seed "${SEED}" \
  --save_file "${BASELINE_OUTPUT}"

echo "First frame: ${FIRST_FRAME_PATH}"
echo "PhaseLock output: ${PHASELOCK_OUTPUT}"
echo "Baseline output: ${BASELINE_OUTPUT}"
