#!/bin/sh
# sh /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/infer.sh
set -eu

MODE=${MODE:-context}
GPU_ID=${GPU_ID:-1}
PYTHON_BIN=${PYTHON_BIN:-/data/gaoya/miniconda3/envs/wan/bin/python}
BASE_PYTHONPATH=${BASE_PYTHONPATH:-/home/gaoya/Code_Video/Code_data/Code_train/train_0419:/home/gaoya/Code_Video/DiffSynth-Studio-main}

if [ "${MODE}" = "t2v" ]; then
  SCRIPT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/infer_t2v_lora.py
  WAN_ROOT=${WAN_ROOT:-/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B}
  LORA_PATH=${LORA_PATH:-/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors}
  PROMPT=${PROMPT:-A ball flew in from the left, knocking a wooden block that was stationary on the ground far away.}
  SEED=${SEED:-20250622}
  OUTPUT_VIDEO_PATH=${OUTPUT_VIDEO_PATH:-/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/AAAshow/t2v_step000500_same_prompt_seed20250622.mp4}
  HEIGHT=${HEIGHT:-704}
  WIDTH=${WIDTH:-1280}
  NUM_FRAMES=${NUM_FRAMES:-121}
  FPS=${FPS:-24}
  NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-50}
  CFG_SCALE=${CFG_SCALE:-5.0}

  CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH="${BASE_PYTHONPATH}" \
    "${PYTHON_BIN}" "${SCRIPT}" \
      --wan_root "${WAN_ROOT}" \
      --lora_path "${LORA_PATH}" \
      --output_video_path "${OUTPUT_VIDEO_PATH}" \
      --prompt "${PROMPT}" \
      --seed "${SEED}" \
      --height "${HEIGHT}" \
      --width "${WIDTH}" \
      --num_frames "${NUM_FRAMES}" \
      --fps "${FPS}" \
      --num_inference_steps "${NUM_INFERENCE_STEPS}" \
      --cfg_scale "${CFG_SCALE}" \
      --overwrite
  exit 0
fi

CONTEXT_PATH=/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/raw_v1/industrial_s1_scale2_merged_h264_batch1500/val/F5_drop_support/sample_000335/context_video.mp4
PROMPT="A sphere rolls after landing on the platform and leaves the support surface, testing support switching."
OUTPUT_ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/AAAshow
NUM_FRAMES=80

SCRIPT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/batch_eval_lora.py

LORAS="
/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora/checkpoints/step-010000/checkpoint.safetensors
/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors
/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-001000/checkpoint.safetensors
"

for LORA_PATH in $LORAS; do
  method_dir_name=$(basename "$(dirname "$LORA_PATH")")
  case "$LORA_PATH" in
    *openvid*)
      method_name="openvid_${method_dir_name}"
      ;;
    *)
      method_name="0613lora_${method_dir_name}"
      ;;
  esac
  context_base=$(basename "$CONTEXT_PATH")
  output_base=${context_base##*_}

  CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH="${BASE_PYTHONPATH}" \
    "${PYTHON_BIN}" "${SCRIPT}" \
      --context_path "${CONTEXT_PATH}" \
      --prompt "${PROMPT}" \
      --lora_path "${LORA_PATH}" \
      --output_video_path "${OUTPUT_ROOT}/${method_name}/${output_base}" \
      --no_metadata \
      --num_frames "${NUM_FRAMES}" \
      --overwrite
done
