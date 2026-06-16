#!/bin/sh
# sh /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/infer.sh
set -eu

GPU_ID=1
CONTEXT_PATH=/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/raw_v1/industrial_s1_scale2_merged_h264_batch1500/val/F5_drop_support/sample_000335/context_video.mp4
PROMPT="A sphere rolls after landing on the platform and leaves the support surface, testing support switching." 
OUTPUT_ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/AAAshow
NUM_FRAMES=80

PYTHON_BIN=/data/gaoya/miniconda3/envs/wan/bin/python
SCRIPT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/batch_eval_lora.py
BASE_PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_train/train_0419:/home/gaoya/Code_Video/DiffSynth-Studio-main

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
