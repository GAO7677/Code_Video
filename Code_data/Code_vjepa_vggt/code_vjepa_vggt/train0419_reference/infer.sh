#!/bin/sh
# sh /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/infer.sh
set -eu

GPU_ID=1
CONTEXT_PATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/AAAshow/train0419_step010000_0002p5s3p5.mp4
PROMPT="Two pillows on a table and two grabber tools hanging above them from which a brown tennis ball and an orange block are suspended. The grabber tools let go of the ball and block. Static shot with no camera movement.,"
OUTPUT_ROOT=/data/gaoya/AAA_test_video/0529/vjepa_vggt/test/outputs/D_clean
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
