# sh /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/infer.sh
set -eu
GPU_ID=0
CONTEXT_PATH="/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/raw_v1/industrial_s1_scale2_merged_h264_batch1500/val/F5_drop_support/sample_000335/context_video.mp4"
PROMPT="A sphere rolls after landing on the platform and leaves the support surface, testing support switching."

# PROMPT="A woven basket is hanging from a rope with a strong magnet attached to the bottom. An orange tennis ball is placed on a table beneath it. The basket is lowered and covers the ball and then the basket starts to lift again. Static shot with no camera movement."
# CONTEXT_PATH="/data/gaoya/dataset/physics-iq-benchmark/mytest/0170_perspective-center_trimmed-solid-ball-peakaboo/context_video.mp4"

PYTHON_BIN=/data/gaoya/miniconda3/envs/wan/bin/python
SCRIPT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/batch_eval_lora.py
BASE_PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_train/train_0419:/home/gaoya/Code_Video/DiffSynth-Studio-main
OUTPUT_ROOT=/data/gaoya/AAA_test_video/0529/vjepa_vggt/tmp/train0419_reference

if [ -z "$PROMPT" ] || [ -z "$CONTEXT_PATH" ]; then
  echo "PROMPT and CONTEXT_PATH must be set." >&2
  exit 1
fi

case_name=$(basename "$(dirname "$CONTEXT_PATH")")

run_case() {
  LORA_PATH="$1"
  method=$(basename "$(dirname "$LORA_PATH")")
  RUN_OUTPUT_DIR="${OUTPUT_ROOT}/${case_name}"
  OUTPUT_VIDEO_PATH="${RUN_OUTPUT_DIR}/${case_name}_${method}.mp4"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH="${BASE_PYTHONPATH}" \
    "${PYTHON_BIN}" "${SCRIPT}" \
      --context_path "${CONTEXT_PATH}" \
      --prompt "${PROMPT}" \
      --lora_path "${LORA_PATH}" \
      --output_video_path "${OUTPUT_VIDEO_PATH}" \
      --no_metadata \
      --overwrite
}

run_case "/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora/checkpoints/step-010000/checkpoint.safetensors"
run_case "/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors"
run_case "/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-001000/checkpoint.safetensors"
