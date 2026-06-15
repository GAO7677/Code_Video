
set -eu

PROMPT="A woven basket is hanging from a rope with a strong magnet attached to the bottom. An orange tennis ball is placed on a table beneath it. The basket is lowered and covers the ball and then the basket starts to lift again. Static shot with no camera movement."
CONTEXT_PATH="/data/gaoya/dataset/physics-iq-benchmark/mytest/0170_perspective-center_trimmed-solid-ball-peakaboo/context_video.mp4"

PYTHON_BIN=/data/gaoya/miniconda3/envs/wan/bin/python
SCRIPT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/batch_eval_lora.py
BASE_PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_train/train_0419:/home/gaoya/Code_Video/DiffSynth-Studio-main
OUTPUT_ROOT=/data/gaoya/AAA_test_video/0529/vjepa_vggt/tmp

if [ -z "$PROMPT" ] || [ -z "$CONTEXT_PATH" ]; then
  echo "PROMPT and CONTEXT_PATH must be set." >&2
  exit 1
fi

run_case() {
  LORA_PATH="$1"
  LORA_NAME=$(basename "$(dirname "$(dirname "$LORA_PATH")")")
  RUN_OUTPUT_ROOT="${OUTPUT_ROOT}/${LORA_NAME}"
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${BASE_PYTHONPATH}" \
    "${PYTHON_BIN}" "${SCRIPT}" \
      --context_path "${CONTEXT_PATH}" \
      --prompt "${PROMPT}" \
      --lora_path "${LORA_PATH}" \
      --output_root "${RUN_OUTPUT_ROOT}" \
      --overwrite
}

run_case "/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora/checkpoints/step-010000/checkpoint.safetensors"
run_case "/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors"
run_case "/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-001000/checkpoint.safetensors"
