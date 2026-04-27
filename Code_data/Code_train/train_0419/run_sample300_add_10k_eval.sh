#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=/data/gaoya/miniconda3/envs/wan/bin/python
SCRIPT_ROOT=/home/gaoya/Code_Video/Code_data/Code_train/train_0419
WAN_ROOT=/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B
TEST_ROOT=/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora/test
BENCHMARK_ROOT=/data/gaoya/AAA_test_video/Benchmark/openvid_mixed_ctx24_384x672_lora_sample300_full49
SAMPLE_LIST=${SCRIPT_ROOT}/benchmark_meta_json_paths_full_sample300.txt
LORA_10K=/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora/checkpoints/step-010000/checkpoint.safetensors

GENERATION_ROOT=${BENCHMARK_ROOT}/generated_videos
RUNTIME_ROOT=${BENCHMARK_ROOT}/runtime
COMPARE_BASE_10K_ROOT=${RUNTIME_ROOT}/comparison_base_vs_step-010000
COMPARE_8K_10K_ROOT=${RUNTIME_ROOT}/comparison_step-008000_vs_step-010000
LEGACY_CURRENT_GENERATION_ROOT=${TEST_ROOT}/sample300_full49_generation
LEGACY_CURRENT_RUNTIME_ROOT=${TEST_ROOT}/_benchmark_runtime/sample300_full49_generation

BASE_MODEL_NAME=base-ti2v-5b
MODEL_8K_NAME=step-008000
MODEL_10K_NAME=step-010000

HEIGHT=384
WIDTH=672
FPS=8
NUM_FRAMES=49
CONTEXT_FRAMES=16
NUM_INFERENCE_STEPS=50
CFG_SCALE=5.0
SEED=42
EVAL_CUDA_VISIBLE_DEVICES=${EVAL_CUDA_VISIBLE_DEVICES:-1,4}
COMPARE_CUDA_VISIBLE_DEVICES=${COMPARE_CUDA_VISIBLE_DEVICES:-1}

mkdir -p \
  "${GENERATION_ROOT}" \
  "${RUNTIME_ROOT}" \
  "${COMPARE_BASE_10K_ROOT}" \
  "${COMPARE_8K_10K_ROOT}"

export TOKENIZERS_PARALLELISM=false

seed_existing_outputs() {
  cp -an "${LEGACY_CURRENT_GENERATION_ROOT}/." "${GENERATION_ROOT}/" 2>/dev/null || true
  cp -an "${LEGACY_CURRENT_RUNTIME_ROOT}/." "${RUNTIME_ROOT}/" 2>/dev/null || true
}

seed_existing_outputs

CUDA_VISIBLE_DEVICES="${EVAL_CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" "${SCRIPT_ROOT}/batch_eval_lora.py" \
  --wan_root "${WAN_ROOT}" \
  --output_root "${GENERATION_ROOT}/${MODEL_10K_NAME}" \
  --runtime_root "${RUNTIME_ROOT}/${MODEL_10K_NAME}" \
  --lora_path "${LORA_10K}" \
  --meta_list_path "${SAMPLE_LIST}" \
  --model_name "${MODEL_10K_NAME}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --fps "${FPS}" \
  --num_frames "${NUM_FRAMES}" \
  --context_frames "${CONTEXT_FRAMES}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --cfg_scale "${CFG_SCALE}" \
  --seed "${SEED}" \
  --multi_gpu

CUDA_VISIBLE_DEVICES="${COMPARE_CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" "${SCRIPT_ROOT}/run_validation_vbench.py" \
  --compare_base_name "${BASE_MODEL_NAME}" \
  --compare_base_generated_dir "${GENERATION_ROOT}/${BASE_MODEL_NAME}" \
  --compare_base_runtime_root "${RUNTIME_ROOT}/${BASE_MODEL_NAME}" \
  --compare_ft_name "${MODEL_10K_NAME}" \
  --compare_ft_generated_dir "${GENERATION_ROOT}/${MODEL_10K_NAME}" \
  --compare_ft_runtime_root "${RUNTIME_ROOT}/${MODEL_10K_NAME}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --compare_output_root "${COMPARE_BASE_10K_ROOT}"

CUDA_VISIBLE_DEVICES="${COMPARE_CUDA_VISIBLE_DEVICES}" "${PYTHON_BIN}" "${SCRIPT_ROOT}/run_validation_vbench.py" \
  --compare_base_name "${MODEL_8K_NAME}" \
  --compare_base_generated_dir "${GENERATION_ROOT}/${MODEL_8K_NAME}" \
  --compare_base_runtime_root "${RUNTIME_ROOT}/${MODEL_8K_NAME}" \
  --compare_ft_name "${MODEL_10K_NAME}" \
  --compare_ft_generated_dir "${GENERATION_ROOT}/${MODEL_10K_NAME}" \
  --compare_ft_runtime_root "${RUNTIME_ROOT}/${MODEL_10K_NAME}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --compare_output_root "${COMPARE_8K_10K_ROOT}"

"${PYTHON_BIN}" "${SCRIPT_ROOT}/build_training_eval_portal.py" \
  --benchmark_root "${BENCHMARK_ROOT}" \
  --compare_model_names "${BASE_MODEL_NAME},${MODEL_8K_NAME},${MODEL_10K_NAME}"
