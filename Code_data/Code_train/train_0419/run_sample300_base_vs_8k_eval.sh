#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=/data/gaoya/miniconda3/envs/wan/bin/python
SCRIPT_ROOT=/home/gaoya/Code_Video/Code_data/Code_train/train_0419
WAN_ROOT=/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B
TEST_ROOT=/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora/test
BENCHMARK_ROOT=/data/gaoya/AAA_test_video/Benchmark/openvid_mixed_ctx24_384x672_lora_sample300_full49
SAMPLE_LIST=${SCRIPT_ROOT}/benchmark_meta_json_paths_full_sample300.txt
LORA_8K=/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora/checkpoints/step-008000/checkpoint.safetensors

GENERATION_ROOT=${BENCHMARK_ROOT}/generated_videos
RUNTIME_ROOT=${BENCHMARK_ROOT}/runtime
COMPARE_ROOT=${RUNTIME_ROOT}/comparison_base_vs_step-008000
LEGACY_STEP8K_ROOT=${TEST_ROOT}/full49_generation/step-008000
LEGACY_CURRENT_GENERATION_ROOT=${TEST_ROOT}/sample300_full49_generation
LEGACY_CURRENT_RUNTIME_ROOT=${TEST_ROOT}/_benchmark_runtime/sample300_full49_generation

BASE_MODEL_NAME=base-ti2v-5b
FT_MODEL_NAME=step-008000

HEIGHT=384
WIDTH=672
FPS=8
NUM_FRAMES=49
CONTEXT_FRAMES=16
NUM_INFERENCE_STEPS=50
CFG_SCALE=5.0
SEED=42

mkdir -p "${GENERATION_ROOT}" "${RUNTIME_ROOT}" "${COMPARE_ROOT}"

export TOKENIZERS_PARALLELISM=false

seed_existing_outputs() {
  cp -an "${LEGACY_CURRENT_GENERATION_ROOT}/." "${GENERATION_ROOT}/" 2>/dev/null || true
  cp -an "${LEGACY_CURRENT_RUNTIME_ROOT}/." "${RUNTIME_ROOT}/" 2>/dev/null || true
}

reuse_step8k_outputs() {
  "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import json
import shutil
import sys

sys.path.insert(0, "/home/gaoya/Code_Video/Code_data/Code_train/train_0419")
import batch_eval_lora as bel

sample_list = Path("/home/gaoya/Code_Video/Code_data/Code_train/train_0419/benchmark_meta_json_paths_full_sample300.txt")
legacy_root = Path("/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora/test/full49_generation/step-008000")
target_root = Path("/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora/test/sample300_full49_generation/step-008000")
if not legacy_root.is_dir():
    print("legacy_step8k_root_missing")
    raise SystemExit(0)
target_root.mkdir(parents=True, exist_ok=True)
cases = bel.collect_cases(bel.load_meta_paths(sample_list), limit=None)
copied = 0
for case in cases:
    stem = Path(case["output_name"]).stem
    for suffix in (".mp4", ".json"):
        src = legacy_root / f"{stem}{suffix}"
        dst = target_root / f"{stem}{suffix}"
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
print(json.dumps({"reused_files": copied}, ensure_ascii=False))
PY
}

seed_existing_outputs
reuse_step8k_outputs

CUDA_VISIBLE_DEVICES=5,6 "${PYTHON_BIN}" "${SCRIPT_ROOT}/batch_eval_lora.py" \
  --wan_root "${WAN_ROOT}" \
  --output_root "${GENERATION_ROOT}/${BASE_MODEL_NAME}" \
  --runtime_root "${RUNTIME_ROOT}/${BASE_MODEL_NAME}" \
  --meta_list_path "${SAMPLE_LIST}" \
  --model_name "${BASE_MODEL_NAME}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --fps "${FPS}" \
  --num_frames "${NUM_FRAMES}" \
  --context_frames "${CONTEXT_FRAMES}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --cfg_scale "${CFG_SCALE}" \
  --seed "${SEED}" \
  --multi_gpu

CUDA_VISIBLE_DEVICES=5,6 "${PYTHON_BIN}" "${SCRIPT_ROOT}/batch_eval_lora.py" \
  --wan_root "${WAN_ROOT}" \
  --output_root "${GENERATION_ROOT}/${FT_MODEL_NAME}" \
  --runtime_root "${RUNTIME_ROOT}/${FT_MODEL_NAME}" \
  --lora_path "${LORA_8K}" \
  --meta_list_path "${SAMPLE_LIST}" \
  --model_name "${FT_MODEL_NAME}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --fps "${FPS}" \
  --num_frames "${NUM_FRAMES}" \
  --context_frames "${CONTEXT_FRAMES}" \
  --num_inference_steps "${NUM_INFERENCE_STEPS}" \
  --cfg_scale "${CFG_SCALE}" \
  --seed "${SEED}" \
  --multi_gpu

CUDA_VISIBLE_DEVICES=5 "${PYTHON_BIN}" "${SCRIPT_ROOT}/run_validation_vbench.py" \
  --compare_base_name "${BASE_MODEL_NAME}" \
  --compare_base_generated_dir "${GENERATION_ROOT}/${BASE_MODEL_NAME}" \
  --compare_base_runtime_root "${RUNTIME_ROOT}/${BASE_MODEL_NAME}" \
  --compare_ft_name "${FT_MODEL_NAME}" \
  --compare_ft_generated_dir "${GENERATION_ROOT}/${FT_MODEL_NAME}" \
  --compare_ft_runtime_root "${RUNTIME_ROOT}/${FT_MODEL_NAME}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --compare_output_root "${COMPARE_ROOT}"

"${PYTHON_BIN}" "${SCRIPT_ROOT}/build_training_eval_portal.py" \
  --benchmark_root "${BENCHMARK_ROOT}" \
  --compare_model_names "${BASE_MODEL_NAME},${FT_MODEL_NAME}"
