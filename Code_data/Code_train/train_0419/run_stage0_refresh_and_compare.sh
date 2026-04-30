#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=/data/gaoya/miniconda3/envs/wan/bin/python
SCRIPT_ROOT=/home/gaoya/Code_Video/Code_data/Code_train/train_0419
WAN_ROOT=/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B
VACE_ROOT=/data/gaoya/ckpt/Wan-AI-Wan2.1-VACE-1.3B
LORA_8K=/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora/checkpoints/step-008000/checkpoint.safetensors
LORA_10K=/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora/checkpoints/step-010000/checkpoint.safetensors

STAGE0_ROOT=/data/gaoya/AAA_test_video/Benchmark/stage0_V2V
RUN_TAG=sample300_compare_all_20260430
OUTPUT_ROOT=${STAGE0_ROOT}/output
RESULT_ROOT=${STAGE0_ROOT}/result
TOOLS_ROOT=${STAGE0_ROOT}/tools
RUNTIME_ROOT=${TOOLS_ROOT}/runtime
META_ROOT=${TOOLS_ROOT}/meta
LOG_ROOT=${TOOLS_ROOT}/logs/${RUN_TAG}
PORTAL_SUBDIR=tools/visualization/benchmark_compare_portal
SUMMARY_ROOT=${RESULT_ROOT}/model_summaries

SAMPLE300_LIST=${SCRIPT_ROOT}/benchmark_meta_json_paths_full_sample300.txt
GENESIS_LIST=${META_ROOT}/benchmark_meta_json_paths_full_sample300_genesis56.txt

BASE_MODEL=base-ti2v-5b
STEP8_MODEL=step-008000
STEP10_MODEL=step-010000
PURE_MODEL=wan_pure_ti2v_5b
VACE_TI2V_MODEL=vace_ti2v_firstframe
VACE_CTX01_MODEL=vace_v2v_ctx01f
VACE_CTX02_MODEL=vace_v2v_ctx02f
VACE_CTX04_MODEL=vace_v2v_ctx04f
VACE_CTX08_MODEL=vace_v2v_ctx08f

BASE_OUTPUT_DIR=${OUTPUT_ROOT}/wan2_2_5B_baseline_TI2V
STEP8_OUTPUT_DIR=${OUTPUT_ROOT}/wan2.25B_lora_sample300_full49/step-008000
STEP10_OUTPUT_DIR=${OUTPUT_ROOT}/wan2.25B_lora_sample300_full49/step-010000
PURE_OUTPUT_DIR=${OUTPUT_ROOT}/Wan2_2_5B_pure_TI2V
VACE_TI2V_OUTPUT_DIR=${OUTPUT_ROOT}/VACE_1_3B_TI2V
VACE_CTX01_OUTPUT_DIR=${OUTPUT_ROOT}/VACE_1_3B_V2V/context_01f
VACE_CTX02_OUTPUT_DIR=${OUTPUT_ROOT}/VACE_1_3B_V2V/context_02f
VACE_CTX04_OUTPUT_DIR=${OUTPUT_ROOT}/VACE_1_3B_V2V/context_04f
VACE_CTX08_OUTPUT_DIR=${OUTPUT_ROOT}/VACE_1_3B_V2V/context_08f

MODEL_NAMES=${BASE_MODEL}=output/wan2_2_5B_baseline_TI2V,${STEP8_MODEL}=output/wan2.25B_lora_sample300_full49/step-008000,${STEP10_MODEL}=output/wan2.25B_lora_sample300_full49/step-010000,${PURE_MODEL}=output/Wan2_2_5B_pure_TI2V,${VACE_TI2V_MODEL}=output/VACE_1_3B_TI2V,${VACE_CTX01_MODEL}=output/VACE_1_3B_V2V/context_01f,${VACE_CTX02_MODEL}=output/VACE_1_3B_V2V/context_02f,${VACE_CTX04_MODEL}=output/VACE_1_3B_V2V/context_04f,${VACE_CTX08_MODEL}=output/VACE_1_3B_V2V/context_08f

HEIGHT=384
WIDTH=672
FPS=8
NUM_FRAMES=49
WAN_CONTEXT_FRAMES=16
NUM_INFERENCE_STEPS=50
CFG_SCALE=5.0
SEED=42

VACE_HEIGHT=544
VACE_WIDTH=720
VACE_FPS=16

mkdir -p \
  "${OUTPUT_ROOT}" \
  "${RESULT_ROOT}" \
  "${TOOLS_ROOT}" \
  "${RUNTIME_ROOT}" \
  "${META_ROOT}" \
  "${LOG_ROOT}" \
  "${SUMMARY_ROOT}" \
  "${BASE_OUTPUT_DIR}" \
  "${STEP8_OUTPUT_DIR}" \
  "${STEP10_OUTPUT_DIR}" \
  "${PURE_OUTPUT_DIR}" \
  "${VACE_TI2V_OUTPUT_DIR}" \
  "${VACE_CTX01_OUTPUT_DIR}" \
  "${VACE_CTX02_OUTPUT_DIR}" \
  "${VACE_CTX04_OUTPUT_DIR}" \
  "${VACE_CTX08_OUTPUT_DIR}"

export TOKENIZERS_PARALLELISM=false

generate_genesis_list() {
  "${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import json

sample_list = Path("/home/gaoya/Code_Video/Code_data/Code_train/train_0419/benchmark_meta_json_paths_full_sample300.txt")
output_path = Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V/tools/meta/benchmark_meta_json_paths_full_sample300_genesis56.txt")
paths = [Path(line.strip()) for line in sample_list.read_text(encoding="utf-8").splitlines() if line.strip()]
selected = []
for path in paths:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("dataset", "")).strip() == "GenesisRigid" or "genesis" in str(path).lower():
        selected.append(str(path))
output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text("\n".join(selected) + "\n", encoding="utf-8")
print(output_path)
print(f"genesis_count={len(selected)}")
PY
}

remove_selected_outputs() {
  local meta_list="$1"
  local output_dir="$2"
  "${PYTHON_BIN}" - <<PY
from pathlib import Path
import json

meta_list = Path(${meta_list@Q})
output_dir = Path(${output_dir@Q})
for line in meta_list.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    meta_path = Path(line)
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    dataset_name = "version_1_genesis_rigid_data_all_cases"
    sample_id = str(payload.get("sample_id") or meta_path.parent.name)
    stem = f"{dataset_name}__{sample_id}"
    for suffix in (".mp4", ".json"):
        target = output_dir / f"{stem}{suffix}"
        if target.exists():
            target.unlink()
PY
}

prepare_runtime_dir() {
  local model_name="$1"
  rm -rf "${RUNTIME_ROOT:?}/${model_name}"
  mkdir -p "${RUNTIME_ROOT}/${model_name}"
}

prepare_fresh_output_and_runtime_dirs() {
  local output_dir="$1"
  local model_name="$2"
  rm -rf "${output_dir}" "${RUNTIME_ROOT:?}/${model_name}"
  mkdir -p "${output_dir}" "${RUNTIME_ROOT}/${model_name}"
}

run_wan_genesis_refresh() {
  prepare_runtime_dir "${BASE_MODEL}"
  CUDA_VISIBLE_DEVICES=0,1 "${PYTHON_BIN}" "${SCRIPT_ROOT}/batch_eval_lora.py" \
    --wan_root "${WAN_ROOT}" \
    --output_root "${BASE_OUTPUT_DIR}" \
    --runtime_root "${RUNTIME_ROOT}/${BASE_MODEL}" \
    --meta_list_path "${GENESIS_LIST}" \
    --model_name "${BASE_MODEL}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --fps "${FPS}" \
    --num_frames "${NUM_FRAMES}" \
    --context_frames "${WAN_CONTEXT_FRAMES}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --cfg_scale "${CFG_SCALE}" \
    --seed "${SEED}" \
    --multi_gpu

  prepare_runtime_dir "${STEP8_MODEL}"
  CUDA_VISIBLE_DEVICES=0,1 "${PYTHON_BIN}" "${SCRIPT_ROOT}/batch_eval_lora.py" \
    --wan_root "${WAN_ROOT}" \
    --output_root "${STEP8_OUTPUT_DIR}" \
    --runtime_root "${RUNTIME_ROOT}/${STEP8_MODEL}" \
    --lora_path "${LORA_8K}" \
    --meta_list_path "${GENESIS_LIST}" \
    --model_name "${STEP8_MODEL}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --fps "${FPS}" \
    --num_frames "${NUM_FRAMES}" \
    --context_frames "${WAN_CONTEXT_FRAMES}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --cfg_scale "${CFG_SCALE}" \
    --seed "${SEED}" \
    --multi_gpu

  prepare_runtime_dir "${STEP10_MODEL}"
  CUDA_VISIBLE_DEVICES=0,1 "${PYTHON_BIN}" "${SCRIPT_ROOT}/batch_eval_lora.py" \
    --wan_root "${WAN_ROOT}" \
    --output_root "${STEP10_OUTPUT_DIR}" \
    --runtime_root "${RUNTIME_ROOT}/${STEP10_MODEL}" \
    --lora_path "${LORA_10K}" \
    --meta_list_path "${GENESIS_LIST}" \
    --model_name "${STEP10_MODEL}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --fps "${FPS}" \
    --num_frames "${NUM_FRAMES}" \
    --context_frames "${WAN_CONTEXT_FRAMES}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --cfg_scale "${CFG_SCALE}" \
    --seed "${SEED}" \
    --multi_gpu
}

run_pure_ti2v_300() {
  prepare_fresh_output_and_runtime_dirs "${PURE_OUTPUT_DIR}" "${PURE_MODEL}"
  CUDA_VISIBLE_DEVICES=2,3 "${PYTHON_BIN}" "${SCRIPT_ROOT}/batch_eval_lora.py" \
    --wan_root "${WAN_ROOT}" \
    --output_root "${PURE_OUTPUT_DIR}" \
    --runtime_root "${RUNTIME_ROOT}/${PURE_MODEL}" \
    --meta_list_path "${SAMPLE300_LIST}" \
    --model_name "${PURE_MODEL}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --fps "${FPS}" \
    --num_frames "${NUM_FRAMES}" \
    --context_frames 1 \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --cfg_scale "${CFG_SCALE}" \
    --seed "${SEED}" \
    --conditioning_mode input_image_only \
    --multi_gpu
}

run_vace_job() {
  local cuda_visible="$1"
  local model_name="$2"
  local output_dir="$3"
  local mode="$4"
  local context_frames="$5"
  prepare_fresh_output_and_runtime_dirs "${output_dir}" "${model_name}"
  CUDA_VISIBLE_DEVICES="${cuda_visible}" "${PYTHON_BIN}" "${SCRIPT_ROOT}/batch_eval_vace.py" \
    --vace_root "${VACE_ROOT}" \
    --meta_list_path "${SAMPLE300_LIST}" \
    --output_root "${output_dir}" \
    --runtime_root "${RUNTIME_ROOT}/${model_name}" \
    --model_name "${model_name}" \
    --mode "${mode}" \
    --device cuda:0 \
    --height "${VACE_HEIGHT}" \
    --width "${VACE_WIDTH}" \
    --fps "${VACE_FPS}" \
    --num_frames "${NUM_FRAMES}" \
    --context_frames "${context_frames}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --cfg_scale "${CFG_SCALE}" \
    --seed "${SEED}" \
    --overwrite
}

build_metrics_and_portal() {
  for model_name in "${BASE_MODEL}" "${STEP8_MODEL}" "${STEP10_MODEL}" "${PURE_MODEL}" "${VACE_TI2V_MODEL}" "${VACE_CTX01_MODEL}" "${VACE_CTX02_MODEL}" "${VACE_CTX04_MODEL}" "${VACE_CTX08_MODEL}"; do
    if [ -f "${RUNTIME_ROOT}/${model_name}/summary.json" ]; then
      mkdir -p "${SUMMARY_ROOT}/${model_name}"
      cp -f "${RUNTIME_ROOT}/${model_name}/summary.json" "${SUMMARY_ROOT}/${model_name}/summary.json"
    fi
  done

  "${PYTHON_BIN}" "${SCRIPT_ROOT}/compare_stage0_model_metrics.py" \
    --benchmark_root "${STAGE0_ROOT}" \
    --model_names "${MODEL_NAMES}" \
    --reference_model "${BASE_MODEL}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --output_root "${RESULT_ROOT}/model_metrics_all"

  "${PYTHON_BIN}" "${SCRIPT_ROOT}/build_training_eval_portal.py" \
    --benchmark_root "${STAGE0_ROOT}" \
    --compare_runtime_root "${SUMMARY_ROOT}" \
    --compare_portal_subdir "${PORTAL_SUBDIR}" \
    --compare_model_names "${MODEL_NAMES}"
}

main() {
  generate_genesis_list

  remove_selected_outputs "${GENESIS_LIST}" "${BASE_OUTPUT_DIR}"
  remove_selected_outputs "${GENESIS_LIST}" "${STEP8_OUTPUT_DIR}"
  remove_selected_outputs "${GENESIS_LIST}" "${STEP10_OUTPUT_DIR}"

  (
    run_wan_genesis_refresh
  ) > "${LOG_ROOT}/wan_genesis_refresh.log" 2>&1 &
  WAN_PID=$!

  (
    run_pure_ti2v_300
  ) > "${LOG_ROOT}/pure_ti2v_300.log" 2>&1 &
  PURE_PID=$!

  wait "${WAN_PID}"
  wait "${PURE_PID}"

  (
    run_vace_job 0 "${VACE_TI2V_MODEL}" "${VACE_TI2V_OUTPUT_DIR}" ti2v_firstframe 1
  ) > "${LOG_ROOT}/${VACE_TI2V_MODEL}.log" 2>&1 &
  PID_A=$!
  (
    run_vace_job 1 "${VACE_CTX01_MODEL}" "${VACE_CTX01_OUTPUT_DIR}" v2v_clipref 1
  ) > "${LOG_ROOT}/${VACE_CTX01_MODEL}.log" 2>&1 &
  PID_B=$!
  (
    run_vace_job 2 "${VACE_CTX02_MODEL}" "${VACE_CTX02_OUTPUT_DIR}" v2v_clipref 2
  ) > "${LOG_ROOT}/${VACE_CTX02_MODEL}.log" 2>&1 &
  PID_C=$!
  (
    run_vace_job 3 "${VACE_CTX04_MODEL}" "${VACE_CTX04_OUTPUT_DIR}" v2v_clipref 4
  ) > "${LOG_ROOT}/${VACE_CTX04_MODEL}.log" 2>&1 &
  PID_D=$!

  wait "${PID_A}"
  wait "${PID_B}"
  wait "${PID_C}"
  wait "${PID_D}"

  (
    run_vace_job 0 "${VACE_CTX08_MODEL}" "${VACE_CTX08_OUTPUT_DIR}" v2v_clipref 8
  ) > "${LOG_ROOT}/${VACE_CTX08_MODEL}.log" 2>&1
  build_metrics_and_portal
}

main "$@"
