#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=/data/gaoya/miniconda3/envs/wan/bin/python
SCRIPT_ROOT=/home/gaoya/Code_Video/Code_data/Code_train/train_0419
WAN_ROOT=/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B
VACE_ROOT=/data/gaoya/ckpt/Wan-AI-Wan2.1-VACE-1.3B

STAGE0_ROOT=/data/gaoya/AAA_test_video/Benchmark/stage0_V2V
OUTPUT_ROOT=${STAGE0_ROOT}/output
RESULT_ROOT=${STAGE0_ROOT}/result
TOOLS_ROOT=${STAGE0_ROOT}/tools
RUNTIME_ROOT=${TOOLS_ROOT}/runtime
LOG_ROOT=${TOOLS_ROOT}/logs
RUN_TAG=sample300_continue_missing_$(date -u +%Y%m%dT%H%M%SZ)
RUN_LOG_DIR=${LOG_ROOT}/${RUN_TAG}

SAMPLE300_LIST=${SCRIPT_ROOT}/benchmark_meta_json_paths_full_sample300.txt

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

MODEL_NAMES_ALL=${BASE_MODEL}=output/wan2_2_5B_baseline_TI2V,${STEP8_MODEL}=output/wan2.25B_lora_sample300_full49/step-008000,${STEP10_MODEL}=output/wan2.25B_lora_sample300_full49/step-010000,${PURE_MODEL}=output/Wan2_2_5B_pure_TI2V,${VACE_TI2V_MODEL}=output/VACE_1_3B_TI2V,${VACE_CTX01_MODEL}=output/VACE_1_3B_V2V/context_01f,${VACE_CTX02_MODEL}=output/VACE_1_3B_V2V/context_02f,${VACE_CTX04_MODEL}=output/VACE_1_3B_V2V/context_04f,${VACE_CTX08_MODEL}=output/VACE_1_3B_V2V/context_08f
MODEL_NAMES_WAN_FAIR=${BASE_MODEL}=output/wan2_2_5B_baseline_TI2V,${STEP8_MODEL}=output/wan2.25B_lora_sample300_full49/step-008000,${STEP10_MODEL}=output/wan2.25B_lora_sample300_full49/step-010000
MODEL_NAMES_VACE=${VACE_TI2V_MODEL}=output/VACE_1_3B_TI2V,${VACE_CTX01_MODEL}=output/VACE_1_3B_V2V/context_01f,${VACE_CTX02_MODEL}=output/VACE_1_3B_V2V/context_02f,${VACE_CTX04_MODEL}=output/VACE_1_3B_V2V/context_04f,${VACE_CTX08_MODEL}=output/VACE_1_3B_V2V/context_08f

mkdir -p \
  "${OUTPUT_ROOT}" \
  "${RESULT_ROOT}" \
  "${RUNTIME_ROOT}" \
  "${RUN_LOG_DIR}" \
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

count_outputs() {
  local output_dir="$1"
  find "${output_dir}" -maxdepth 1 -type f -name '*.mp4' | wc -l
}

log_counts() {
  cat <<EOF
[count] wan2_2_5B_baseline_TI2V $(count_outputs "${BASE_OUTPUT_DIR}") / 300
[count] step-008000 $(count_outputs "${STEP8_OUTPUT_DIR}") / 300
[count] step-010000 $(count_outputs "${STEP10_OUTPUT_DIR}") / 300
[count] Wan2_2_5B_pure_TI2V $(count_outputs "${PURE_OUTPUT_DIR}") / 300
[count] VACE_1_3B_TI2V $(count_outputs "${VACE_TI2V_OUTPUT_DIR}") / 300
[count] VACE_1_3B_V2V/context_01f $(count_outputs "${VACE_CTX01_OUTPUT_DIR}") / 300
[count] VACE_1_3B_V2V/context_02f $(count_outputs "${VACE_CTX02_OUTPUT_DIR}") / 300
[count] VACE_1_3B_V2V/context_04f $(count_outputs "${VACE_CTX04_OUTPUT_DIR}") / 300
[count] VACE_1_3B_V2V/context_08f $(count_outputs "${VACE_CTX08_OUTPUT_DIR}") / 300
EOF
}

run_bg() {
  local name="$1"
  shift
  local log_path="${RUN_LOG_DIR}/${name}.log"
  echo "[launch] ${name} -> ${log_path}" >&2
  (
    set -x
    "$@"
  ) >"${log_path}" 2>&1 &
  RUN_BG_PID=$!
}

wait_and_report() {
  local name="$1"
  local pid="$2"
  if wait "${pid}"; then
    echo "[done] ${name}"
  else
    echo "[fail] ${name} (pid=${pid})"
    return 1
  fi
}

echo "[stage0_continue] run_tag=${RUN_TAG}"
echo "[stage0_continue] starting counts:"
log_counts

declare -A PIDS

run_bg wan_pure_ti2v_5b \
  env CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${SCRIPT_ROOT}/batch_eval_lora.py" \
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
    --conditioning_mode input_image_only
PIDS[wan_pure_ti2v_5b]=${RUN_BG_PID}

run_bg vace_ti2v_firstframe \
  env CUDA_VISIBLE_DEVICES=1 "${PYTHON_BIN}" "${SCRIPT_ROOT}/batch_eval_vace.py" \
    --vace_root "${VACE_ROOT}" \
    --meta_list_path "${SAMPLE300_LIST}" \
    --output_root "${VACE_TI2V_OUTPUT_DIR}" \
    --runtime_root "${RUNTIME_ROOT}/${VACE_TI2V_MODEL}" \
    --model_name "${VACE_TI2V_MODEL}" \
    --mode ti2v_firstframe \
    --device cuda:0 \
    --height "${VACE_HEIGHT}" \
    --width "${VACE_WIDTH}" \
    --fps "${VACE_FPS}" \
    --num_frames "${NUM_FRAMES}" \
    --context_frames 1 \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --cfg_scale "${CFG_SCALE}" \
    --seed "${SEED}"
PIDS[vace_ti2v_firstframe]=${RUN_BG_PID}

run_bg vace_v2v_ctx01f \
  env CUDA_VISIBLE_DEVICES=2 "${PYTHON_BIN}" "${SCRIPT_ROOT}/batch_eval_vace.py" \
    --vace_root "${VACE_ROOT}" \
    --meta_list_path "${SAMPLE300_LIST}" \
    --output_root "${VACE_CTX01_OUTPUT_DIR}" \
    --runtime_root "${RUNTIME_ROOT}/${VACE_CTX01_MODEL}" \
    --model_name "${VACE_CTX01_MODEL}" \
    --mode v2v_clipref \
    --device cuda:0 \
    --height "${VACE_HEIGHT}" \
    --width "${VACE_WIDTH}" \
    --fps "${VACE_FPS}" \
    --num_frames "${NUM_FRAMES}" \
    --context_frames 1 \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --cfg_scale "${CFG_SCALE}" \
    --seed "${SEED}"
PIDS[vace_v2v_ctx01f]=${RUN_BG_PID}

run_bg vace_v2v_ctx02f \
  env CUDA_VISIBLE_DEVICES=3 "${PYTHON_BIN}" "${SCRIPT_ROOT}/batch_eval_vace.py" \
    --vace_root "${VACE_ROOT}" \
    --meta_list_path "${SAMPLE300_LIST}" \
    --output_root "${VACE_CTX02_OUTPUT_DIR}" \
    --runtime_root "${RUNTIME_ROOT}/${VACE_CTX02_MODEL}" \
    --model_name "${VACE_CTX02_MODEL}" \
    --mode v2v_clipref \
    --device cuda:0 \
    --height "${VACE_HEIGHT}" \
    --width "${VACE_WIDTH}" \
    --fps "${VACE_FPS}" \
    --num_frames "${NUM_FRAMES}" \
    --context_frames 2 \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --cfg_scale "${CFG_SCALE}" \
    --seed "${SEED}"
PIDS[vace_v2v_ctx02f]=${RUN_BG_PID}

run_bg vace_v2v_ctx04f \
  env CUDA_VISIBLE_DEVICES=6 "${PYTHON_BIN}" "${SCRIPT_ROOT}/batch_eval_vace.py" \
    --vace_root "${VACE_ROOT}" \
    --meta_list_path "${SAMPLE300_LIST}" \
    --output_root "${VACE_CTX04_OUTPUT_DIR}" \
    --runtime_root "${RUNTIME_ROOT}/${VACE_CTX04_MODEL}" \
    --model_name "${VACE_CTX04_MODEL}" \
    --mode v2v_clipref \
    --device cuda:0 \
    --height "${VACE_HEIGHT}" \
    --width "${VACE_WIDTH}" \
    --fps "${VACE_FPS}" \
    --num_frames "${NUM_FRAMES}" \
    --context_frames 4 \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --cfg_scale "${CFG_SCALE}" \
    --seed "${SEED}"
PIDS[vace_v2v_ctx04f]=${RUN_BG_PID}

run_bg vace_v2v_ctx08f \
  env CUDA_VISIBLE_DEVICES=5 "${PYTHON_BIN}" "${SCRIPT_ROOT}/batch_eval_vace.py" \
    --vace_root "${VACE_ROOT}" \
    --meta_list_path "${SAMPLE300_LIST}" \
    --output_root "${VACE_CTX08_OUTPUT_DIR}" \
    --runtime_root "${RUNTIME_ROOT}/${VACE_CTX08_MODEL}" \
    --model_name "${VACE_CTX08_MODEL}" \
    --mode v2v_clipref \
    --device cuda:0 \
    --height "${VACE_HEIGHT}" \
    --width "${VACE_WIDTH}" \
    --fps "${VACE_FPS}" \
    --num_frames "${NUM_FRAMES}" \
    --context_frames 8 \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --cfg_scale "${CFG_SCALE}" \
    --seed "${SEED}"
PIDS[vace_v2v_ctx08f]=${RUN_BG_PID}

FAILURES=0
for name in "${!PIDS[@]}"; do
  if ! wait_and_report "${name}" "${PIDS[${name}]}"; then
    FAILURES=$((FAILURES + 1))
  fi
done

echo "[stage0_continue] ending counts:"
log_counts

WAN_FAIR_RESULT_DIR=${RESULT_ROOT}/model_metrics_wan_v2v_fair
VACE_RESULT_DIR=${RESULT_ROOT}/model_metrics_vace_family_native

rm -rf "${WAN_FAIR_RESULT_DIR}" "${VACE_RESULT_DIR}"
mkdir -p "${WAN_FAIR_RESULT_DIR}" "${VACE_RESULT_DIR}"

"${PYTHON_BIN}" "${SCRIPT_ROOT}/compare_stage0_model_metrics.py" \
  --benchmark_root "${STAGE0_ROOT}" \
  --model_names "${MODEL_NAMES_WAN_FAIR}" \
  --reference_model "${BASE_MODEL}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --output_root "${WAN_FAIR_RESULT_DIR}"

"${PYTHON_BIN}" "${SCRIPT_ROOT}/compare_stage0_model_metrics.py" \
  --benchmark_root "${STAGE0_ROOT}" \
  --model_names "${MODEL_NAMES_VACE}" \
  --reference_model "${VACE_TI2V_MODEL}" \
  --height "${VACE_HEIGHT}" \
  --width "${VACE_WIDTH}" \
  --output_root "${VACE_RESULT_DIR}"

echo "[stage0_continue] metric outputs:"
echo "${WAN_FAIR_RESULT_DIR}/metrics_by_model.csv"
echo "${VACE_RESULT_DIR}/metrics_by_model.csv"

if [[ "${FAILURES}" -ne 0 ]]; then
  echo "[stage0_continue] completed with ${FAILURES} failed tasks"
  exit 1
fi

echo "[stage0_continue] completed successfully"
