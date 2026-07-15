#!/usr/bin/env bash
set -euo pipefail

BASE=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box
PROJ=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
PYTHON_BIN="${PYTHON_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
INFER_SCRIPT="${BASE}/wan_stage1b_scheme_c_entity_caption_physical_v2v.py"

GPU_ID="${GPU_ID:-6}"
STEP="${STEP:-step-003500}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_raw49f_scheme_c_entity_caption_physical_fresh_20260714T174707Z/checkpoints}"
CHECKPOINT="${CHECKPOINT:-${CHECKPOINT_ROOT}/${STEP}}"
INPUT_JSON="${INPUT_JSON:-/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/agent-data/outputs/scheme_c_entity_single_case}"
TMP_ROOT="${TMP_ROOT:-/data/gaoya/agent-data/cache/t/scheme_c_entity_single_case}"

HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-896}"
CONTEXT_FRAMES="${CONTEXT_FRAMES:-8}"
OUTPUT_FRAMES="${OUTPUT_FRAMES:-49}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
CFG_SCALE="${CFG_SCALE:-5.0}"
SEED="${SEED:-42}"
FPS="${FPS:-30}"
OBJECT_BRANCH_RESIDUAL_SCALE="${OBJECT_BRANCH_RESIDUAL_SCALE:-1.5}"
FORCE="${FORCE:-1}"

[[ -s "${INPUT_JSON}" ]] || { echo "Missing input JSON: ${INPUT_JSON}" >&2; exit 1; }
[[ -s "${CHECKPOINT}/checkpoint.safetensors" ]] || { echo "Missing checkpoint.safetensors: ${CHECKPOINT}" >&2; exit 1; }
[[ -s "${CHECKPOINT}/training_state.pt" ]] || { echo "Missing training_state.pt: ${CHECKPOINT}" >&2; exit 1; }
[[ "${GPU_ID}" =~ ^[0-9]+$ ]] || { echo "GPU_ID must be one integer: ${GPU_ID}" >&2; exit 2; }
[[ "${FORCE}" == 0 || "${FORCE}" == 1 ]] || { echo "FORCE must be 0 or 1" >&2; exit 2; }

CASE_STEM="$(basename "${INPUT_JSON}" .json)"
SCALE_TAG="${OBJECT_BRANCH_RESIDUAL_SCALE/./p}"
RUN_OUTPUT_ROOT="${OUTPUT_ROOT}/${STEP}/object_residual_${SCALE_TAG}x/${CASE_STEM}"
TMP_DIR="${TMP_ROOT}/${STEP}_object_residual_${SCALE_TAG}x_${CASE_STEM}"
SINGLE_CASE_LIST="${TMP_DIR}/input_json.txt"
MODEL_NAME="scheme_c_entity_${STEP}_object_residual_${SCALE_TAG}x"
RESULT_JSON="${RUN_OUTPUT_ROOT}/results/result.json"
LOG="${RUN_OUTPUT_ROOT}/inference.log"

mkdir -p "${RUN_OUTPUT_ROOT}" "${TMP_DIR}"
printf '%s\n' "${INPUT_JSON}" > "${SINGLE_CASE_LIST}"
printf '%s\n' \
  "input_json=${INPUT_JSON}" \
  "checkpoint=${CHECKPOINT}" \
  "inference_script=${INFER_SCRIPT}" \
  "output_root=${RUN_OUTPUT_ROOT}" \
  "gpu_id=${GPU_ID}" \
  "resolution=${WIDTH}x${HEIGHT}" \
  "resize_mode=cover_crop" \
  "context_frames=${CONTEXT_FRAMES}" \
  "context_sampling=prefix" \
  "output_frames=${OUTPUT_FRAMES}" \
  "num_inference_steps=${NUM_INFERENCE_STEPS}" \
  "cfg_scale=${CFG_SCALE}" \
  "seed=${SEED}" \
  "fps=${FPS}" \
  "negative_prompt=null" \
  "object_branch=enabled" \
  "object_branch_residual_scale=${OBJECT_BRANCH_RESIDUAL_SCALE}" \
  "grounding_caption_prompt_mode=physical_noun_phrases" \
  "grounding_caption_max_phrases=4" \
  "grounding_caption_min_score=4.0" \
  "compact_object_context_slots=1" \
  "object_adapter_mlp_residual_max_ratio=3.0" \
  "object_branch_ratio_guard_max_ratio=0.30" \
  "object_branch_ratio_guard_max_block_id=-1" \
  > "${RUN_OUTPUT_ROOT}/run_config.txt"

ARGS=(
  --weights-root "${CHECKPOINT}"
  --input-json-list-path "${SINGLE_CASE_LIST}"
  --model-name "${MODEL_NAME}"
  --output-root "${RUN_OUTPUT_ROOT}"
  --step-output-dir-name results
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --context-frames "${CONTEXT_FRAMES}"
  --num-frames "${OUTPUT_FRAMES}"
  --num-inference-steps "${NUM_INFERENCE_STEPS}"
  --cfg-scale "${CFG_SCALE}"
  --seed "${SEED}"
  --fps "${FPS}"
  --object-branch-residual-scale "${OBJECT_BRANCH_RESIDUAL_SCALE}"
)
[[ "${FORCE}" == 1 ]] && ARGS+=(--force)

echo "[scheme-c single] input=${INPUT_JSON}"
echo "[scheme-c single] checkpoint=${CHECKPOINT} gpu=${GPU_ID}"
echo "[scheme-c single] output=${RUN_OUTPUT_ROOT}"
env \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="${PROJ}:${DIFFSYNTH_ROOT}" \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  TMPDIR="${TMP_DIR}" TMP="${TMP_DIR}" TEMP="${TMP_DIR}" \
  "${PYTHON_BIN}" "${INFER_SCRIPT}" "${ARGS[@]}" 2>&1 | tee "${LOG}"

"${PYTHON_BIN}" - "${RESULT_JSON}" "${INPUT_JSON}" "${CHECKPOINT}" "${OUTPUT_FRAMES}" "${WIDTH}" "${HEIGHT}" <<'PY'
import json
import sys
from pathlib import Path

import cv2

result_path, input_json, checkpoint = map(Path, sys.argv[1:4])
expected_frames, expected_width, expected_height = map(int, sys.argv[4:7])
if not result_path.is_file():
    raise SystemExit(f"Missing result JSON: {result_path}")
result = json.loads(result_path.read_text(encoding="utf-8"))
entries = result.get("entries", [])
if not (
    int(result.get("num_total", -1)) == 1
    and int(result.get("num_success", -1)) == 1
    and int(result.get("num_failed", -1)) == 0
    and len(entries) == 1
    and Path(entries[0].get("input_json", "")).resolve() == input_json.resolve()
    and Path(result.get("checkpoint_dir", "")).resolve() == checkpoint.resolve()
):
    raise SystemExit("Single-case batch verification failed")

entry = entries[0]
video = Path(entry.get("output_video", ""))
capture = cv2.VideoCapture(str(video))
frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
ok, _ = capture.read()
capture.release()
if not video.is_file() or not ok or (frames, width, height) != (
    expected_frames, expected_width, expected_height
):
    raise SystemExit(f"Invalid output video: {video}, frames={frames}, size={width}x{height}")

binding = entry.get("object_debug", {}).get("entity_id_binding", {})
matched = binding.get("matched", [])
entity_ids = [item.get("entity_id") for item in matched]
metrics = binding.get("adapter_metrics", {})
if not (
    binding.get("enabled")
    and matched
    and not binding.get("unmatched", [])
    and len(entity_ids) == len(set(entity_ids))
    and int(metrics.get("train/entity_binding_id_collision_count", -1)) == 0
):
    raise SystemExit("Scheme-C entity-binding verification failed")
print(f"verified output: {video} ({frames} frames, {width}x{height}, {len(matched)} slots)")
PY

echo "[scheme-c single] success: ${RESULT_JSON}"
