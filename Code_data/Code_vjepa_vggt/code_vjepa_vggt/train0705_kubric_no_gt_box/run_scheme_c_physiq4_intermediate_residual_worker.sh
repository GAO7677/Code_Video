#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 STEP SCALE GPU_ID" >&2
  exit 2
fi

STEP="$1"
SCALE="$2"
GPU_ID="$3"

BASE=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box
PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
PROJ=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
INFER_SCRIPT="${BASE}/wan_stage1b_scheme_c_entity_caption_physical_v2v.py"
CHECKPOINT_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_raw49f_scheme_c_entity_caption_physical_fresh_20260714T174707Z/checkpoints
INPUT_LIST=/data/gaoya/agent-data/cache/scheme_c_physiq4_intermediate_residual_20260715/missing_three_input_jsons.txt
OUTPUT_ROOT=/data/gaoya/agent-data/outputs/scheme_c_physiq4_step2500_3500_residual_compare_20260715/new_inference
TMP_ROOT=/data/gaoya/agent-data/cache/t/scheme_c_physiq4_intermediate_residual_20260715
EXPECTED_CASES=3

case "${STEP}" in
  step-002500|step-003500) ;;
  *) echo "Unsupported step: ${STEP}" >&2; exit 2 ;;
esac
case "${SCALE}" in
  1.2|1.3|1.4) ;;
  *) echo "Unsupported intermediate scale: ${SCALE}" >&2; exit 2 ;;
esac
if ! [[ "${GPU_ID}" =~ ^[0-6]$ ]]; then
  echo "Invalid GPU id: ${GPU_ID}" >&2
  exit 2
fi

SCALE_TAG="${SCALE/./p}"
CHECKPOINT="${CHECKPOINT_ROOT}/${STEP}"
COMBO_ROOT="${OUTPUT_ROOT}/${STEP}/object_residual_${SCALE_TAG}x"
RESULTS_ROOT="${COMBO_ROOT}/results"
RESULT_JSON="${RESULTS_ROOT}/result.json"
LOG_ROOT="${OUTPUT_ROOT}/_logs"
LOG="${LOG_ROOT}/${STEP}_object_residual_${SCALE_TAG}x_gpu${GPU_ID}.log"
TMP_DIR="${TMP_ROOT}/${STEP}_object_residual_${SCALE_TAG}x"
MODEL_NAME="scheme_c_physiq4_${STEP}_object_residual_${SCALE_TAG}x"

mkdir -p "${COMBO_ROOT}" "${LOG_ROOT}" "${TMP_DIR}"
exec > >(tee -a "${LOG}") 2>&1

if [[ ! -s "${CHECKPOINT}/checkpoint.safetensors" || ! -s "${CHECKPOINT}/training_state.pt" ]]; then
  echo "Incomplete checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ "$(sed '/^[[:space:]]*$/d' "${INPUT_LIST}" | wc -l)" -ne "${EXPECTED_CASES}" ]]; then
  echo "Expected ${EXPECTED_CASES} input JSONs in ${INPUT_LIST}" >&2
  exit 1
fi

printf '%s\n' \
  "step=${STEP}" \
  "checkpoint=${CHECKPOINT}" \
  "gpu=${GPU_ID}" \
  "object_branch_residual_scale=${SCALE}" \
  "input_list=${INPUT_LIST}" \
  "expected_cases=${EXPECTED_CASES}" \
  "seed=42" \
  "cfg_scale=5.0" \
  "num_inference_steps=40" \
  "context_frames=8" \
  "output_frames=49" \
  "resolution=896x512" \
  "fps=30" \
  "negative_prompt=null" \
  > "${COMBO_ROOT}/run_config.txt"

echo "[worker] start step=${STEP} scale=${SCALE} gpu=${GPU_ID} cases=${EXPECTED_CASES}"
env \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="${PROJ}:${DIFFSYNTH_ROOT}" \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  TMPDIR="${TMP_DIR}" TMP="${TMP_DIR}" TEMP="${TMP_DIR}" \
  "${PYTHON_BIN}" "${INFER_SCRIPT}" \
    --weights-root "${CHECKPOINT}" \
    --input-json-list-path "${INPUT_LIST}" \
    --model-name "${MODEL_NAME}" \
    --output-root "${COMBO_ROOT}" \
    --step-output-dir-name results \
    --height 512 \
    --width 896 \
    --context-frames 8 \
    --num-frames 49 \
    --num-inference-steps 40 \
    --cfg-scale 5.0 \
    --seed 42 \
    --fps 30 \
    --object-branch-residual-scale "${SCALE}" \
    --force

"${PYTHON_BIN}" - "${RESULT_JSON}" "${INPUT_LIST}" "${CHECKPOINT}" "${EXPECTED_CASES}" "${SCALE}" <<'PY'
import json
import sys
from pathlib import Path

import cv2

result_path = Path(sys.argv[1])
input_list = Path(sys.argv[2])
checkpoint_path = Path(sys.argv[3])
expected_count = int(sys.argv[4])
expected_scale = float(sys.argv[5])
expected_inputs = {
    str(Path(line.strip()).resolve())
    for line in input_list.read_text(encoding="utf-8").splitlines()
    if line.strip()
}
result = json.loads(result_path.read_text(encoding="utf-8"))
entries = result.get("entries", [])
actual_inputs = {str(Path(entry.get("input_json", "")).resolve()) for entry in entries}
if not (
    str(Path(result.get("checkpoint_dir", "")).resolve()) == str(checkpoint_path.resolve())
    and len(expected_inputs) == expected_count
    and result.get("num_total") == expected_count
    and result.get("num_success") == expected_count
    and result.get("num_failed") == 0
    and result.get("num_skipped") == 0
    and len(entries) == expected_count
    and actual_inputs == expected_inputs
):
    raise SystemExit(f"Invalid batch result: {result}")

for entry in entries:
    metadata_path = Path(entry["output_video"]).with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    video_path = Path(metadata["output_video"])
    if metadata.get("negative_prompt") is not None:
        raise SystemExit(f"Expected null negative prompt: {metadata_path}")
    capture = cv2.VideoCapture(str(video_path))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, _ = capture.read()
    capture.release()
    if not ok or frames != 49 or abs(fps - 30.0) > 0.01 or (width, height) != (896, 512):
        raise SystemExit(
            f"Invalid output {video_path}: decodes={ok} frames={frames} "
            f"fps={fps} size={width}x{height}"
        )
    print(
        f"verified {video_path.name}: configured_scale={expected_scale} "
        f"frames={frames} fps={fps} size={width}x{height}"
    )
PY

printf 'step=%s scale=%s gpu=%s cases=%s result=%s\n' \
  "${STEP}" "${SCALE}" "${GPU_ID}" "${EXPECTED_CASES}" "${RESULT_JSON}" \
  > "${COMBO_ROOT}/worker_complete.txt"
echo "[worker] success step=${STEP} scale=${SCALE} gpu=${GPU_ID} cases=${EXPECTED_CASES}"
