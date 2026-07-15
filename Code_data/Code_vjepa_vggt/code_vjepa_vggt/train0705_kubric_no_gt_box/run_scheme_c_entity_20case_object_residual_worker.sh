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
INPUT_LIST=/data/gaoya/agent-data/cache/stage1b_scheme_c_entity_caption_physical_fresh_val/checkpoint_val_20_input_jsons.txt
STRICT_INPUT_LIST=/data/gaoya/agent-data/cache/stage1b_scheme_c_entity_caption_physical_fresh_val/physiq4_input_jsons.txt
OUTPUT_ROOT=/data/gaoya/agent-data/outputs/AAA_physv/stage1b_scheme_c_entity_caption_physical_fresh_physiq4_checkpoint_val_20260714T174707Z
TMP_ROOT=/data/gaoya/agent-data/cache/t/scheme_c_entity_20case_object_residual
EXPECTED_CASES=20

case "${STEP}" in
  step-002500|step-003500) ;;
  *) echo "Unsupported step: ${STEP}" >&2; exit 2 ;;
esac
case "${SCALE}" in
  1.0|1.5|2.0) ;;
  *) echo "Unsupported object residual scale: ${SCALE}" >&2; exit 2 ;;
esac
if ! [[ "${GPU_ID}" =~ ^[0-6]$ ]]; then
  echo "Invalid GPU id: ${GPU_ID}" >&2
  exit 2
fi

SCALE_TAG="${SCALE/./p}"
CHECKPOINT="${CHECKPOINT_ROOT}/${STEP}"
COMBO_ROOT="${OUTPUT_ROOT}/${STEP}/object_residual_${SCALE_TAG}x"
RESULT_JSON="${COMBO_ROOT}/results/result.json"
LOG_ROOT="${OUTPUT_ROOT}/_sweep_logs"
LOG="${LOG_ROOT}/${STEP}_object_residual_${SCALE_TAG}x_gpu${GPU_ID}.log"
TMP_DIR="${TMP_ROOT}/${STEP}_object_residual_${SCALE_TAG}x"
MODEL_NAME="stage1b_scheme_c_entity_caption_physical_fresh_${STEP}_object_residual_${SCALE_TAG}x"

mkdir -p "${COMBO_ROOT}" "${LOG_ROOT}" "${TMP_DIR}"
exec > >(tee -a "${LOG}") 2>&1

if [[ ! -s "${CHECKPOINT}/checkpoint.safetensors" || ! -s "${CHECKPOINT}/training_state.pt" ]]; then
  echo "[worker] incomplete checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi
if [[ "$(sed '/^[[:space:]]*$/d' "${INPUT_LIST}" | wc -l)" -ne "${EXPECTED_CASES}" ]]; then
  echo "[worker] input list does not contain ${EXPECTED_CASES} cases: ${INPUT_LIST}" >&2
  exit 1
fi

printf '%s\n' \
  "step=${STEP}" \
  "checkpoint=${CHECKPOINT}" \
  "gpu=${GPU_ID}" \
  "object_branch_residual_scale=${SCALE}" \
  "input_list=${INPUT_LIST}" \
  "strict_input_list=${STRICT_INPUT_LIST}" \
  "expected_cases=${EXPECTED_CASES}" \
  "seed=42" \
  "cfg_scale=5.0" \
  "num_inference_steps=40" \
  "context_frames=8" \
  "output_frames=49" \
  "resolution=896x512" \
  > "${COMBO_ROOT}/sweep_config.txt"

echo "[worker] start step=${STEP} scale=${SCALE} gpu=${GPU_ID} output=${COMBO_ROOT}"
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

"${PYTHON_BIN}" - "${RESULT_JSON}" "${CHECKPOINT}" "${INPUT_LIST}" "${STRICT_INPUT_LIST}" <<'PY'
import json
import sys
from pathlib import Path

import cv2

result_path = Path(sys.argv[1])
expected_checkpoint = str(Path(sys.argv[2]).resolve())
expected_inputs = [
    str(Path(line.strip()).resolve())
    for line in Path(sys.argv[3]).read_text(encoding="utf-8").splitlines()
    if line.strip()
]
strict_inputs = {
    str(Path(line.strip()).resolve())
    for line in Path(sys.argv[4]).read_text(encoding="utf-8").splitlines()
    if line.strip()
}
result = json.loads(result_path.read_text(encoding="utf-8"))
entries = result.get("entries", [])
if not (
    str(Path(result.get("checkpoint_dir", "")).resolve()) == expected_checkpoint
    and len(expected_inputs) == 20
    and len(set(expected_inputs)) == 20
    and int(result.get("num_total", -1)) == 20
    and int(result.get("num_success", -1)) == 20
    and int(result.get("num_failed", -1)) == 0
    and int(result.get("num_skipped", -1)) == 0
    and len(entries) == 20
    and {str(Path(entry.get("input_json", "")).resolve()) for entry in entries}
    == set(expected_inputs)
):
    raise SystemExit("batch-level result verification failed")

output_videos = set()
for entry in entries:
    input_path = str(Path(entry.get("input_json", "")).resolve())
    video_path = Path(entry.get("output_video", ""))
    if not video_path.is_file() or video_path.stat().st_size <= 0:
        raise SystemExit(f"missing output video: {video_path}")
    output_videos.add(str(video_path.resolve()))
    capture = cv2.VideoCapture(str(video_path))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, _ = capture.read()
    capture.release()
    if not ok or frames != 49 or (width, height) != (896, 512):
        raise SystemExit(
            f"invalid video {video_path}: decodes={ok} frames={frames} size={width}x{height}"
        )

    if input_path not in strict_inputs:
        continue
    binding = entry.get("object_debug", {}).get("entity_id_binding", {})
    matched = binding.get("matched", [])
    unmatched = binding.get("unmatched", [])
    metrics = binding.get("adapter_metrics", {})
    entity_ids = [item.get("entity_id") for item in matched]
    if not (
        binding.get("enabled")
        and matched
        and not unmatched
        and len(set(entity_ids)) == len(entity_ids)
        and float(metrics.get("train/entity_binding_active", 0.0)) == 1.0
        and int(metrics.get("train/entity_binding_valid_slot_count", -1)) == len(matched)
        and int(metrics.get("train/entity_binding_matched_slot_count", -1)) == len(matched)
        and int(metrics.get("train/entity_binding_id_collision_count", -1)) == 0
    ):
        raise SystemExit(f"strict entity-binding verification failed: {input_path}")

if len(output_videos) != 20:
    raise SystemExit(f"expected 20 unique output videos, found {len(output_videos)}")
print(f"verified: {result_path} (20/20 videos; strict entity binding passed)")
PY

printf 'step=%s scale=%s gpu=%s result=%s\n' \
  "${STEP}" "${SCALE}" "${GPU_ID}" "${RESULT_JSON}" \
  > "${COMBO_ROOT}/worker_complete.txt"
echo "[worker] success step=${STEP} scale=${SCALE} gpu=${GPU_ID} result=${RESULT_JSON}"
