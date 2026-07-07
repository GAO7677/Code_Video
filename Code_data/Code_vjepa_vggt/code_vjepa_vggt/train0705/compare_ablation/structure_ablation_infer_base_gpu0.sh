#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
if [ "${GPU}" = "4" ]; then
  echo "ERROR: gpu4 故障, 禁止使用。" >&2
  exit 1
fi

STRUCTURE_ABLATION_TYPE="${STRUCTURE_ABLATION_TYPE:?STRUCTURE_ABLATION_TYPE is required}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
INPUT_JSON_LIST_PATH="${INPUT_JSON_LIST_PATH:-/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt}"
CASE_INDEX="${CASE_INDEX:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/agent-data/outputs/train0705_ablation_tmp/${STRUCTURE_ABLATION_TYPE}}"
SAMPLING_STEPS="${SAMPLING_STEPS:-4}"
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-896}"
NUM_FRAMES="${NUM_FRAMES:-24}"
CONTEXT_FRAMES="${CONTEXT_FRAMES:-8}"

PROJ=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
INFER_SCRIPT="${PROJ}/code_vjepa_vggt/train0705/compare_ablation/structure_ablation_infer_stage1b_context_only_no_gt_box_v_newtrain.py"
PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python

mkdir -p "${OUTPUT_ROOT}"

readarray -t CASE_INFO < <(python3 - "${INPUT_JSON_LIST_PATH}" "${CASE_INDEX}" <<'PY'
import json
import sys
from pathlib import Path

list_path = Path(sys.argv[1]).expanduser().resolve()
case_index = int(sys.argv[2])
lines = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if case_index < 1 or case_index > len(lines):
    raise RuntimeError(f"CASE_INDEX={case_index} out of range, total={len(lines)}")
json_path = Path(lines[case_index - 1]).expanduser().resolve()
payload = json.loads(json_path.read_text(encoding="utf-8"))
context_video = str(payload.get("input_video") or "").strip()
prompt = str(payload.get("input_caption") or "").strip()
if not context_video or not prompt:
    raise RuntimeError(f"missing input_video or input_caption in {json_path}")
print(str(json_path))
print(context_video)
print(prompt)
print(json_path.stem)
PY
)

JSON_PATH="${CASE_INFO[0]}"
CONTEXT_VIDEO="${CASE_INFO[1]}"
PROMPT="${CASE_INFO[2]}"
CASE_STEM="${CASE_INFO[3]}"
CASE_DIR="${OUTPUT_ROOT}/${CASE_STEM}"
mkdir -p "${CASE_DIR}"

export PYTHONPATH="${PROJ}:${DIFFSYNTH_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"

echo "[infer] ablation=${STRUCTURE_ABLATION_TYPE} case=${CASE_STEM}"
"${PYTHON_BIN}" "${INFER_SCRIPT}" \
  --structure-ablation-type "${STRUCTURE_ABLATION_TYPE}" \
  --checkpoint "${CHECKPOINT}" \
  --context-video "${CONTEXT_VIDEO}" \
  --prompt "${PROMPT}" \
  --output-dir "${CASE_DIR}" \
  --sampling-steps "${SAMPLING_STEPS}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --num-frames "${NUM_FRAMES}" \
  --context-frames "${CONTEXT_FRAMES}" \
  --seed 42 \
  --cfg-scale 5.0

echo "[infer] json=${JSON_PATH}"
echo "[infer] output_dir=${CASE_DIR}"
