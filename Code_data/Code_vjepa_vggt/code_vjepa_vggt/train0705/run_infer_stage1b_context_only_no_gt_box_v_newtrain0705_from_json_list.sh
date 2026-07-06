#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-7}"
if [ "${GPU}" = "4" ]; then
  echo "ERROR: gpu4 故障, 禁止使用。" >&2
  exit 1
fi

CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
INPUT_JSON_LIST_PATH="${INPUT_JSON_LIST_PATH:-/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/agent-data/outputs/tmp_stage1b_physinone_smoke_test5}"
SAMPLING_STEPS="${SAMPLING_STEPS:-12}"
PROJ=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
INFER_SCRIPT="${PROJ}/code_vjepa_vggt/train0705/infer_stage1b_context_only_no_gt_box_v_newtrain0705.py"
PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python

mkdir -p "${OUTPUT_ROOT}"
MANIFEST_JSONL="${OUTPUT_ROOT}/manifest.jsonl"
: > "${MANIFEST_JSONL}"

export PYTHONPATH="${PROJ}:${DIFFSYNTH_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"

python3 - "${INPUT_JSON_LIST_PATH}" <<'PY' | while IFS=$'\t' read -r row_id json_path context_video prompt case_stem; do
import json
import sys
from pathlib import Path

list_path = Path(sys.argv[1]).expanduser().resolve()
lines = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
for idx, line in enumerate(lines, start=1):
    json_path = Path(line).expanduser().resolve()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    context_video = str(payload.get("input_video") or "").strip()
    prompt = str(payload.get("input_caption") or "").strip()
    if not context_video or not prompt:
        raise RuntimeError(f"missing input_video or input_caption in {json_path}")
    case_stem = json_path.stem
    print(f"{idx:03d}\t{json_path}\t{context_video}\t{prompt}\t{case_stem}")
PY
  CASE_DIR="${OUTPUT_ROOT}/${row_id}_${case_stem}"
  mkdir -p "${CASE_DIR}"
  echo "[infer] case=${row_id}_${case_stem}"
  "${PYTHON_BIN}" "${INFER_SCRIPT}" \
    --checkpoint "${CHECKPOINT}" \
    --context-video "${context_video}" \
    --prompt "${prompt}" \
    --output-dir "${CASE_DIR}" \
    --sampling-steps "${SAMPLING_STEPS}" \
    --seed 42 \
    --cfg-scale 5.0
  python3 - "${MANIFEST_JSONL}" "${row_id}" "${json_path}" "${context_video}" "${prompt}" "${CASE_DIR}" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
row_id = sys.argv[2]
json_path = sys.argv[3]
context_video = sys.argv[4]
prompt = sys.argv[5]
case_dir = sys.argv[6]
record = {
    "row_id": row_id,
    "input_json": json_path,
    "context_video": context_video,
    "prompt": prompt,
    "output_dir": case_dir,
}
with manifest.open("a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
PY
done

echo "[infer] manifest=${MANIFEST_JSONL}"
