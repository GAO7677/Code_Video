#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 SCALE GPU_ID" >&2
  exit 2
fi

SCALE="$1"
GPU_ID="$2"
SCRIPT_DIR=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box
INFER_SH="${SCRIPT_DIR}/run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh"
CHECKPOINT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_raw49f_kubric_openvid_replay_sourceaware_fp32gate_fixedctx8_init3500_save500_keepall_20260713T090024Z/checkpoints/step-003500
INPUT_LIST=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt
OUTPUT_BASE=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/physicIQ/train_stage1b_mixdataset/step-003500
LOG_ROOT=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/physicIQ/train_stage1b_mixdataset/_sweep_logs
EXPECTED_CASES=67
SCALE_TAG="${SCALE/./p}"
OUTPUT_ROOT="${OUTPUT_BASE}/object_residual_${SCALE_TAG}x"
METHOD="mixdataset_step-003500_object_residual_${SCALE_TAG}x"
LOG="${LOG_ROOT}/step-003500_object_residual_${SCALE_TAG}x.log"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
rm -f "${OUTPUT_ROOT}/worker_complete.txt"
printf '%s\n' \
  "step=step-003500" \
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
  > "${OUTPUT_ROOT}/sweep_config.txt"

echo "[worker] start scale=${SCALE} gpu=${GPU_ID} output=${OUTPUT_ROOT}"
GPU_PAIR="${GPU_ID},${GPU_ID}" \
TEST_JSON_TXT="${INPUT_LIST}" \
WEIGHTS_ROOT="${CHECKPOINT}" \
METHOD_NAME="${METHOD}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
OUTPUT_FRAMES=49 \
CTX=8 \
NUM_INFERENCE_STEPS=40 \
CFG_SCALE=5.0 \
SEED=42 \
COMPACT_OBJECT_CONTEXT_SLOTS=1 \
OBJECT_ADAPTER_MLP_RESIDUAL_MAX_RATIO=3.0 \
OBJECT_BRANCH_RESIDUAL_SCALE="${SCALE}" \
OBJECT_BRANCH_RATIO_GUARD_MAX_RATIO=0.20 \
OBJECT_BRANCH_RATIO_GUARD_MAX_BLOCK_ID=-1 \
bash "${INFER_SH}" 2>&1 | tee -a "${LOG}"

RESULT_JSON="$(find "${OUTPUT_ROOT}" -mindepth 2 -maxdepth 2 -type f -name result.json | sort | head -n 1)"
if [[ -z "${RESULT_JSON}" ]]; then
  echo "[worker] result.json not found: ${OUTPUT_ROOT}" >&2
  exit 1
fi

/home/gaoya/miniconda3/envs/wan-cu128/bin/python - "${RESULT_JSON}" "${EXPECTED_CASES}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
expected = int(sys.argv[2])
if not (
    int(payload.get("num_total", -1)) == expected
    and int(payload.get("num_success", -1)) == expected
    and int(payload.get("num_failed", -1)) == 0
):
    raise SystemExit(
        f"incomplete result: total={payload.get('num_total')} "
        f"success={payload.get('num_success')} failed={payload.get('num_failed')}"
    )
PY

printf 'scale=%s gpu=%s result=%s\n' "${SCALE}" "${GPU_ID}" "${RESULT_JSON}" \
  > "${OUTPUT_ROOT}/worker_complete.txt"
echo "[worker] success scale=${SCALE} gpu=${GPU_ID} result=${RESULT_JSON}"
