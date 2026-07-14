#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box
INFER_SH="${SCRIPT_DIR}/run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh"
BENCH_SH="${SCRIPT_DIR}/bench.sh"
CHECKPOINT_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_raw49f_kubric_openvid_replay_sourceaware_fp32gate_fixedctx8_init3500_save500_keepall_20260713T090024Z/checkpoints
INPUT_LIST=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt
OUTPUT_ROOT=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/physicIQ/train_stage1b_mixdataset
EVAL_ALL="${SCRIPT_DIR}/AAAeval.txt"
EVAL_PHYSIQ="${SCRIPT_DIR}/AAAevalphysiq.txt"
RUN_LOG_ROOT="${OUTPUT_ROOT}/_sweep_logs"
STEP=step-003500
EXPECTED_CASES=67

mkdir -p "${OUTPUT_ROOT}" "${RUN_LOG_ROOT}"

append_unique() {
  local line="$1"
  local file="$2"
  touch "${file}"
  if ! grep -Fqx -- "${line}" "${file}"; then
    printf '%s\n' "${line}" >> "${file}"
  fi
}

verify_result() {
  local result_json="$1"
  /home/gaoya/miniconda3/envs/wan-cu128/bin/python - "${result_json}" "${EXPECTED_CASES}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
with path.open(encoding="utf-8") as handle:
    payload = json.load(handle)
ok = (
    int(payload.get("num_total", -1)) == expected
    and int(payload.get("num_success", -1)) == expected
    and int(payload.get("num_failed", -1)) == 0
)
if not ok:
    raise SystemExit(
        f"incomplete result: total={payload.get('num_total')} "
        f"success={payload.get('num_success')} failed={payload.get('num_failed')}"
    )
PY
}

run_one() {
  local scale="$1"
  local scale_tag="${scale/./p}"
  local checkpoint="${CHECKPOINT_ROOT}/${STEP}"
  local combo_root="${OUTPUT_ROOT}/${STEP}/object_residual_${scale_tag}x"
  local method="mixdataset_${STEP}_object_residual_${scale_tag}x"
  local log="${RUN_LOG_ROOT}/${STEP}_object_residual_${scale_tag}x.log"
  local leaf result_json

  if [[ ! -s "${checkpoint}/checkpoint.safetensors" ]]; then
    echo "[sweep] missing checkpoint: ${checkpoint}" >&2
    return 1
  fi

  mkdir -p "${combo_root}"
  printf '%s\n' \
    "step=${STEP}" \
    "checkpoint=${checkpoint}" \
    "object_branch_residual_scale=${scale}" \
    "input_list=${INPUT_LIST}" \
    "expected_cases=${EXPECTED_CASES}" \
    "seed=42" \
    "cfg_scale=5.0" \
    "num_inference_steps=40" \
    "context_frames=8" \
    "output_frames=49" \
    > "${combo_root}/sweep_config.txt"

  echo "[sweep] start step=${STEP} scale=${scale} output=${combo_root}"
  GPU_PAIR="6,6" \
  TEST_JSON_TXT="${INPUT_LIST}" \
  WEIGHTS_ROOT="${checkpoint}" \
  METHOD_NAME="${method}" \
  OUTPUT_ROOT="${combo_root}" \
  OUTPUT_FRAMES=49 \
  CTX=8 \
  NUM_INFERENCE_STEPS=40 \
  CFG_SCALE=5.0 \
  SEED=42 \
  COMPACT_OBJECT_CONTEXT_SLOTS=1 \
  OBJECT_ADAPTER_MLP_RESIDUAL_MAX_RATIO=3.0 \
  OBJECT_BRANCH_RESIDUAL_SCALE="${scale}" \
  OBJECT_BRANCH_RATIO_GUARD_MAX_RATIO=0.20 \
  OBJECT_BRANCH_RATIO_GUARD_MAX_BLOCK_ID=-1 \
  bash "${INFER_SH}" 2>&1 | tee -a "${log}"

  result_json="$(find "${combo_root}" -mindepth 2 -maxdepth 2 -type f -name result.json | sort | head -n 1)"
  if [[ -z "${result_json}" ]]; then
    echo "[sweep] no result.json found under ${combo_root}" >&2
    return 1
  fi
  verify_result "${result_json}"
  leaf="$(dirname "${result_json}")"
  append_unique "${leaf}" "${EVAL_ALL}"
  append_unique "${leaf}" "${EVAL_PHYSIQ}"
  echo "[sweep] completed step=${STEP} scale=${scale} leaf=${leaf}"
}

for scale in 1.0 1.5 2.0; do
  run_one "${scale}"
done

echo "[sweep] all step-003500 inference groups completed; starting metrics"
CUDA_VISIBLE_DEVICES=0 \
BENCH_CUDA_VISIBLE_DEVICES=0 \
bash "${BENCH_SH}" "${EVAL_PHYSIQ}" 2>&1 | tee -a "${RUN_LOG_ROOT}/bench_AAAevalphysiq_step-003500.log"

echo "[sweep] step-003500 inference and metrics completed"
