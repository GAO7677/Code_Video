#!/usr/bin/env bash
set -euo pipefail

BASE=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box
RUNNER="${BASE}/run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh"
INFER_SCRIPT="${BASE}/visualize_scheme_c_text_noun_attention_x0_v2v.py"
ANALYZER="${BASE}/analyze_text_attention_boundary_metrics.py"
PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
CHECKPOINT_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_raw49f_scheme_c_entity_caption_physical_fresh_20260714T174707Z/checkpoints
CASE_LIST=/data/gaoya/agent-data/cache/scheme_c_text_noun_attention_physiq4_20260715/input_jsons.txt
OUTPUT_ROOT=/data/gaoya/agent-data/outputs/AAA_physv/scheme_c_text_noun_attention_sharedscale_metrics_step2500_3500_residual_sweep_physiq4_20260715
LOG_ROOT="${OUTPUT_ROOT}/_logs"
EXPECTED_CASES=4

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

verify_combo() {
  local combo_root="$1"
  local checkpoint="$2"
  local result_json
  result_json="$(find "${combo_root}" -type f -name result.json | sort | head -n 1)"
  if [[ -z "${result_json}" ]]; then
    echo "[attention-sweep] result.json not found under ${combo_root}" >&2
    return 1
  fi
  "${PYTHON_BIN}" - "${result_json}" "${checkpoint}" "${EXPECTED_CASES}" <<'PY'
import json
import sys
from pathlib import Path

result_path = Path(sys.argv[1])
checkpoint = str(Path(sys.argv[2]).resolve())
expected = int(sys.argv[3])
payload = json.loads(result_path.read_text(encoding="utf-8"))
entries = payload.get("entries", [])
if not (
    str(Path(payload.get("checkpoint_dir", "")).resolve()) == checkpoint
    and int(payload.get("num_total", -1)) == expected
    and int(payload.get("num_success", -1)) == expected
    and int(payload.get("num_failed", -1)) == 0
    and int(payload.get("num_skipped", -1)) == 0
    and len(entries) == expected
):
    raise SystemExit(f"incomplete attention batch: {payload}")
for entry in entries:
    attention = entry.get("text_noun_attention", {})
    manifest_path = Path(attention.get("manifest", ""))
    if not manifest_path.is_file():
        raise SystemExit(f"missing attention manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not (
        len(manifest.get("capture_progress_indices", [])) == 5
        and len(manifest.get("predicted_x0_videos", {})) == 5
        and manifest.get("raw_all_maps")
        and Path(manifest["raw_all_maps"]).is_file()
        and manifest.get("nouns")
    ):
        raise SystemExit(f"incomplete attention artifacts: {manifest_path}")
print(f"verified attention artifacts: {result_path}")
PY
}

run_combo() {
  local step="$1"
  local scale="$2"
  local tag="${scale/./p}"
  local checkpoint="${CHECKPOINT_ROOT}/${step}"
  local combo_root="${OUTPUT_ROOT}/${step}/object_residual_${tag}x"
  local method="scheme_c_entity_attention_${step}_object_residual_${tag}x"
  local log="${LOG_ROOT}/${step}_object_residual_${tag}x_gpu6.log"

  mkdir -p "${combo_root}"
  printf '%s\n' \
    "checkpoint=${checkpoint}" \
    "input_list=${CASE_LIST}" \
    "object_branch_residual_scale=${scale}" \
    "capture_progress_indices=0,10,20,30,39" \
    "capture_remaining_steps=40,30,20,10,1" \
    "save_all_attention_maps=1" \
    "shared_context_future_scale=1" \
    "context_frames=8" \
    "output_frames=49" \
    "resolution=896x512" \
    > "${combo_root}/attention_sweep_config.txt"

  echo "[attention-sweep] start step=${step} scale=${scale} output=${combo_root}"
  GPU_PAIR="6,6" \
  AUTO_SPLIT_INPUT=0 \
  TEST_JSON_TXT="${CASE_LIST}" \
  WEIGHTS_ROOT="${checkpoint}" \
  METHOD_NAME="${method}" \
  OUTPUT_ROOT="${combo_root}" \
  OUTPUT_FRAMES=49 \
  CTX=8 \
  NUM_INFERENCE_STEPS=40 \
  CFG_SCALE=5.0 \
  SEED=42 \
  FORCE=1 \
  COMPACT_OBJECT_CONTEXT_SLOTS=1 \
  OBJECT_ADAPTER_MLP_RESIDUAL_MAX_RATIO=3.0 \
  OBJECT_BRANCH_RESIDUAL_SCALE="${scale}" \
  OBJECT_BRANCH_RATIO_GUARD_MAX_RATIO=0.30 \
  OBJECT_BRANCH_RATIO_GUARD_MAX_BLOCK_ID=-1 \
  INFER_SCRIPT_OVERRIDE="${INFER_SCRIPT}" \
  EXTRA_INFER_ARGS="--attention-capture-progress-indices auto5 --attention-query-chunk 256 --attention-save-all-maps 1 --attention-shared-context-future-scale 1" \
  bash "${RUNNER}" 2>&1 | tee -a "${log}"

  verify_combo "${combo_root}" "${checkpoint}"
  "${PYTHON_BIN}" "${ANALYZER}" "${combo_root}"
  printf 'step=%s scale=%s cases=%s\n' "${step}" "${scale}" "${EXPECTED_CASES}" \
    > "${combo_root}/worker_complete.txt"
  echo "[attention-sweep] success step=${step} scale=${scale}"
}

for step in step-002500 step-003500; do
  for scale in 1.0 1.5 2.0; do
    run_combo "${step}" "${scale}"
  done
done

echo "[attention-sweep] all six combinations completed"
