#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  echo "Usage: $0 GPU_ID" >&2
  exit 2
fi

GPU_ID="$1"
if ! [[ "${GPU_ID}" =~ ^[0-9]+$ ]] || [[ "${GPU_ID}" == "4" ]]; then
  echo "GPU_ID must be one physical GPU other than forbidden GPU 4." >&2
  exit 2
fi

PYTHON_BIN="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
REPO_ROOT="/home/gaoya/Code_Video/DiffTrack-main"
CONTROL_DIR="${REPO_ROOT}/AAA_my_test/object_query_ablation_metrics/training_free_m1_control"
EXPERIMENT_ROOT="/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1"
OUTPUT_ROOT="${EXPERIMENT_ROOT}/training_free_m1_direct_enhancement_v2/test5_20case_5seed"
MANIFEST="${OUTPUT_ROOT}/test5_phase_bd_manifest.json"
RANKING="${EXPERIMENT_ROOT}/head_scopes_latest3350_with_random100.json"
TRACKS="${OUTPUT_ROOT}/frozen_baseline_tracks"
SELECTION="${OUTPUT_ROOT}/phase_b_selection.json"
STATUS="${OUTPUT_ROOT}/pipeline_status.json"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="/data/gaoya/agent-data/cache/huggingface"
export TORCH_HOME="/data/gaoya/agent-data/cache/torch"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "${OUTPUT_ROOT}/logs"

write_status() {
  local stage="$1"
  local state="$2"
  local started_at="$3"
  "${PYTHON_BIN}" - "${STATUS}" "${stage}" "${state}" "${started_at}" "${GPU_ID}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path, stage, state, started_at, gpu = sys.argv[1:]
payload = {
    "stage": stage,
    "state": state,
    "stage_started_at_utc": started_at,
    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    "physical_gpu": int(gpu),
    "visible_cuda_device": 0,
}
target = Path(path)
target.parent.mkdir(parents=True, exist_ok=True)
temporary = target.with_suffix(target.suffix + ".tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
temporary.replace(target)
PY
}

run_stage() {
  local stage="$1"
  shift
  local started_at
  started_at="$(date -u +%FT%TZ)"
  write_status "${stage}" running "${started_at}"
  echo "[$(date -u +%FT%TZ)] START ${stage}: $*" | tee -a "${OUTPUT_ROOT}/logs/pipeline.log"
  if "$@" 2>&1 | tee "${OUTPUT_ROOT}/logs/${stage}.log"; then
    write_status "${stage}" complete "${started_at}"
    echo "[$(date -u +%FT%TZ)] COMPLETE ${stage}" | tee -a "${OUTPUT_ROOT}/logs/pipeline.log"
  else
    local exit_code=${PIPESTATUS[0]}
    write_status "${stage}" failed "${started_at}"
    echo "[$(date -u +%FT%TZ)] FAILED ${stage} exit=${exit_code}" | tee -a "${OUTPUT_ROOT}/logs/pipeline.log"
    return "${exit_code}"
  fi
}

run_stage 00_prepare_manifest \
  "${PYTHON_BIN}" "${CONTROL_DIR}/prepare_test5_phase_bd.py" \
    --output-root "${OUTPUT_ROOT}" \
    --head-ranking-path "${RANKING}"

run_stage 01_clean_baselines \
  "${PYTHON_BIN}" "${CONTROL_DIR}/run_test5_phase_bd_batch.py" \
    --mode baselines \
    --manifest-path "${MANIFEST}" \
    --head-ranking-path "${RANKING}" \
    --tracks-root "${TRACKS}" \
    --output-root "${OUTPUT_ROOT}" \
    --device cuda

run_stage 02_frozen_baseline_tracks \
  "${PYTHON_BIN}" "${CONTROL_DIR}/prepare_test5_phase_bd_tracks.py" \
    --manifest-path "${MANIFEST}" \
    --tracks-root "${TRACKS}" \
    --device cuda

run_stage 03_phase_b_small_gain \
  "${PYTHON_BIN}" "${CONTROL_DIR}/run_test5_phase_bd_batch.py" \
    --mode phase_b \
    --manifest-path "${MANIFEST}" \
    --head-ranking-path "${RANKING}" \
    --tracks-root "${TRACKS}" \
    --output-root "${OUTPUT_ROOT}" \
    --device cuda

run_stage 04_phase_b_post_generation_selection \
  "${PYTHON_BIN}" "${CONTROL_DIR}/select_test5_phase_b_alpha.py" \
    --manifest-path "${MANIFEST}" \
    --output-root "${OUTPUT_ROOT}" \
    --selection-path "${SELECTION}" \
    --device cuda

run_stage 05_phase_d_windows \
  "${PYTHON_BIN}" "${CONTROL_DIR}/run_test5_phase_bd_batch.py" \
    --mode phase_d \
    --manifest-path "${MANIFEST}" \
    --head-ranking-path "${RANKING}" \
    --tracks-root "${TRACKS}" \
    --output-root "${OUTPUT_ROOT}" \
    --selection-path "${SELECTION}" \
    --device cuda

write_status all complete "$(date -u +%FT%TZ)"
echo "[$(date -u +%FT%TZ)] ALL STAGES COMPLETE" | tee -a "${OUTPUT_ROOT}/logs/pipeline.log"
