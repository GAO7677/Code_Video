#!/usr/bin/env bash
set -u -o pipefail

if [[ $# -ne 1 || ( "$1" != "0" && "$1" != "1" ) ]]; then
  echo "usage: $0 <0|1>" >&2
  exit 2
fi

GPU=$1
PYTHON=/data/gaoya/agent-data/envs/physrvg-full-sa/bin/python
WATCHER=/home/gaoya/code_V2V_baselines/PhysRVG-main/scripts_mytrain/evaluation/test70/watchers/watch_lineage_test70.py
MODEL_KEY=full_sa_physrvg_raw_rl_lora_context_noise_lambda025_initial0907_generic_full_2175_clean_context_20260923
LOG_ROOT=/data/gaoya/agent-data/outputs/context_noise_interpolation_20260923/test70_lambda025_logs
mkdir -p "$LOG_ROOT"

if [[ "$GPU" == "0" ]]; then
  STEPS=(500 1500 2500)
else
  STEPS=(1000 2000)
fi

run_phase() {
  local step=$1
  local phase=$2
  local task_id
  task_id=$(printf '%s__step-%06d' "$MODEL_KEY" "$step")
  echo "[$(date -u +%FT%TZ)] gpu=$GPU phase=$phase task=$task_id"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    "$PYTHON" -u "$WATCHER" --task "$task_id" --phase "$phase" --gpu "$GPU" \
    >> "$LOG_ROOT/gpu${GPU}.log" 2>&1
}

for step in "${STEPS[@]}"; do
  if ! run_phase "$step" generation; then
    echo "[$(date -u +%FT%TZ)] generation failed: step=$step" >> "$LOG_ROOT/gpu${GPU}.log"
  fi
done

for step in "${STEPS[@]}"; do
  if ! run_phase "$step" metrics; then
    echo "[$(date -u +%FT%TZ)] metrics failed: step=$step" >> "$LOG_ROOT/gpu${GPU}.log"
  fi
done

echo "[$(date -u +%FT%TZ)] gpu=$GPU queue complete" >> "$LOG_ROOT/gpu${GPU}.log"
