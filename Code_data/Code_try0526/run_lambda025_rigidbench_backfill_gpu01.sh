#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 || ( "$1" != "0" && "$1" != "1" ) ]]; then
  echo "usage: $0 <0|1> <step[,step...]>" >&2
  exit 2
fi

GPU=$1
IFS=, read -r -a STEPS <<< "$2"
PYTHON=/data/gaoya/agent-data/envs/physrvg-full-sa/bin/python
WATCHER=/home/gaoya/code_V2V_baselines/PhysRVG-main/scripts_mytrain/evaluation/test70/watchers/watch_lineage_test70.py
MODEL_KEY=full_sa_physrvg_raw_rl_lora_context_noise_lambda025_initial0907_generic_full_2175_clean_context_20260923
LOG=/data/gaoya/agent-data/outputs/context_noise_interpolation_20260923/lambda025-rigidbench-gpu${GPU}.log

mkdir -p "$(dirname "$LOG")"
exec >> "$LOG" 2>&1
echo "[$(date -u +%FT%TZ)] start RigidBench backfill gpu=$GPU steps=$2"

for step in "${STEPS[@]}"; do
  task_id=$(printf '%s__step-%06d' "$MODEL_KEY" "$step")
  echo "[$(date -u +%FT%TZ)] start task=$task_id"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
    "$PYTHON" -u "$WATCHER" --task "$task_id" --phase metrics --gpu "$GPU"
  echo "[$(date -u +%FT%TZ)] complete task=$task_id"
done

echo "[$(date -u +%FT%TZ)] queue complete gpu=$GPU"
