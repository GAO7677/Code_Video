#!/usr/bin/env bash
set -u -o pipefail

if [[ $# -lt 2 || ! "$1" =~ ^(0|1|2|3)$ ]]; then
  echo "usage: $0 <gpu:0|1|2|3> <lambda-tag:step> [...]" >&2
  exit 2
fi

GPU=$1
shift
PYTHON=/data/gaoya/agent-data/envs/physrvg-full-sa/bin/python
WATCHER=/home/gaoya/code_V2V_baselines/PhysRVG-main/scripts_mytrain/evaluation/test70/watchers/watch_lineage_test70.py
LOG=/data/gaoya/agent-data/outputs/context_noise_interpolation_20260923/rigidbench-remaining-gpu${GPU}.log

mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "[$(date -u +%FT%TZ)] queue start gpu=$GPU tasks=$*"

for item in "$@"; do
  tag=${item%%:*}
  step=${item##*:}
  task_id=$(printf 'full_sa_physrvg_raw_rl_lora_context_noise_lambda%s_initial0907_generic_full_2175_clean_context_20260923__step-%06d' "$tag" "$step")
  completed=0
  for attempt in 1 2 3; do
    echo "[$(date -u +%FT%TZ)] start gpu=$GPU attempt=$attempt task=$task_id"
    if CUDA_VISIBLE_DEVICES="$GPU" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
      "$PYTHON" -u "$WATCHER" --task "$task_id" --phase metrics --gpu "$GPU"; then
      completed=1
      echo "[$(date -u +%FT%TZ)] complete gpu=$GPU task=$task_id"
      break
    fi
    echo "[$(date -u +%FT%TZ)] failed gpu=$GPU attempt=$attempt task=$task_id"
    sleep $((attempt * 15))
  done
  if [[ "$completed" != 1 ]]; then
    echo "[$(date -u +%FT%TZ)] giving up after 3 attempts task=$task_id"
  fi
done

echo "[$(date -u +%FT%TZ)] queue complete gpu=$GPU"
