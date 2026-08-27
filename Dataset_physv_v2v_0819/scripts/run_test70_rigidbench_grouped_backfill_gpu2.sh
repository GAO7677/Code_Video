#!/usr/bin/env bash
set -euo pipefail

RUNNER="/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/run_test70_rigidbench_grouped_backfill.py"
BUILDER="/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/build_test70_rigidbench_metrics.py"
PYTHON="${PYTHON:-/home/gaoya/miniconda3/envs/sam/bin/python}"
INPUT_ROOT="/data/gaoya/agent-data/outputs/physv_v2v_0819_rigidbench_strict_test70"
LOG_ROOT="${INPUT_ROOT}/logs/metric_backfill_gpu2_grouped"
PROGRESS_ROOT="${LOG_ROOT}/progress"
export PYTHONNOUSERSITE=1
mkdir -p "$LOG_ROOT" "$PROGRESS_ROOT"

TASK_ARGS=()
for task_dir in "$INPUT_ROOT"/runs/*; do
  if [[ -d "$task_dir/generated" && -d "$task_dir/metrics" ]]; then
    TASK_ARGS+=(--task-id "$(basename "$task_dir")")
  fi
done

run_group() {
  local group="$1"
  local log="$LOG_ROOT/gpu2_${group}.log"
  {
    echo "[backfill-grouped] physical_gpu=2 group=$group started $(date -Is)"
    CUDA_VISIBLE_DEVICES=2 "$PYTHON" "$RUNNER" \
      "${TASK_ARGS[@]}" --group "$group" --device cuda \
      --progress-dir "$PROGRESS_ROOT" --resume
    echo "[backfill-grouped] physical_gpu=2 group=$group finished $(date -Is)"
  } >"$log" 2>&1
}

# Four workers replace the old ten independent metric processes.  Each worker
# shares the expensive prediction extraction among metrics in its group.
run_group mask & pid_mask=$!
run_group depth & pid_depth=$!
run_group track & pid_track=$!
run_group image & pid_image=$!

status=0
for pid in "$pid_mask" "$pid_depth" "$pid_track" "$pid_image"; do
  wait "$pid" || status=1
done

if [[ "$status" -eq 0 ]]; then
  "$PYTHON" "$BUILDER"
fi
exit "$status"
