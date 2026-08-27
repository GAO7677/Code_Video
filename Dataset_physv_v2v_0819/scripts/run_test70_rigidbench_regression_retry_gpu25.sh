#!/usr/bin/env bash
set -euo pipefail

COMPARE="/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/compare_test70_rigidbench_grouped.py"
BACKFILL="/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/run_test70_rigidbench_metric_backfill_gpu25.sh"
PYTHON="${PYTHON:-/home/gaoya/miniconda3/envs/sam/bin/python}"
ROOT="/data/gaoya/agent-data/outputs/physv_v2v_0819_rigidbench_strict_test70/runs"
LOG_ROOT="/data/gaoya/agent-data/outputs/physv_v2v_0819_rigidbench_strict_test70/logs/regression_gpu2_grouped"
REPORT_ROOT="$LOG_ROOT/reports"
PROGRESS_ROOT="$LOG_ROOT/progress"
export PYTHONNOUSERSITE=1
mkdir -p "$LOG_ROOT" "$REPORT_ROOT" "$PROGRESS_ROOT"

TASK_ARGS=()
for task_dir in "$ROOT"/*; do
  if [[ -d "$task_dir/generated" ]]; then
    TASK_ARGS+=(--task-id "$(basename "$task_dir")")
  fi
done

run_regression_group() {
  local gpu="$1"
  local group="$2"
  local log="$LOG_ROOT/gpu${gpu}_${group}.log"
  {
    echo "[regression-grouped] physical_gpu=$gpu group=$group started $(date -Is)"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$COMPARE" \
      "${TASK_ARGS[@]}" --group "$group" --all-complete --device cuda \
      --report-dir "$REPORT_ROOT" --progress-dir "$PROGRESS_ROOT"
    echo "[regression-grouped] physical_gpu=$gpu group=$group finished $(date -Is)"
  } >"$log" 2>&1
}

# Group workers share prediction extraction within each case.  They are still
# colocated on physical GPU 2, while GPU 5 remains available for other work.
run_regression_group 2 mask & pid_mask=$!
run_regression_group 2 depth & pid_depth=$!
run_regression_group 2 identity & pid_identity=$!

status=0
for pid in "$pid_mask" "$pid_depth" "$pid_identity"; do
  wait "$pid" || status=1
done

if [[ "$status" -ne 0 ]]; then
  echo "[regression-grouped] mismatch or error; pending backfill not started" >&2
  exit "$status"
fi

echo "[regression-grouped] all grouped regression workers matched"
# Keep this opt-in so a successful regression never launches the old
# ungrouped ten-process backfill unexpectedly.  The backfill can be replaced
# by the same grouped implementation after this regression is validated.
if [[ "${RUN_BACKFILL_AFTER_REGRESSION:-0}" == "1" ]]; then
  exec "$BACKFILL"
fi
