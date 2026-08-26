#!/usr/bin/env bash
set -euo pipefail

COMPARE="/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/compare_test70_rigidbench_new_vs_old.py"
BACKFILL="/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/run_test70_rigidbench_metric_backfill_gpu25.sh"
PYTHON="${PYTHON:-/home/gaoya/miniconda3/envs/sam/bin/python}"
ROOT="/data/gaoya/agent-data/outputs/physv_v2v_0819_rigidbench_strict_test70/runs"
LOG_ROOT="/data/gaoya/agent-data/outputs/physv_v2v_0819_rigidbench_strict_test70/logs/regression_gpu2"
export PYTHONNOUSERSITE=1
mkdir -p "$LOG_ROOT"

TASK_ARGS=()
for task_dir in "$ROOT"/*; do
  if [[ -d "$task_dir/generated" ]]; then
    TASK_ARGS+=(--task-id "$(basename "$task_dir")")
  fi
done

run_regression() {
  local gpu="$1"
  local metric="$2"
  local log="$LOG_ROOT/gpu${gpu}_${metric}.log"
  local report="$LOG_ROOT/gpu${gpu}_${metric}.json"
  {
    echo "[regression-retry] physical_gpu=$gpu metric=$metric started $(date -Is)"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$COMPARE" \
      "${TASK_ARGS[@]}" --metric "$metric" --all-complete --device cuda \
      --report "$report"
    echo "[regression-retry] physical_gpu=$gpu metric=$metric finished $(date -Is)"
  } >"$log" 2>&1
}

# Re-run only metrics whose previous report was incomplete/empty.  All
# workers are intentionally colocated on physical GPU 2 so GPU 5 remains
# available for other workloads.
run_regression 2 iou & pid_iou=$!
run_regression 2 ate3d & pid_ate3d=$!
run_regression 2 bgdrift & pid_bgdrift=$!
run_regression 2 l2 & pid_l2=$!
run_regression 2 chamfer & pid_chamfer=$!
run_regression 2 si_mse & pid_si_mse=$!
run_regression 2 iddrift & pid_iddrift=$!

status=0
for pid in "$pid_iou" "$pid_ate3d" "$pid_bgdrift" \
           "$pid_l2" "$pid_chamfer" "$pid_si_mse" "$pid_iddrift"; do
  wait "$pid" || status=1
done

if [[ "$status" -ne 0 ]]; then
  echo "[regression-retry] mismatch or error; pending backfill not started" >&2
  exit "$status"
fi

echo "[regression-retry] all retry workers matched; starting pending backfill"
exec "$BACKFILL"
