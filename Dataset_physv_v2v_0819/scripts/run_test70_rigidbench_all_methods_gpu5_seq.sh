#!/usr/bin/env bash
set -uo pipefail

# Resume the dynamic dashboard inventory serially on physical GPU5.  GPU4 and
# SSH118 GPU0-3 are intentionally outside this launcher.
PYTHON="/home/gaoya/miniconda3/envs/sam/bin/python"
EVALUATOR="/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/evaluate_test70_rigidbench_all_methods.py"
BUILDER="/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/build_test70_rigidbench_all_methods.py"
DASHBOARD="/data/gaoya/agent-data/physv_v2v_0819/visualization/hub/physv-v2v-0819-test70-no-event-timing-40step/dashboard.json"
STRICT_ROOT="/data/gaoya/AAA_test_video/physv_v2v_0819_strict"
OUTPUT_ROOT="/data/gaoya/agent-data/outputs/physv_v2v_0819_rigidbench_all_methods"
LOG_ROOT="${OUTPUT_ROOT}/logs/gpu5-seq"
LOG_FILE="${LOG_ROOT}/run.log"

mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[all-methods-gpu5] start $(date -Is)"

# Publish partial cells while a long metric family is running.  The builder
# takes an advisory lock, so this is safe beside the evaluator's writers.
builder_loop() {
  while true; do
    sleep 60
    "$PYTHON" "$BUILDER" >>"${LOG_ROOT}/builder.log" 2>&1 || true
  done
}
builder_loop &
BUILDER_PID=$!
cleanup_builder() {
  kill "$BUILDER_PID" 2>/dev/null || true
  wait "$BUILDER_PID" 2>/dev/null || true
}
trap cleanup_builder EXIT INT TERM

status=0
if ! "$PYTHON" "$EVALUATOR" \
  --dashboard "$DASHBOARD" \
  --strict-root "$STRICT_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --initialize; then
  status=1
fi
if ! "$PYTHON" "$BUILDER"; then
  status=1
fi

for group in mask depth track image; do
  echo "[all-methods-gpu5] group=${group} start $(date -Is)"
  if ! CUDA_VISIBLE_DEVICES=5 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
    "$PYTHON" "$EVALUATOR" \
    --dashboard "$DASHBOARD" \
    --strict-root "$STRICT_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --group "$group" \
    --shard-index 0 \
    --shard-count 1 \
    --gpu-label 5 \
    --device cuda \
    --resume; then
    status=1
  fi
  if ! "$PYTHON" "$BUILDER"; then
    status=1
  fi
  echo "[all-methods-gpu5] group=${group} end $(date -Is)"
done

echo "[all-methods-gpu5] end status=${status} $(date -Is)"
exit "$status"
