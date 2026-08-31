#!/usr/bin/env bash
set -uo pipefail

# Resume every missing all-methods RigidBench cell on the seven allowed local
# GPUs.  Each physical GPU owns one model family/shard so expensive model
# families never overlap on the same card.  Local GPU4 is forbidden.

PYTHON="/home/gaoya/miniconda3/envs/sam/bin/python"
PUBLISH_PYTHON="/data/gaoya/agent-data/envs/physrvg-full-sa/bin/python"
EVALUATOR="/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/evaluate_test70_rigidbench_all_methods.py"
BUILDER="/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/build_test70_rigidbench_all_methods.py"
PUBLISHER="/home/gaoya/code_V2V_baselines/PhysRVG-main/scripts_mytrain/evaluation/test70/publishing/publish_test70_resume_to_page.py"
DASHBOARD="/data/gaoya/agent-data/physv_v2v_0819/visualization/hub/physv-v2v-0819-test70-no-event-timing-40step/dashboard.json"
STRICT_ROOT="/data/gaoya/AAA_test_video/physv_v2v_0819_strict"
OUTPUT_ROOT="/data/gaoya/agent-data/outputs/physv_v2v_0819_rigidbench_all_methods"
LOG_ROOT="${OUTPUT_ROOT}/logs/gpu0123567"

export PYTHONNOUSERSITE=1
mkdir -p "$LOG_ROOT"

echo "[rigidbench-7gpu] initialize registry $(date -Is)"
status=0
if ! "$PYTHON" "$EVALUATOR" \
  --dashboard "$DASHBOARD" \
  --strict-root "$STRICT_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --initialize; then
  exit 1
fi
"$PYTHON" "$BUILDER" || status=1

builder_loop() {
  while true; do
    sleep 60
    "$PYTHON" "$BUILDER" >>"${LOG_ROOT}/builder.log" 2>&1 || true
  done
}
builder_loop &
builder_pid=$!
cleanup_builder() {
  kill "$builder_pid" 2>/dev/null || true
  wait "$builder_pid" 2>/dev/null || true
}
trap cleanup_builder EXIT INT TERM

run_worker() {
  local gpu="$1"
  local group="$2"
  local shard_index="$3"
  local shard_count="$4"
  local log_path="${LOG_ROOT}/gpu${gpu}_${group}_shard${shard_index}.log"
  echo "[rigidbench-7gpu] start gpu=${gpu} group=${group} shard=${shard_index}/${shard_count}"
  (
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
      "$PYTHON" "$EVALUATOR" \
      --dashboard "$DASHBOARD" \
      --strict-root "$STRICT_ROOT" \
      --output-root "$OUTPUT_ROOT" \
      --group "$group" \
      --shard-index "$shard_index" \
      --shard-count "$shard_count" \
      --gpu-label "$gpu" \
      --device cuda \
      --resume
  ) >"$log_path" 2>&1 &
  worker_pids+=("$!")
}

for pass in 1 2; do
  echo "[rigidbench-7gpu] pass=${pass} $(date -Is)"
  worker_pids=()
  run_worker 0 mask 0 2
  run_worker 1 mask 1 2
  run_worker 2 depth 0 2
  run_worker 3 depth 1 2
  run_worker 5 track 0 2
  run_worker 6 track 1 2
  run_worker 7 image 0 1

  for pid in "${worker_pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  "$PYTHON" "$BUILDER" || status=1
done

"$PUBLISH_PYTHON" "$PUBLISHER" || status=1
echo "[rigidbench-7gpu] finished status=${status} $(date -Is)"
exit "$status"
