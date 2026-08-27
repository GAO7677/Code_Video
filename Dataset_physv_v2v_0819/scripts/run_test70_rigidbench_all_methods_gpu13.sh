#!/usr/bin/env bash
set -euo pipefail

# Full 61-method × 70-case RigidBench-style evaluation. Each selected physical
# GPU runs one complementary shard of every expensive model family. GPU4/GPU5
# are never used. Each family worker loads its models once and computes all
# metrics in that family before moving to the next case.

PYTHON="${PYTHON:-/home/gaoya/miniconda3/envs/sam/bin/python}"
EVALUATOR="/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/evaluate_test70_rigidbench_all_methods.py"
BUILDER="/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/build_test70_rigidbench_all_methods.py"
DASHBOARD="/data/gaoya/agent-data/physv_v2v_0819/visualization/hub/physv-v2v-0819-test70-no-event-timing-40step/dashboard.json"
STRICT_ROOT="/data/gaoya/AAA_test_video/physv_v2v_0819_strict"
OUTPUT_ROOT="/data/gaoya/agent-data/outputs/physv_v2v_0819_rigidbench_all_methods"
GPU_IDLE_MEMORY_MB="${GPU_IDLE_MEMORY_MB:-2000}"
GPU_A="${GPU_A:-2}"
GPU_B="${GPU_B:-3}"
LOG_ROOT="${OUTPUT_ROOT}/logs/gpu${GPU_A}${GPU_B}"
BUILDER_INTERVAL_SEC="${BUILDER_INTERVAL_SEC:-60}"

export PYTHONNOUSERSITE=1
mkdir -p "$LOG_ROOT"

echo "[all-methods] initializing registry $(date -Is)"
"$PYTHON" "$EVALUATOR" \
  --dashboard "$DASHBOARD" \
  --strict-root "$STRICT_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --initialize
"$PYTHON" "$BUILDER"

# Keep the standalone data source current while workers are running.  The
# builder has its own lock so this is safe alongside an HTTP page refresh.
builder_loop() {
  while true; do
    sleep "$BUILDER_INTERVAL_SEC"
    "$PYTHON" "$BUILDER" >>"${LOG_ROOT}/builder.log" 2>&1 || \
      echo "[all-methods] periodic builder failed $(date -Is)" >>"${LOG_ROOT}/builder.log"
  done
}
builder_loop &
BUILDER_PID=$!
cleanup_builder() {
  kill "$BUILDER_PID" 2>/dev/null || true
  wait "$BUILDER_PID" 2>/dev/null || true
}
trap cleanup_builder EXIT INT TERM

gpu_memory_used() {
  nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}

echo "[all-methods] waiting for physical GPU${GPU_A} and GPU${GPU_B} memory <= ${GPU_IDLE_MEMORY_MB} MiB"
while true; do
  gpu_a_used="$(gpu_memory_used "$GPU_A")"
  gpu_b_used="$(gpu_memory_used "$GPU_B")"
  if [[ "$gpu_a_used" =~ ^[0-9]+$ && "$gpu_b_used" =~ ^[0-9]+$ \
        && "$gpu_a_used" -le "$GPU_IDLE_MEMORY_MB" \
        && "$gpu_b_used" -le "$GPU_IDLE_MEMORY_MB" ]]; then
    break
  fi
  echo "[all-methods] still waiting: GPU${GPU_A}=${gpu_a_used} MiB GPU${GPU_B}=${gpu_b_used} MiB $(date -Is)"
  sleep 60
done
echo "[all-methods] GPU${GPU_A}/GPU${GPU_B} available: GPU${GPU_A}=${gpu_a_used} MiB GPU${GPU_B}=${gpu_b_used} MiB"

run_worker() {
  local gpu="$1"
  local group="$2"
  local shard="$3"
  local log="${LOG_ROOT}/gpu${gpu}_${group}_shard${shard}.log"
  echo "[all-methods] starting gpu=${gpu} group=${group} shard=${shard} log=${log}"
  (
    echo "[all-methods] physical_gpu=${gpu} group=${group} shard=${shard}/2 started $(date -Is)"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$EVALUATOR" \
      --dashboard "$DASHBOARD" \
      --strict-root "$STRICT_ROOT" \
      --output-root "$OUTPUT_ROOT" \
      --group "$group" \
      --shard-index "$shard" \
      --shard-count 2 \
      --gpu-label "$gpu" \
      --device cuda \
      --resume
    echo "[all-methods] physical_gpu=${gpu} group=${group} shard=${shard}/2 finished $(date -Is)"
  ) >"$log" 2>&1 &
  WORKER_PIDS+=("$!")
}

WORKER_PIDS=()
for group in mask depth track image; do
  # shard 0 and shard 1 are complementary; do not launch the same shard on
  # both GPUs, otherwise all work would be duplicated.
  run_worker "$GPU_A" "$group" 0
  run_worker "$GPU_B" "$group" 1
done

status=0
for pid in "${WORKER_PIDS[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

# Always rebuild the page after workers finish, including a partial run, so
# an interrupted run remains visible and can be resumed without losing work.
"$PYTHON" "$BUILDER" || status=1
echo "[all-methods] all workers finished status=${status} $(date -Is)"
exit "$status"
