#!/usr/bin/env bash
set -euo pipefail

# Full 61-method × 70-case RigidBench-style evaluation.  Each physical GPU
# runs one complementary shard of every expensive model family; no GPU4/GPU5
# is used. Each family worker loads its models once and computes all metrics
# in that family before moving to the next case.

PYTHON="${PYTHON:-/home/gaoya/miniconda3/envs/sam/bin/python}"
EVALUATOR="/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/evaluate_test70_rigidbench_all_methods.py"
BUILDER="/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/build_test70_rigidbench_all_methods.py"
DASHBOARD="/data/gaoya/agent-data/physv_v2v_0819/visualization/hub/physv-v2v-0819-test70-no-event-timing-40step/dashboard.json"
STRICT_ROOT="/data/gaoya/AAA_test_video/physv_v2v_0819_strict"
OUTPUT_ROOT="/data/gaoya/agent-data/outputs/physv_v2v_0819_rigidbench_all_methods"
LOG_ROOT="${OUTPUT_ROOT}/logs/gpu13"
GPU_IDLE_MEMORY_MB="${GPU_IDLE_MEMORY_MB:-2000}"

export PYTHONNOUSERSITE=1
mkdir -p "$LOG_ROOT"

echo "[all-methods] initializing registry $(date -Is)"
"$PYTHON" "$EVALUATOR" \
  --dashboard "$DASHBOARD" \
  --strict-root "$STRICT_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --initialize
"$PYTHON" "$BUILDER"

gpu_memory_used() {
  nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}

echo "[all-methods] waiting for physical GPU1 and GPU3 memory <= ${GPU_IDLE_MEMORY_MB} MiB"
while true; do
  gpu1_used="$(gpu_memory_used 1)"
  gpu3_used="$(gpu_memory_used 3)"
  if [[ "$gpu1_used" =~ ^[0-9]+$ && "$gpu3_used" =~ ^[0-9]+$ \
        && "$gpu1_used" -le "$GPU_IDLE_MEMORY_MB" \
        && "$gpu3_used" -le "$GPU_IDLE_MEMORY_MB" ]]; then
    break
  fi
  echo "[all-methods] still waiting: GPU1=${gpu1_used} MiB GPU3=${gpu3_used} MiB $(date -Is)"
  sleep 60
done
echo "[all-methods] GPU1/GPU3 available: GPU1=${gpu1_used} MiB GPU3=${gpu3_used} MiB"

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
  run_worker 1 "$group" 0
  run_worker 3 "$group" 1
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
