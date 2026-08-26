#!/usr/bin/env bash
set -uo pipefail

PYTHON="/home/gaoya/miniconda3/envs/sam/bin/python"
SCRIPT="/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/run_test70_rigidbench_metrics_only.py"
LOG_ROOT="/data/gaoya/agent-data/outputs/physv_v2v_0819_rigidbench_strict_test70/logs/metrics_gpu25"
PYTHONPATH_VALUE="/home/gaoya/Code_Video/Dataset_physv_v2v_0819/RigidBench/src:/home/gaoya/Code_Video/Dataset_physv_v2v_0819/RigidBench/vendor/Video-Depth-Anything"
mkdir -p "$LOG_ROOT"

pids=()
for gpu in 2 5; do
  for slot in 0 1 2 3 4; do
    worker="gpu${gpu}_slot${slot}"
    log="$LOG_ROOT/${worker}.log"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1 PYTHONPATH="$PYTHONPATH_VALUE" \
      "$PYTHON" "$SCRIPT" --worker-id "$worker" >> "$log" 2>&1 &
    pids+=("$!")
  done
done

echo "Started ${#pids[@]} metric-only workers on physical GPU2/GPU5 (5 per GPU)."
rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || rc=1
done
exit "$rc"
