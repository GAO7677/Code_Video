#!/usr/bin/env bash
set -euo pipefail

# Keep the all-methods RigidBench canonical snapshot fresh for a run that was
# started before the periodic builder was added to its launcher.
PYTHON="${PYTHON:-/home/gaoya/miniconda3/envs/sam/bin/python}"
BUILDER="/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/build_test70_rigidbench_all_methods.py"
OUTPUT_ROOT="/data/gaoya/agent-data/outputs/physv_v2v_0819_rigidbench_all_methods"
LOG="${OUTPUT_ROOT}/logs/gpu13/builder_watcher.log"
INTERVAL_SEC="${INTERVAL_SEC:-60}"
PARENT_PID="${1:-}"

if [[ ! "$PARENT_PID" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 RIGIDBENCH_LAUNCHER_PID" >&2
  exit 2
fi

mkdir -p "$(dirname "$LOG")"
while kill -0 "$PARENT_PID" 2>/dev/null; do
  "$PYTHON" "$BUILDER" >>"$LOG" 2>&1 || \
    echo "[watcher] builder failed $(date -Is)" >>"$LOG"
  sleep "$INTERVAL_SEC"
done

# Publish one final snapshot after the launcher exits.
"$PYTHON" "$BUILDER" >>"$LOG" 2>&1 || true
