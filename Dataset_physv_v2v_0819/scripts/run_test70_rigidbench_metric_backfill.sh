#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/gaoya/miniconda3/envs/sam/bin/python}"
SCRIPT="/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/run_test70_rigidbench_metric_backfill.py"
export PYTHONPATH="/home/gaoya/Code_Video/Code_data/Code_try0526:/home/gaoya/Code_Video/Dataset_physv_v2v_0819/RigidBench/src:/home/gaoya/Code_Video/Dataset_physv_v2v_0819/RigidBench/vendor/Video-Depth-Anything${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1

exec "$PYTHON" "$SCRIPT" "$@"
