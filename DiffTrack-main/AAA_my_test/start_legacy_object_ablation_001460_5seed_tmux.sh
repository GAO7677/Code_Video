#!/usr/bin/env bash
set -euo pipefail

SESSION="legacy_object_ablation_001460_5seed"
ROOT="/home/gaoya/Code_Video/DiffTrack-main"
BUILDER="${ROOT}/AAA_my_test/build_legacy_object_ablation_001460_5seed_manifest.py"
WORKER="${ROOT}/AAA_my_test/run_legacy_object_ablation_001460_5seed_gpu.sh"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

cd "${ROOT}"
"${PYTHON}" "${BUILDER}"
tmux new-session -d -s "${SESSION}" -n gpu0 "bash '${WORKER}' 0 90094 13248; exec bash"
tmux new-window -t "${SESSION}" -n gpu1 "bash '${WORKER}' 1 68613; exec bash"
tmux new-window -t "${SESSION}" -n gpu2 "bash '${WORKER}' 2 35075; exec bash"
tmux new-window -t "${SESSION}" -n gpu3 "bash '${WORKER}' 3 32466; exec bash"
tmux select-window -t "${SESSION}:gpu0"

echo "started ${SESSION} on GPU0/1/2/3"
echo "tube assignment: gpu0=90094,13248 gpu1=68613 gpu2=35075 gpu3=32466"
echo "fixed assignment: 120 tasks sharded evenly, 30 per GPU"
