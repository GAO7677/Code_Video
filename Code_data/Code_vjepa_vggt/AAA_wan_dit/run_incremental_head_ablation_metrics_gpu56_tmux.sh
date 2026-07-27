#!/usr/bin/env bash
set -euo pipefail

# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_incremental_head_ablation_metrics_gpu56_tmux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION="${SESSION:-wan_head_ablation_incremental_metrics_gpu56}"
CONFIG="${SCRIPT_DIR}/head_ablation_allblocks_test5_gpu56.json"
RUN_ROOT=/data/gaoya/AAA_test_video/0623/test/v2v_wan_test5/_pipeline/metrics_incremental
INPUT_LIST=/data/gaoya/AAA_test_video/0623/test/v2v_wan_test5/_pipeline/input_unique.txt
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
WORKER="${SCRIPT_DIR}/run_incremental_head_ablation_metric_worker.sh"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi
for gpu in 5 6; do
  nvidia-smi -i "${gpu}" >/dev/null
done
if nvidia-smi -i 7 >/dev/null 2>&1; then
  echo "GPU7 is present but this launcher is intentionally configured for GPU5/6."
else
  echo "GPU7 is unavailable; using GPU5/6 only."
fi

mkdir -p "${RUN_ROOT}/logs"
tmux new-session -d -s "${SESSION}" -n coordinator \
  "${PYTHON} '${SCRIPT_DIR}/enqueue_incremental_head_ablation_metrics.py' \
  --config '${CONFIG}' --output-root '${RUN_ROOT}' 2>&1 | \
  tee -a '${RUN_ROOT}/logs/coordinator.log'"

worker_index=0
for gpu in 5 6; do
  for index in $(seq 0 7); do
    name="g${gpu}_cpu${index}"
    tmux new-window -t "${SESSION}" -n "${name}" \
      "bash '${WORKER}' '${gpu}' cpu '${name}' '${RUN_ROOT}' '${INPUT_LIST}' 49140"
    worker_index=$((worker_index + 1))
  done
done

for gpu in 5 6; do
  name="g${gpu}_common"
  tmux new-window -t "${SESSION}" -n "${name}" \
    "bash '${WORKER}' '${gpu}' gpu_common '${name}' '${RUN_ROOT}' '${INPUT_LIST}' 35000"
  worker_index=$((worker_index + 1))
  name="g${gpu}_heavy"
  tmux new-window -t "${SESSION}" -n "${name}" \
    "bash '${WORKER}' '${gpu}' gpu_heavy '${name}' '${RUN_ROOT}' '${INPUT_LIST}' 25000"
  worker_index=$((worker_index + 1))
done

tmux new-window -t "${SESSION}" -n summary \
  "bash '${SCRIPT_DIR}/watch_incremental_head_ablation_metric_summary.sh' \
  '${RUN_ROOT}' '${INPUT_LIST}' '${worker_index}' 2>&1 | \
  tee -a '${RUN_ROOT}/logs/summary.log'"

echo "started ${SESSION}: ${worker_index} workers on GPU5/6"
echo "incremental metrics: ${RUN_ROOT}"
echo "attach: tmux attach -t ${SESSION}"
