#!/usr/bin/env bash
# Run:
#   bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_stc_bench_all_gpus_parallel_tmux.sh

set -euo pipefail

ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
BATCH_ROOT=/data/gaoya/agent-data/outputs/wan_dit_stc_bench
RUN_ROOT="${BATCH_ROOT}/run_20260728_parallel7"
REPORT_ROOT=/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/gallery/multiseed/benchmark-metrics
SESSION="${SESSION:-wan_stc_bench_gpu0123456_parallel_20260728}"
WORKER="${ROOT}/run_stc_bench_parallel_shard_worker.sh"
NUM_SHARDS=7
MAX_USED_MIB=2048

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/status" "${RUN_ROOT}/task_summaries"

tmux new-session -d -s "${SESSION}" -n coordinator \
  "while true; do workers=\$(find '${RUN_ROOT}/status' -maxdepth 1 -name '*.worker_complete' -type f | wc -l); complete=\$(find '${RUN_ROOT}/status' -maxdepth 1 -name '*.shard*.complete' -type f | wc -l); failed=\$(find '${RUN_ROOT}/status' -maxdepth 1 -name '*.shard*.failed' -type f | wc -l); printf '[parallel-coordinator] workers=%s/7 shards=%s/63 failed=%s\\n' \"\$workers\" \"\$complete\" \"\$failed\"; [ \"\$workers\" -eq 7 ] && break; sleep 30; done; '${PYTHON}' '${ROOT}/summarize_stc_bench_metrics.py' --batch-root '${BATCH_ROOT}'; '${PYTHON}' '${ROOT}/render_stc_bench_metric_report.py' --batch-root '${BATCH_ROOT}' --output-dir '${REPORT_ROOT}'; printf '[parallel-coordinator] final summary and report complete\\n'; exec bash"

for gpu in 0 1 2 3 4 5 6; do
  tmux new-window -t "${SESSION}" -n "gpu${gpu}" \
    "bash '${WORKER}' '${gpu}' '${gpu}' '${NUM_SHARDS}' '${BATCH_ROOT}' '${RUN_ROOT}' '${MAX_USED_MIB}'"
done

tmux select-window -t "${SESSION}:coordinator"
echo "tmux session: ${SESSION}"
echo "run root: ${RUN_ROOT}"
echo "workers: 7 GPU shards"
echo "remaining metrics: 9"
