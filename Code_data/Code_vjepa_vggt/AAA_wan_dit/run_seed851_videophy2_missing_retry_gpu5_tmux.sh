#!/usr/bin/env bash
# Run:
#   bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_seed851_videophy2_missing_retry_gpu5_tmux.sh

set -euo pipefail

ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit
BATCH_ROOT=/data/gaoya/agent-data/outputs/wan_dit_common22_test5_st_phased_seed851_bench
RUN_ROOT="${BATCH_ROOT}/run_20260728_videophy2_long_prompt_retry"
SESSION="${SESSION:-wan_seed851_videophy2_retry_gpu5_20260728}"
WORKER="${ROOT}/run_seed851_memory_guarded_metric_worker.sh"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi
if [[ -e "${RUN_ROOT}" ]]; then
  echo "run root already exists: ${RUN_ROOT}" >&2
  exit 1
fi

mkdir -p \
  "${RUN_ROOT}/queues" "${RUN_ROOT}/logs" "${RUN_ROOT}/state" \
  "${RUN_ROOT}/task_summaries"
: > "${RUN_ROOT}/completed_tasks.tsv"
: > "${RUN_ROOT}/failed_tasks.tsv"
: > "${RUN_ROOT}/queues/videophy.lock"
printf '1\n' > "${RUN_ROOT}/queues/videophy.cursor"
printf 'videophy-0000\tvideophy2\t0\t2\nvideophy-0001\tvideophy2\t1\t2\n' \
  > "${RUN_ROOT}/queues/videophy.tsv"

tmux new-session -d -s "${SESSION}" -n vp_g5_0 \
  "METRIC_WORKER_THREADS=2 bash '${WORKER}' videophy 5 vp_g5_0 '${RUN_ROOT}' '${BATCH_ROOT}' 46000"
tmux new-window -t "${SESSION}" -n vp_g5_1 \
  "METRIC_WORKER_THREADS=2 bash '${WORKER}' videophy 5 vp_g5_1 '${RUN_ROOT}' '${BATCH_ROOT}' 46000"

echo "tmux session: ${SESSION}"
echo "run root: ${RUN_ROOT}"
