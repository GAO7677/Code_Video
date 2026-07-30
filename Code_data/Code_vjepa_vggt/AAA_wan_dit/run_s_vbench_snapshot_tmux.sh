#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-wan_s_vbench_incremental}"
ROOT="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
WORKER="${ROOT}/run_bench_v2v_wan_queue_worker.sh"
SUMMARY="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/summarize_benchmark_txt_metrics.py"
ALLOWLIST="/data/gaoya/agent-data/outputs/wan_dit_fulltoken_head_roles_50seeds/input_lists/test5_unique20.txt"
LATEST="/data/gaoya/agent-data/outputs/wan_dit_s_motion_analysis/vbench_snapshots/latest"
read -r -a GPUS <<< "${GPU_LIST:-0 1 2 3 5 6 7}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"

SNAPSHOT_DIR="${1:-$(cat "${LATEST}")}"
if [[ ! -s "${SNAPSHOT_DIR}/queues/gpu_common.tsv" ]]; then
  echo "missing or empty VBench queue: ${SNAPSHOT_DIR}" >&2
  exit 1
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

EXPECTED_TASKS="$(wc -l < "${SNAPSHOT_DIR}/queues/gpu_common.tsv")"
EXPECTED_WORKERS=$(( ${#GPUS[@]} * WORKERS_PER_GPU ))
rm -f "${SNAPSHOT_DIR}/run.complete" "${SNAPSHOT_DIR}/run.failed"
tmux new-session -d -s "${SESSION}" -n coordinator \
  "while true; do workers=\$(find '${SNAPSHOT_DIR}/state' -maxdepth 1 -name '*.complete' -type f | wc -l); cursor=\$(cat '${SNAPSHOT_DIR}/queues/gpu_common.cursor'); completed=\$(wc -l < '${SNAPSHOT_DIR}/completed_tasks.tsv'); failed=\$(wc -l < '${SNAPSHOT_DIR}/failed_tasks.tsv'); printf '[s-vbench] workers=%s/${EXPECTED_WORKERS} claimed=%s/${EXPECTED_TASKS} completed=%s failed=%s\\n' \"\${workers}\" \"\$((cursor - 1))\" \"\${completed}\" \"\${failed}\"; [ \"\${workers}\" -eq '${EXPECTED_WORKERS}' ] && break; sleep 30; done; '${PYTHON}' '${SUMMARY}' --input-txt '${SNAPSHOT_DIR}/leaf_folders.txt' --output-csv '${SNAPSHOT_DIR}/metric_summary.csv' --input-json-allowlist '${ALLOWLIST}'; failed=\$(wc -l < '${SNAPSHOT_DIR}/failed_tasks.tsv'); if [ \"\${failed}\" -eq 0 ]; then touch '${SNAPSHOT_DIR}/run.complete'; else touch '${SNAPSHOT_DIR}/run.failed'; fi; exec bash"

for gpu in "${GPUS[@]}"; do
  for worker_index in $(seq 0 $((WORKERS_PER_GPU - 1))); do
    name="g${gpu}_${worker_index}"
    tmux new-window -t "${SESSION}" -n "${name}" \
      "bash '${WORKER}' '${gpu}' gpu_common '${name}' '${SNAPSHOT_DIR}' '${ALLOWLIST}'; exec bash"
  done
done

tmux select-window -t "${SESSION}:coordinator"
echo "session=${SESSION}"
echo "snapshot=${SNAPSHOT_DIR}"
echo "gpus=${GPUS[*]}"
echo "workers=${EXPECTED_WORKERS}"
echo "tasks=${EXPECTED_TASKS}"
