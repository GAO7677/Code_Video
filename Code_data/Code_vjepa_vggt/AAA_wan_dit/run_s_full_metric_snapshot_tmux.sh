#!/usr/bin/env bash
set -euo pipefail

# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_s_full_metric_snapshot_tmux.sh SNAPSHOT_DIR

ROOT="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit"
WORKER="${ROOT}/run_head_role_incremental_metric_worker.sh"
INPUT_LIST="/data/gaoya/agent-data/outputs/wan_dit_fulltoken_head_roles_50seeds/input_lists/test5_unique20.txt"
RUN_ROOT="${1:?usage: run_s_full_metric_snapshot_tmux.sh SNAPSHOT_DIR}"
read -r -a GPUS <<< "${GPU_LIST:-0 1 2 3 5 6 7}"
CPU_WORKERS_PER_GPU="${CPU_WORKERS_PER_GPU:-3}"
GPU_WORKERS_PER_GPU="${GPU_WORKERS_PER_GPU:-1}"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-30000}"
SESSION="${SESSION:-wan_s_full_metrics_$(basename "${RUN_ROOT}")}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

cpu_workers=$((${#GPUS[@]} * CPU_WORKERS_PER_GPU))
gpu_workers=$((${#GPUS[@]} * GPU_WORKERS_PER_GPU))
expected_workers=$((cpu_workers + gpu_workers))
expected_tasks="$(
  awk 'END { print NR }' \
    "${RUN_ROOT}/queues/cpu.tsv" \
    "${RUN_ROOT}/queues/gpu.tsv"
)"

tmux new-session -d -s "${SESSION}" -n monitor \
  "while true; do complete=\$(wc -l < '${RUN_ROOT}/completed_tasks.tsv'); failed=\$(wc -l < '${RUN_ROOT}/failed_tasks.tsv'); workers=\$(find '${RUN_ROOT}/state' -maxdepth 1 -name '*.complete' -type f | wc -l); printf '[s-full-metrics] complete=%s/${expected_tasks} failed=%s workers=%s/${expected_workers}\\n' \"\${complete}\" \"\${failed}\" \"\${workers}\"; if [ \"\${workers}\" -eq '${expected_workers}' ]; then if [ \"\${failed}\" -eq 0 ] && [ \"\${complete}\" -eq '${expected_tasks}' ]; then touch '${RUN_ROOT}/run.complete'; else touch '${RUN_ROOT}/run.failed'; fi; break; fi; sleep 30; done; exec bash"

for gpu in "${GPUS[@]}"; do
  for index in $(seq 0 $((CPU_WORKERS_PER_GPU - 1))); do
    name="cpu_g${gpu}_${index}"
    tmux new-window -t "${SESSION}" -n "${name}" \
      "METRIC_WORKER_THREADS=2 bash '${WORKER}' '${gpu}' cpu '${name}' '${RUN_ROOT}' '${INPUT_LIST}' 0 0; exec bash"
  done
  for index in $(seq 0 $((GPU_WORKERS_PER_GPU - 1))); do
    name="gpu_g${gpu}_${index}"
    tmux new-window -t "${SESSION}" -n "${name}" \
      "METRIC_WORKER_THREADS=2 bash '${WORKER}' '${gpu}' gpu '${name}' '${RUN_ROOT}' '${INPUT_LIST}' '${GPU_MIN_FREE_MIB}' 20; exec bash"
  done
done

tmux select-window -t "${SESSION}:monitor"
echo "session=${SESSION}"
echo "run_root=${RUN_ROOT}"
echo "tasks=${expected_tasks}"
