#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
OUTPUT_BASE="/data/gaoya/agent-data/outputs/wan_dit_s_motion_analysis/physics_metric_snapshots"
INPUT_LIST="/data/gaoya/agent-data/outputs/wan_dit_fulltoken_head_roles_50seeds/input_lists/test5_unique20.txt"
WORKER="${ROOT}/run_head_role_incremental_metric_worker.sh"
GPUS=(0 1 2 3 5 6 7)
CPU_WORKERS_PER_GPU=3
HEAVY_MIN_FREE_MIB=30000

"${PYTHON}" "${ROOT}/build_s_current_physics_metric_snapshot.py"
RUN_ROOT="$(cat "${OUTPUT_BASE}/latest")"
SESSION="${SESSION:-wan_s_current_physics_metrics_$(basename "${RUN_ROOT}")}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

cpu_workers=$((${#GPUS[@]} * CPU_WORKERS_PER_GPU))
heavy_workers=${#GPUS[@]}
tmux new-session -d -s "${SESSION}" -n monitor \
  "while true; do cpu_cursor=\$(cat '${RUN_ROOT}/queues/cpu.cursor'); heavy_cursor=\$(cat '${RUN_ROOT}/queues/heavy.cursor'); complete=\$(wc -l < '${RUN_ROOT}/completed_tasks.tsv'); failed=\$(wc -l < '${RUN_ROOT}/failed_tasks.tsv'); workers=\$(find '${RUN_ROOT}/state' -maxdepth 1 -name '*.complete' -type f | wc -l); printf '[s-physics-metrics] cpu_claimed=%s heavy_claimed=%s complete=%s failed=%s workers=%s/$((cpu_workers + heavy_workers))\\n' \"\$((cpu_cursor - 1))\" \"\$((heavy_cursor - 1))\" \"\$complete\" \"\$failed\" \"\$workers\"; [ \"\$workers\" -eq '$((cpu_workers + heavy_workers))' ] && break; sleep 30; done; exec bash"

for gpu in "${GPUS[@]}"; do
  for index in $(seq 0 $((CPU_WORKERS_PER_GPU - 1))); do
    name="cpu_g${gpu}_${index}"
    tmux new-window -t "${SESSION}" -n "${name}" \
      "METRIC_WORKER_THREADS=2 bash '${WORKER}' '${gpu}' cpu '${name}' '${RUN_ROOT}' '${INPUT_LIST}' 0 0; exec bash"
  done
  name="heavy_g${gpu}"
  tmux new-window -t "${SESSION}" -n "${name}" \
    "METRIC_WORKER_THREADS=2 bash '${WORKER}' '${gpu}' heavy '${name}' '${RUN_ROOT}' '${INPUT_LIST}' '${HEAVY_MIN_FREE_MIB}' 30; exec bash"
done

tmux select-window -t "${SESSION}:monitor"
echo "session=${SESSION}"
echo "run_root=${RUN_ROOT}"
echo "cpu_workers=${cpu_workers}"
echo "heavy_workers=${heavy_workers}"
