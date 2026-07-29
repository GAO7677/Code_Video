#!/usr/bin/env bash
set -euo pipefail

# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_head_role_incremental_metrics_tmux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
CONFIG="${SCRIPT_DIR}/head_role_dose_control_pilot.json"
ROOT=/data/gaoya/agent-data/outputs/wan_dit_head_role_dose_control/pilot
RUN_ROOT="${ROOT}/incremental_metrics_live"
INPUT_LIST=/data/gaoya/agent-data/outputs/wan_dit_fulltoken_head_roles_50seeds/input_lists/test5_unique20.txt
SESSION=wan_head_role_dose_control
WORKER="${SCRIPT_DIR}/run_head_role_incremental_metric_worker.sh"
CPU_GPUS=(0 1 2 3 5 6 7 0)
GPU_COMMON_GPUS=(1 2 3 5 7)
EXPECTED_WORKERS=$((${#CPU_GPUS[@]} + ${#GPU_COMMON_GPUS[@]}))

"${PYTHON}" "${SCRIPT_DIR}/build_head_role_incremental_metric_snapshot.py" \
  --config "${CONFIG}" \
  --workers "${EXPECTED_WORKERS}"

for index in "${!CPU_GPUS[@]}"; do
  gpu="${CPU_GPUS[$index]}"
  name="inc_cpu_${index}"
  command=(bash "${WORKER}" "${gpu}" cpu "${name}" "${RUN_ROOT}" "${INPUT_LIST}" 0 0)
  printf -v shell_command '%q ' "${command[@]}"
  tmux new-window -d -t "${SESSION}" -n "${name}" "${shell_command}; exec bash"
done

for gpu in "${GPU_COMMON_GPUS[@]}"; do
  name="inc_gpu_g${gpu}"
  command=(bash "${WORKER}" "${gpu}" gpu_common "${name}" "${RUN_ROOT}" "${INPUT_LIST}" 14000 75)
  printf -v shell_command '%q ' "${command[@]}"
  tmux new-window -d -t "${SESSION}" -n "${name}" "${shell_command}; exec bash"
done

touch "${RUN_ROOT}/started"
echo "[incremental-metrics] workers=${EXPECTED_WORKERS} root=${RUN_ROOT}"
