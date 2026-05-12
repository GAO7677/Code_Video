#!/usr/bin/env bash
# 用途：用 tmux 在 4 张 GPU 上启动零重力 counterfactual 修复任务。
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-physx_cf_fix_gravity_4gpu}"
RUNNER="/home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/runs/run_fix_zero_gravity_counterfactual_sharded.sh"
TASK_SCRIPT="/home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/repair/list_zero_gravity_counterfactual_tasks.py"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases}"
FIX_ROOT="${FIX_ROOT:-/home/gaoya/Code_Video/Code_data/Code_train/train_0419/zero_gravity_counterfactual_fix}"
PROCS_PER_GPU="${PROCS_PER_GPU:-15}"
SHARD_COUNT=4
PY3_BIN="${PY3_BIN:-python3}"

if [[ ! -f "${RUNNER}" ]]; then
  echo "ERROR: runner not found: ${RUNNER}" >&2
  exit 1
fi
if [[ ! -f "${TASK_SCRIPT}" ]]; then
  echo "ERROR: task script not found: ${TASK_SCRIPT}" >&2
  exit 1
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "ERROR: tmux session already exists: ${SESSION_NAME}" >&2
  echo "Use: tmux attach -t ${SESSION_NAME} or kill it first." >&2
  exit 1
fi

cmd_for_gpu() {
  local gpu_id="$1"
  local shard_index="$2"
  printf "bash -lc 'GPU_ID=%s SHARD_INDEX=%s SHARD_COUNT=%s PROCS_PER_GPU=%s REBUILD_TASKS=0 OUTPUT_ROOT=%q FIX_ROOT=%q %q; status=\$?; echo; echo RUN_FINISHED status=\$status; exec bash'" \
    "${gpu_id}" "${shard_index}" "${SHARD_COUNT}" "${PROCS_PER_GPU}" "${OUTPUT_ROOT}" "${FIX_ROOT}" "${RUNNER}"
}

mkdir -p "${FIX_ROOT}"
"${PY3_BIN}" "${TASK_SCRIPT}" \
  --output_root "${OUTPUT_ROOT}" \
  --tasks_tsv "${FIX_ROOT}/zero_gravity_tasks.tsv" \
  --summary_json "${FIX_ROOT}/zero_gravity_tasks_summary.json"

tmux new-session -d -s "${SESSION_NAME}" -n gpu0 "$(cmd_for_gpu 0 0)"
tmux new-window -t "${SESSION_NAME}" -n gpu1 "$(cmd_for_gpu 1 1)"
tmux new-window -t "${SESSION_NAME}" -n gpu2 "$(cmd_for_gpu 2 2)"
tmux new-window -t "${SESSION_NAME}" -n gpu3 "$(cmd_for_gpu 3 3)"
tmux setw -t "${SESSION_NAME}" remain-on-exit on

echo "Created tmux session: ${SESSION_NAME}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
tmux list-windows -t "${SESSION_NAME}"
