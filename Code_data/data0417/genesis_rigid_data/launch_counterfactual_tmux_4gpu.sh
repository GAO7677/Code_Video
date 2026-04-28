#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${SESSION_NAME:-physx_cf_all_4gpu}"
RUNNER="/home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/run_add_counterfactual_all_objects_sharded.sh"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases}"
PROCS_PER_GPU="${PROCS_PER_GPU:-15}"
SHARD_COUNT=4

if [[ ! -f "${RUNNER}" ]]; then
  echo "ERROR: runner not found: ${RUNNER}" >&2
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
  printf "bash -lc 'GPU_ID=%s SHARD_INDEX=%s SHARD_COUNT=%s PROCS_PER_GPU=%s OUTPUT_ROOT=%q GENERATE_CAPTIONS=0 %q; status=\$?; echo; echo RUN_FINISHED status=\$status; exec bash'" \
    "${gpu_id}" "${shard_index}" "${SHARD_COUNT}" "${PROCS_PER_GPU}" "${OUTPUT_ROOT}" "${RUNNER}"
}

tmux new-session -d -s "${SESSION_NAME}" -n gpu0 "$(cmd_for_gpu 0 0)"
tmux new-window -t "${SESSION_NAME}" -n gpu1 "$(cmd_for_gpu 1 1)"
tmux new-window -t "${SESSION_NAME}" -n gpu2 "$(cmd_for_gpu 2 2)"
tmux new-window -t "${SESSION_NAME}" -n gpu3 "$(cmd_for_gpu 3 3)"
tmux setw -t "${SESSION_NAME}" remain-on-exit on

echo "Created tmux session: ${SESSION_NAME}"
echo "Attach with: tmux attach -t ${SESSION_NAME}"
tmux list-windows -t "${SESSION_NAME}"
