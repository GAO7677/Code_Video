#!/usr/bin/env bash
set -euo pipefail

# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_priority_case_head_ablation_tmux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-${SCRIPT_DIR}/config_priority_case_head_ablation_test5.sh}"
# shellcheck source=/dev/null
source "${CONFIG}"
WORKER="${SCRIPT_DIR}/run_priority_case_head_ablation_worker.sh"
COORDINATOR="${SCRIPT_DIR}/run_priority_case_head_ablation_coordinator.sh"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi
if [[ ! -f "${PRIORITY_INPUT_JSON}" ]]; then
  echo "missing priority input: ${PRIORITY_INPUT_JSON}" >&2
  exit 2
fi
if [[ ! -s "${PRIORITY_RUN_ROOT}/paused_worker_pids.txt" ]]; then
  echo "missing paused worker pid file" >&2
  exit 2
fi

mkdir -p "${PRIORITY_RUN_ROOT}/task_state" \
  "${PRIORITY_RUN_ROOT}/validations" "${PRIORITY_RUN_ROOT}/logs"
printf '%s\n' "${PRIORITY_INPUT_JSON}" > "${PRIORITY_RUN_ROOT}/priority_input.txt"
: > "${PRIORITY_RUN_ROOT}/queue.tsv"
index=0
head_start="${PRIORITY_HEADS%-*}"
head_end="${PRIORITY_HEADS#*-}"
for block in ${PRIORITY_BLOCKS}; do
  for head in $(seq "${head_start}" "${head_end}"); do
    for model in ${PRIORITY_MODELS}; do
      printf 'priority-%04d\t%s\t%s\t%s\n' \
        "${index}" "${model}" "${block}" "${head}" \
        >> "${PRIORITY_RUN_ROOT}/queue.tsv"
      index=$((index + 1))
    done
  done
done
if [[ "${index}" -ne "${EXPECTED_PRIORITY_TASKS}" ]]; then
  echo "expected ${EXPECTED_PRIORITY_TASKS} tasks, got ${index}" >&2
  exit 2
fi
printf '1\n' > "${PRIORITY_RUN_ROOT}/cursor"
: > "${PRIORITY_RUN_ROOT}/completed.tsv"
: > "${PRIORITY_RUN_ROOT}/failed.tsv"
rm -f "${PRIORITY_RUN_ROOT}/priority.complete" \
  "${PRIORITY_RUN_ROOT}/priority.failed"
cp "${CONFIG}" "${PRIORITY_RUN_ROOT}/config_snapshot.sh"

tmux new-session -d -s "${SESSION}" -n coordinator \
  "bash '${COORDINATOR}' '${CONFIG}'; exec bash"
for gpu in ${PRIORITY_GPUS}; do
  tmux new-window -t "${SESSION}" -n "gpu${gpu}" \
    "bash '${WORKER}' '${CONFIG}' '${gpu}' priority_g${gpu}; exec bash"
done
tmux select-window -t "${SESSION}:coordinator"
echo "session=${SESSION}"
echo "tasks=${EXPECTED_PRIORITY_TASKS}"
echo "priority_case=${PRIORITY_INPUT_JSON}"
