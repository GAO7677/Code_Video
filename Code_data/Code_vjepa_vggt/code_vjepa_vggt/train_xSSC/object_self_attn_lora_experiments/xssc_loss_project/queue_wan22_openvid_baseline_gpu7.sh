#!/usr/bin/env bash
# Let the active PCKhead PhysicIQ attempt finish, prioritize the pure OpenVid
# baseline until both suites and all metrics finish, then restore the full watcher.
set -u -o pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WATCH_SESSION="xssc_loss_watch_gpu7"
BASELINE_METHOD="wan22_openvid_lora_baseline"
PCK_METHOD="t_head_pck32_s039_latest3350_top100_no_object_xssc_loss_dinov3_movic_step50000"
ALL_METHODS="full_sa_no_object_xssc_loss_dinov3_movic_step50000,full_sa_object_slot_dedup_xssc50k_xssc_loss_dinov3_movic_step50000,${PCK_METHOD},${BASELINE_METHOD}"
STATE_ROOT="/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_three_run_watch/state"
LOG_ROOT="/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_three_run_watch/logs/xssc_loss_method_watch"
QUEUE_LOG="${LOG_ROOT}/queue_wan22_openvid_baseline_gpu7.log"
EXPECTED_METRICS=14

mkdir -p "${LOG_ROOT}"
exec >>"${QUEUE_LOG}" 2>&1

timestamp() {
  date -u +%FT%TZ
}

stop_watch_session() {
  if ! tmux has-session -t "${WATCH_SESSION}" 2>/dev/null; then
    return
  fi
  tmux send-keys -t "${WATCH_SESSION}:0" C-c
  for _ in $(seq 1 20); do
    if ! tmux has-session -t "${WATCH_SESSION}" 2>/dev/null; then
      return
    fi
    sleep 3
  done
  echo "[$(timestamp)] watcher session did not exit after Ctrl-C; closing that exact tmux session"
  tmux kill-session -t "${WATCH_SESSION}"
}

start_watch_session() {
  local methods="$1"
  local label="$2"
  local watch_log="${LOG_ROOT}/${label}.log"
  tmux new-session -d -s "${WATCH_SESSION}" \
    "cd '${PROJECT_DIR}' && env GPUS=7 METHOD='${methods}' bash run_watch_full_sa_no_object_xssc_loss.sh 2>&1 | tee -a '${watch_log}'"
  echo "[$(timestamp)] started ${WATCH_SESSION} methods=${methods}"
}

echo "[$(timestamp)] waiting for the active PCKhead PhysicIQ process to finish naturally"
while pgrep -f "[x]ssc_lora_physiciq_parallel_infer.py.*--methods ${PCK_METHOD}" >/dev/null; do
  sleep 15
done

echo "[$(timestamp)] current PCKhead inference attempt ended; switching watcher to baseline priority"
stop_watch_session
start_watch_session "${BASELINE_METHOD}" "watcher_gpu7_openvid_baseline"

TEST_MANIFEST="${STATE_ROOT}/checkpoints/${BASELINE_METHOD}/step-000000.json"
PHYS_MANIFEST="${STATE_ROOT}/physiciq/inference/${BASELINE_METHOD}/step-000000.json"
TEST_METRICS="${STATE_ROOT}/metrics/${BASELINE_METHOD}/step-000000"
PHYS_METRICS="${STATE_ROOT}/physiciq/metrics/${BASELINE_METHOD}/step-000000"

while true; do
  test_count=$(find "${TEST_METRICS}" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l)
  phys_count=$(find "${PHYS_METRICS}" -maxdepth 1 -type f -name '*.json' 2>/dev/null | wc -l)
  if [[ -f "${TEST_MANIFEST}" && -f "${PHYS_MANIFEST}" \
        && "${test_count}" -ge "${EXPECTED_METRICS}" \
        && "${phys_count}" -ge "${EXPECTED_METRICS}" ]]; then
    break
  fi
  echo "[$(timestamp)] baseline progress test_manifest=$([[ -f "${TEST_MANIFEST}" ]] && echo 1 || echo 0) phys_manifest=$([[ -f "${PHYS_MANIFEST}" ]] && echo 1 || echo 0) test_metrics=${test_count}/${EXPECTED_METRICS} phys_metrics=${phys_count}/${EXPECTED_METRICS}"
  sleep 60
done

echo "[$(timestamp)] baseline suites and metrics complete; restoring the full watcher"
stop_watch_session
start_watch_session "${ALL_METHODS}" "watcher_gpu7"
echo "[$(timestamp)] queue completed"
