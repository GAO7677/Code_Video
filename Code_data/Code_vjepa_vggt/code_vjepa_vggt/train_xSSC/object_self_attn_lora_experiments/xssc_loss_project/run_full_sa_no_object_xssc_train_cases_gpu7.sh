#!/usr/bin/env bash
# Give this comparison exclusive use of GPU7, restoring its watcher on exit.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WATCH_SESSION="xssc_loss_watch_gpu7"
WATCH_METHODS="full_sa_no_object_xssc_loss_dinov3_movic_step50000,full_sa_object_slot_dedup_xssc50k_xssc_loss_dinov3_movic_step50000,t_head_pck32_s039_latest3350_top100_no_object_xssc_loss_dinov3_movic_step50000,wan22_openvid_lora_baseline"
WATCH_LOG="/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_three_run_watch/logs/xssc_loss_method_watch/watcher_gpu7.log"
watch_was_running=0

restore_watcher() {
  if (( watch_was_running == 0 )); then
    return
  fi
  if tmux has-session -t "${WATCH_SESSION}" 2>/dev/null; then
    return
  fi
  tmux new-session -d -s "${WATCH_SESSION}" \
    "cd '${PROJECT_DIR}' && env GPUS=7 METHOD='${WATCH_METHODS}' bash run_watch_full_sa_no_object_xssc_loss.sh 2>&1 | tee -a '${WATCH_LOG}'"
  echo "[$(date -u +%FT%TZ)] restored ${WATCH_SESSION}"
}
trap restore_watcher EXIT

if tmux has-session -t "${WATCH_SESSION}" 2>/dev/null; then
  watch_was_running=1
  tmux send-keys -t "${WATCH_SESSION}:0" C-c
  for _ in $(seq 1 20); do
    tmux has-session -t "${WATCH_SESSION}" 2>/dev/null || break
    sleep 2
  done
  if tmux has-session -t "${WATCH_SESSION}" 2>/dev/null; then
    echo "Watcher did not stop after Ctrl-C." >&2
    exit 2
  fi
fi

GPU_ID=7 bash "${PROJECT_DIR}/run_full_sa_no_object_xssc_train_cases_gpu1.sh"
