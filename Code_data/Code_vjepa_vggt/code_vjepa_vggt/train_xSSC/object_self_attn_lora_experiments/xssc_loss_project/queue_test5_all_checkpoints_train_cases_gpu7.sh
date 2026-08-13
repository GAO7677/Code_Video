#!/usr/bin/env bash
# Wait for the step-500 gallery job, pause GPU7 watcher, then run all checkpoints.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
CURRENT_SESSION="test5_step500_all_methods_train_cases_gpu7"
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

while tmux has-session -t "${CURRENT_SESSION}" 2>/dev/null; do
  echo "[$(date -u +%FT%TZ)] waiting for ${CURRENT_SESSION}"
  sleep 20
done

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

while true; do
  used="$(nvidia-smi -i 7 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  if [[ "${used}" -le 2000 ]]; then
    break
  fi
  echo "[$(date -u +%FT%TZ)] GPU7 used=${used} MiB; waiting"
  sleep 20
done

"${PYTHON}" "${PROJECT_DIR}/run_test5_all_checkpoints_train_cases.py" --gpu 7
