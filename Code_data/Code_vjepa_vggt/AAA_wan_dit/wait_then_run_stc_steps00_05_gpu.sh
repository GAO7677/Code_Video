#!/usr/bin/env bash
set -euo pipefail

# GPU=0 bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/wait_then_run_stc_steps00_05_gpu.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:?set GPU}"
WAIT_SESSION="${WAIT_SESSION:-wan_stc_steps05_10_handoff}"
WAIT_WINDOW="gpu${GPU}"

while tmux list-windows -t "${WAIT_SESSION}" -F '#{window_name}' 2>/dev/null \
  | grep -Fxq "${WAIT_WINDOW}"; do
  echo "[stc-steps00-05-handoff] GPU${GPU} waiting for ${WAIT_SESSION}:${WAIT_WINDOW} at $(date -u +%FT%TZ)"
  sleep 15
done

echo "[stc-steps00-05-handoff] GPU${GPU} released; starting [0,5) tasks at $(date -u +%FT%TZ)"
GPU="${GPU}" NUM_WORKERS=7 \
  bash "${SCRIPT_DIR}/run_stc_steps00_05_multiseed_worker.sh"
