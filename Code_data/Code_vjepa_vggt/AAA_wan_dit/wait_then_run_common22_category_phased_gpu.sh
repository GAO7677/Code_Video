#!/usr/bin/env bash
set -euo pipefail

# GPU=0 bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/wait_then_run_common22_category_phased_gpu.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:?set GPU}"
WAIT_SESSION="${WAIT_SESSION:-wan_s_score_phased_ablation_seed851}"
WAIT_WINDOW="gpu${GPU}"

while tmux list-windows -t "${WAIT_SESSION}" -F '#{window_name}' 2>/dev/null \
  | grep -Fxq "${WAIT_WINDOW}"; do
  echo "[category-gpu-handoff] GPU${GPU} waiting for ${WAIT_SESSION}:${WAIT_WINDOW} at $(date -u +%FT%TZ)"
  sleep 15
done

echo "[category-gpu-handoff] GPU${GPU} released; starting category tasks at $(date -u +%FT%TZ)"
GPU="${GPU}" NUM_WORKERS=7 SEED=851 \
  bash "${SCRIPT_DIR}/run_common22_category_phased_worker.sh"
