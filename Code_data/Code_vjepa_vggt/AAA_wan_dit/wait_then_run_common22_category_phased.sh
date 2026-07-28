#!/usr/bin/env bash
set -euo pipefail

# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/wait_then_run_common22_category_phased.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WAIT_SESSION="${WAIT_SESSION:-wan_s_score_phased_ablation_seed851}"

while tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; do
  echo "[category-phased-waiter] waiting for ${WAIT_SESSION} at $(date -u +%FT%TZ)"
  sleep 30
done

echo "[category-phased-waiter] ${WAIT_SESSION} finished; starting category tasks"
bash "${SCRIPT_DIR}/run_common22_category_phased_tmux.sh"
