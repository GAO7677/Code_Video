#!/usr/bin/env bash
set -euo pipefail

# Run:
# GPU_WAN=0 GPU_XSSC=1 GPU_PHYRVG=2 \
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_multiblock_ball_query_attention_case001460.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SINGLE_BLOCK_SCRIPT="${SCRIPT_DIR}/run_ball_query_attention_case001460.sh"
ROOT="${OUTPUT_ROOT:-/data/gaoya/agent-data/outputs/wan_dit_ball_query_attention/case001460_frame08_multiblock}"
BLOCK_IDS_TEXT="${BLOCK_IDS:-0 5 11 19 29}"
GPU_WAN="${GPU_WAN:-0}"
GPU_XSSC="${GPU_XSSC:-1}"
GPU_PHYRVG="${GPU_PHYRVG:-2}"

read -r -a BLOCK_IDS_ARRAY <<< "${BLOCK_IDS_TEXT}"
mkdir -p "${ROOT}"

for block_id in "${BLOCK_IDS_ARRAY[@]}"; do
  if (( block_id < 0 || block_id > 29 )); then
    echo "Invalid block id: ${block_id}" >&2
    exit 2
  fi
  block_name="$(printf 'block%02d' "${block_id}")"
  echo "[multiblock-ball-query] starting ${block_name}"
  env \
    ATTENTION_BLOCK="${block_id}" \
    OUTPUT_ROOT="${ROOT}/${block_name}" \
    GPU_WAN="${GPU_WAN}" \
    GPU_XSSC="${GPU_XSSC}" \
    GPU_PHYRVG="${GPU_PHYRVG}" \
    bash "${SINGLE_BLOCK_SCRIPT}"
  echo "[multiblock-ball-query] completed ${block_name}"
done

echo "multiblock_root=${ROOT}"
