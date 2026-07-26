#!/usr/bin/env bash
set -euo pipefail

# Run in foreground:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/resume_physrvg_spatiotemporal_gpu4_and_analyze.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT=/data/gaoya/agent-data/outputs/wan_dit_block17_spatiotemporal_generalization
STATISTICS_ROOT="${OUTPUT_ROOT}/statistics"
LOG_ROOT="${OUTPUT_ROOT}/logs"
GPU=4
MAX_USED_MIB="${MAX_USED_MIB:-4000}"
POLL_SECONDS="${POLL_SECONDS:-30}"
ANALYSIS_PYTHON=/data/gaoya/miniconda3/envs/vjepa2/bin/python

mkdir -p "${LOG_ROOT}"

while true; do
  used_mib="$(
    nvidia-smi -i "${GPU}" \
      --query-gpu=memory.used \
      --format=csv,noheader,nounits \
      | tr -d ' '
  )"
  printf '[wait-gpu] gpu=%s used_mib=%s threshold_mib=%s\n' \
    "${GPU}" "${used_mib}" "${MAX_USED_MIB}"
  if ((used_mib <= MAX_USED_MIB)); then
    break
  fi
  sleep "${POLL_SECONDS}"
done

env \
  OUTPUT_ROOT="${OUTPUT_ROOT}" \
  RUN_MODE=physrvg \
  SKIP_ANALYSIS=1 \
  GPU_PHYRVG="${GPU}" \
  bash "${SCRIPT_DIR}/run_spatiotemporal_generalization_gpu34.sh"

while true; do
  wan_count="$(
    find "${STATISTICS_ROOT}/wan_lora" -name summary.json 2>/dev/null \
      | wc -l
  )"
  physrvg_count="$(
    find "${STATISTICS_ROOT}/physrvg" -name summary.json 2>/dev/null \
      | wc -l
  )"
  printf '[wait-results] wan_lora=%s/69 physrvg=%s/69\n' \
    "${wan_count}" "${physrvg_count}"
  if ((wan_count == 69 && physrvg_count == 69)); then
    break
  fi
  sleep "${POLL_SECONDS}"
done

PYTHONPATH="${SCRIPT_DIR}" \
"${ANALYSIS_PYTHON}" "${SCRIPT_DIR}/analyze_spatiotemporal_head_generalization.py" \
  --statistics-root "${STATISTICS_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/analysis"

echo "analysis=${OUTPUT_ROOT}/analysis/generalization_summary.json"
