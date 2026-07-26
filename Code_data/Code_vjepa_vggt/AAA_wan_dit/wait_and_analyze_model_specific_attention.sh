#!/usr/bin/env bash
set -euo pipefail

# SESSION=wan_model_specific_attention_analysis bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/wait_and_analyze_model_specific_attention.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAPTURE_SESSION="${CAPTURE_SESSION:-wan_model_specific_allblock_test5}"
ROOT=/data/gaoya/agent-data/outputs/wan_dit_model_specific_attention/test5_allblocks
QUERY_ROOT=/data/gaoya/agent-data/outputs/wan_dit_model_specific_query_maps/test5
OUTPUT_DIR="${ROOT}/cross_case_analysis"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python

while tmux has-session -t "${CAPTURE_SESSION}" 2>/dev/null; do
  feature_count="$(
    find "${ROOT}" -name '*moving_query_features.npz' 2>/dev/null | wc -l
  )"
  echo "[attention-analysis-watcher] waiting: features=${feature_count}/7200"
  sleep 60
done

feature_count="$(
  find "${ROOT}"/block*/matrices -name '*moving_query_features.npz' | wc -l
)"
summary_count="$(
  find "${ROOT}"/block*/matrices -name summary.json | wc -l
)"
video_count="$(
  find "${ROOT}"/generated/{wan_lora,xssc,physrvg} -name '*.mp4' | wc -l
)"

echo "[attention-analysis-watcher] final counts: features=${feature_count}, summaries=${summary_count}, videos=${video_count}"
test "${feature_count}" -eq 7200
test "${summary_count}" -eq 1800
test "${video_count}" -eq 60

PYTHONPATH="${SCRIPT_DIR}" "${PYTHON}" \
  "${SCRIPT_DIR}/analyze_cross_case_head_stability.py" \
  --root "${ROOT}" \
  --query-map-root "${QUERY_ROOT}" \
  --output-dir "${OUTPUT_DIR}"

echo "[attention-analysis-watcher] complete: ${OUTPUT_DIR}"
