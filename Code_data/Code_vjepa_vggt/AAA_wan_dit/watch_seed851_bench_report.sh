#!/usr/bin/env bash
# Run:
#   bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/watch_seed851_bench_report.sh

set -euo pipefail

ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
BATCH_ROOT=/data/gaoya/agent-data/outputs/wan_dit_common22_test5_st_phased_seed851_bench
RUN_ROOT="${BATCH_ROOT}/run_20260728_dynamic"
BASELINE_ROOT=/data/gaoya/agent-data/outputs/wan_dit_common22_test5_seed851_baseline_bench
BASELINE_RUN="${BASELINE_ROOT}/run_20260728_dynamic"
OUTPUT_DIR=/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/gallery/multiseed/seed851/benchmark-metrics

while true; do
  "${PYTHON}" "${ROOT}/summarize_stc_bench_metrics.py" \
    --batch-root "${BATCH_ROOT}"
  "${PYTHON}" "${ROOT}/summarize_stc_bench_metrics.py" \
    --batch-root "${BASELINE_ROOT}"
  "${PYTHON}" "${ROOT}/render_stc_bench_metric_report.py" \
    --batch-root "${BATCH_ROOT}" \
    --baseline-batch-root "${BASELINE_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    --title "Seed 851 分阶段消融指标" \
    --companion-url "../../benchmark-metrics/" \
    --companion-label "查看第一批 503-case 指标"
  if [[ -f "${RUN_ROOT}/state/all_complete" ]] \
    && [[ -f "${BASELINE_RUN}/state/all_complete" ]]; then
    break
  fi
  sleep 60
done

echo "[seed851-report-watcher] complete"
exec bash
