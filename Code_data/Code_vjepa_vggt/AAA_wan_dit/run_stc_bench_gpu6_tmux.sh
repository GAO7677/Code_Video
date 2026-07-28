#!/usr/bin/env bash
# Run:
#   bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_stc_bench_gpu6_tmux.sh

set -euo pipefail

ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
BATCH_ROOT=/data/gaoya/agent-data/outputs/wan_dit_stc_bench
RUN_ROOT="${BATCH_ROOT}/run_20260728"
REPORT_ROOT=/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/gallery/multiseed/benchmark-metrics
SESSION="${SESSION:-wan_stc_bench_gpu6_20260728}"
WORKER="${ROOT}/run_stc_bench_worker.sh"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

"${PYTHON}" "${ROOT}/build_stc_bench_batch.py" \
  --output-root "${BATCH_ROOT}"
mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/status"

tmux new-session -d -s "${SESSION}" -n coordinator \
  "while true; do done_count=\$(find '${RUN_ROOT}/status' -maxdepth 1 -name '*.worker_complete' -type f | wc -l); failed_count=\$(find '${RUN_ROOT}/status' -maxdepth 1 -name '*.failed' -type f | wc -l); metric_count=\$(find '${RUN_ROOT}/status' -maxdepth 1 -name '*.complete' -type f | wc -l); printf '[coordinator] workers=%s/5 metrics=%s/14 failed=%s\\n' \"\$done_count\" \"\$metric_count\" \"\$failed_count\"; [ \"\$done_count\" -eq 5 ] && break; sleep 30; done; '${PYTHON}' '${ROOT}/summarize_stc_bench_metrics.py' --batch-root '${BATCH_ROOT}'; '${PYTHON}' '${ROOT}/render_stc_bench_metric_report.py' --batch-root '${BATCH_ROOT}' --output-dir '${REPORT_ROOT}'; exec bash"

tmux new-window -t "${SESSION}" -n pi_ctx \
  "bash '${WORKER}' pi_ctx 6 '${BATCH_ROOT}' '${RUN_ROOT}' physics_iq_with_context"
tmux new-window -t "${SESSION}" -n pi_noctx \
  "bash '${WORKER}' pi_noctx 6 '${BATCH_ROOT}' '${RUN_ROOT}' physics_iq_without_context"
tmux new-window -t "${SESSION}" -n pmf_ctx \
  "bash '${WORKER}' pmf_ctx 6 '${BATCH_ROOT}' '${RUN_ROOT}' pmf_with_context"
tmux new-window -t "${SESSION}" -n pmf_noctx \
  "bash '${WORKER}' pmf_noctx 6 '${BATCH_ROOT}' '${RUN_ROOT}' pmf_without_context"
tmux new-window -t "${SESSION}" -n gpu_metrics \
  "bash '${WORKER}' gpu_metrics 6 '${BATCH_ROOT}' '${RUN_ROOT}' \
    wmreward \
    vbench_subject_consistency \
    vbench_background_consistency \
    vbench_temporal_flickering \
    vbench_motion_smoothness \
    vbench_dynamic_degree \
    vbench_aesthetic_quality \
    vbench_imaging_quality \
    videophy2 \
    cosmos_reason1"

tmux select-window -t "${SESSION}:coordinator"
echo "tmux session: ${SESSION}"
echo "batch root: ${BATCH_ROOT}"
echo "run root: ${RUN_ROOT}"
echo "entries: 503"
