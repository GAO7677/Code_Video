#!/usr/bin/env bash
set -euo pipefail

BASE=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box
WORKER="${BASE}/run_bench_physiq_parallel_worker.sh"
VERIFY="${BASE}/verify_bench_physiq_metrics.py"
SUMMARY="${BASE}/summarize_benchmark_txt_metrics.py"
PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
BASELINE_LIST="${BASELINE_LIST:-${BASE}/AAAevalphysiq.txt}"
SESSION="${SESSION:-bench_physiq_gpu12_multi_20260715}"
RUN_ROOT="${RUN_ROOT:-/data/gaoya/agent-data/outputs/bench_physiq_gpu12_multi_20260715}"
SUMMARY_CSV="${BASE}/AAAresults/AAAevalphysiq_metric_summary.csv"
EXPECTED_WORKERS=8

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi
for gpu in 1 2; do
  used="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  if [[ -z "${used}" || "${used}" -ge 2048 ]]; then
    echo "GPU${gpu} is busy or unavailable: memory.used=${used:-unknown} MiB" >&2
    exit 1
  fi
done

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/state"
rm -f "${RUN_ROOT}/state"/*.complete "${RUN_ROOT}/state"/*.failed "${RUN_ROOT}/verification.json"

launch_worker() {
  local gpu="$1" name="$2" metrics="$3"
  tmux new-window -t "${SESSION}" -n "${name}" \
    "bash '${WORKER}' '${gpu}' '${name}' '${metrics}' '${RUN_ROOT}' '${BASELINE_LIST}'"
}

tmux new-session -d -s "${SESSION}" -n bootstrap "sleep 2"
launch_worker 1 g1_wmreward "wmreward"
launch_worker 1 g1_vbench_consistency "vbench_subject_consistency,vbench_background_consistency"
launch_worker 1 g1_vbench_temporal "vbench_temporal_flickering,vbench_motion_smoothness"
launch_worker 1 g1_physicsiq "physics_iq_with_context,physics_iq_without_context"
launch_worker 2 g2_videophy2 "videophy2"
launch_worker 2 g2_cosmos "cosmos_reason1"
launch_worker 2 g2_vbench_quality "vbench_dynamic_degree,vbench_aesthetic_quality,vbench_imaging_quality"
launch_worker 2 g2_pmf "pmf_with_context,pmf_without_context"

tmux new-window -t "${SESSION}" -n coordinator \
  "while true; do complete=\$(find '${RUN_ROOT}/state' -maxdepth 1 -name '*.complete' -type f | wc -l); failed=\$(find '${RUN_ROOT}/state' -maxdepth 1 -name '*.failed' -type f | wc -l); printf '[coordinator] complete=%s/${EXPECTED_WORKERS} failed=%s\\n' \"\$complete\" \"\$failed\"; if [ \"\$failed\" -gt 0 ]; then exit 1; fi; if [ \"\$complete\" -eq '${EXPECTED_WORKERS}' ]; then break; fi; sleep 30; done; '${PYTHON_BIN}' '${SUMMARY}' --input-txt '${BASELINE_LIST}' --output-csv '${SUMMARY_CSV}' && '${PYTHON_BIN}' '${VERIFY}' --baseline-list '${BASELINE_LIST}' --output '${RUN_ROOT}/verification.json'"
tmux kill-window -t "${SESSION}:bootstrap" 2>/dev/null || true
tmux select-window -t "${SESSION}:coordinator"

echo "tmux session: ${SESSION}"
echo "run root: ${RUN_ROOT}"
tmux list-windows -t "${SESSION}" -F '#I:#W pane_pid=#{pane_pid} active=#{window_active}'
