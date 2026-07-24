#!/usr/bin/env bash
set -euo pipefail

# Run:
#   bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_bench_v2v_wan_gpu45_tmux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${SCRIPT_DIR}/run_bench_v2v_wan_parallel_worker.sh"
VERIFY=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/verify_bench_physiq_metrics.py
SUMMARY=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/summarize_benchmark_txt_metrics.py
PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python

BASELINE_LIST="${BASELINE_LIST:-/data/gaoya/AAA_test_video/0623/test/v2v_wan/leaf_folders.txt}"
INPUT_ALLOWLIST="${INPUT_ALLOWLIST:-/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt}"
SESSION="${SESSION:-bench_v2v_wan_xssc_gpu45_multi_20260724}"
RUN_ROOT="${RUN_ROOT:-/data/gaoya/AAA_test_video/0623/test/v2v_wan/_bench_runs/${SESSION}}"
SUMMARY_CSV="${RUN_ROOT}/xssc_metric_summary.csv"
EXPECTED_WORKERS=8

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi
if [[ ! -s "${BASELINE_LIST}" ]]; then
  echo "Missing or empty baseline list: ${BASELINE_LIST}" >&2
  exit 2
fi
if [[ ! -s "${INPUT_ALLOWLIST}" ]]; then
  echo "Missing or empty input allowlist: ${INPUT_ALLOWLIST}" >&2
  exit 2
fi

gpu4_used="$(nvidia-smi -i 4 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
if [[ -z "${gpu4_used}" || "${gpu4_used}" -ge 2048 ]]; then
  echo "GPU4 is busy or unavailable: memory.used=${gpu4_used:-unknown} MiB" >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/state"

launch_worker() {
  local gpu="$1" name="$2" metrics="$3" wait_for_gpu="$4"
  tmux new-window -t "${SESSION}" -n "${name}" \
    "bash '${WORKER}' '${gpu}' '${name}' '${metrics}' '${RUN_ROOT}' '${BASELINE_LIST}' '${INPUT_ALLOWLIST}' '${wait_for_gpu}'"
}

tmux new-session -d -s "${SESSION}" -n bootstrap "sleep 5"

launch_worker 4 g4_wmreward "wmreward" 0
launch_worker 4 g4_vbench_consistency "vbench_subject_consistency,vbench_background_consistency" 0
launch_worker 4 g4_vbench_temporal "vbench_temporal_flickering,vbench_motion_smoothness" 0
launch_worker 4 g4_physicsiq "physics_iq_with_context,physics_iq_without_context" 0

launch_worker 5 g5_videophy2 "videophy2" 1
launch_worker 5 g5_cosmos "cosmos_reason1" 1
launch_worker 5 g5_vbench_quality "vbench_dynamic_degree,vbench_aesthetic_quality,vbench_imaging_quality" 1
launch_worker 5 g5_pmf "pmf_with_context,pmf_without_context" 1

tmux new-window -t "${SESSION}" -n coordinator \
  "while true; do complete=\$(find '${RUN_ROOT}/state' -maxdepth 1 -name '*.complete' -type f | wc -l); failed=\$(find '${RUN_ROOT}/state' -maxdepth 1 -name '*.failed' -type f | wc -l); printf '[coordinator] complete=%s/${EXPECTED_WORKERS} failed=%s\\n' \"\$complete\" \"\$failed\"; if [ \"\$failed\" -gt 0 ]; then exit 1; fi; if [ \"\$complete\" -eq '${EXPECTED_WORKERS}' ]; then break; fi; sleep 30; done; '${PYTHON_BIN}' '${SUMMARY}' --input-txt '${BASELINE_LIST}' --output-csv '${SUMMARY_CSV}' --input-json-allowlist '${INPUT_ALLOWLIST}' && '${PYTHON_BIN}' '${VERIFY}' --baseline-list '${BASELINE_LIST}' --output '${RUN_ROOT}/verification.json' --input-json-allowlist '${INPUT_ALLOWLIST}'"

tmux kill-window -t "${SESSION}:bootstrap" 2>/dev/null || true
tmux select-window -t "${SESSION}:coordinator"

echo "tmux session: ${SESSION}"
echo "run root: ${RUN_ROOT}"
tmux list-windows -t "${SESSION}" -F '#I:#W pane_pid=#{pane_pid} active=#{window_active}'
