#!/usr/bin/env bash
set -euo pipefail

# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_openvid_lora_head34_gpu2_metrics_tmux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
WORKER="${SCRIPT_DIR}/run_head_role_incremental_metric_worker.sh"
AUGMENT="${SCRIPT_DIR}/augment_openvid_head34_case_gallery.py"
BASELINE_COMPARISON="${SCRIPT_DIR}/build_openvid_baseline_comparison.py"
HEAD_ANALYSIS="${SCRIPT_DIR}/build_openvid_head_ablation_analysis.py"
SUMMARY=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/summarize_benchmark_txt_metrics.py
VERIFY=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/verify_bench_physiq_metrics.py
RESULT_BASE=/data/gaoya/agent-data/outputs/wan_dit_openvid_lora_head34/seed851/generation/openvid_lora_step10000/seed-000851
INPUT_ALLOWLIST=/data/gaoya/agent-data/outputs/wan_dit_fulltoken_head_roles_50seeds/input_lists/test5_unique20.txt
METRIC_BASE=/data/gaoya/agent-data/outputs/wan_dit_openvid_lora_head34/metrics
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-${METRIC_BASE}/gpu2_${RUN_TAG}}"
SESSION="${SESSION:-openvid_lora_head34_gpu2_metrics_${RUN_TAG}}"
GPU_ID=2
MIN_FREE_MIB="${MIN_FREE_MIB:-26000}"
EXPECTED_ROOTS=34
EXPECTED_CASES=20

COMMON_METRICS=(
  wmreward
  vbench_subject_consistency
  vbench_background_consistency
  vbench_temporal_flickering
  vbench_motion_smoothness
  vbench_dynamic_degree
  vbench_aesthetic_quality
  vbench_imaging_quality
)

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi
mkdir -p "${RUN_ROOT}/queues" "${RUN_ROOT}/logs" \
  "${RUN_ROOT}/state" "${RUN_ROOT}/task_summaries"
find "${RESULT_BASE}" -mindepth 2 -maxdepth 2 -type d -name results \
  -print | sort > "${RUN_ROOT}/leaf_folders.snapshot.txt"
mapfile -t RESULT_ROOTS < "${RUN_ROOT}/leaf_folders.snapshot.txt"
if [[ "${#RESULT_ROOTS[@]}" -ne "${EXPECTED_ROOTS}" ]]; then
  echo "Expected ${EXPECTED_ROOTS} result roots, got ${#RESULT_ROOTS[@]}" >&2
  exit 2
fi

for kind in common videophy2 cosmos; do
  : > "${RUN_ROOT}/queues/${kind}.tsv"
  printf '1\n' > "${RUN_ROOT}/queues/${kind}.cursor"
done
: > "${RUN_ROOT}/completed_tasks.tsv"
: > "${RUN_ROOT}/failed_tasks.tsv"

index=0
for metric in "${COMMON_METRICS[@]}"; do
  for result_root in "${RESULT_ROOTS[@]}"; do
    printf 'common-%04d\t%s\t%s\n' "${index}" "${metric}" "${result_root}" \
      >> "${RUN_ROOT}/queues/common.tsv"
    index=$((index + 1))
  done
done
COMMON_TASKS="${index}"
index=0
for result_root in "${RESULT_ROOTS[@]}"; do
  printf 'videophy2-%04d\tvideophy2\t%s\n' "${index}" "${result_root}" \
    >> "${RUN_ROOT}/queues/videophy2.tsv"
  index=$((index + 1))
done
VIDEOPHY2_TASKS="${index}"
index=0
for result_root in "${RESULT_ROOTS[@]}"; do
  printf 'cosmos-%04d\tcosmos_reason1\t%s\n' "${index}" "${result_root}" \
    >> "${RUN_ROOT}/queues/cosmos.tsv"
  index=$((index + 1))
done
COSMOS_TASKS="${index}"
TOTAL_TASKS=$((COMMON_TASKS + VIDEOPHY2_TASKS + COSMOS_TASKS))
EXPECTED_WORKERS=5

tmux new-session -d -s "${SESSION}" -n monitor \
  "while true; do complete=\$(wc -l < '${RUN_ROOT}/completed_tasks.tsv'); failed=\$(wc -l < '${RUN_ROOT}/failed_tasks.tsv'); workers=\$(find '${RUN_ROOT}/state' -maxdepth 1 -name '*.complete' -type f | wc -l); printf '[openvid-gpu2-metrics] complete=%s/${TOTAL_TASKS} failed=%s workers=%s/${EXPECTED_WORKERS}\\n' \"\$complete\" \"\$failed\" \"\$workers\"; [ \"\$workers\" -eq '${EXPECTED_WORKERS}' ] && break; sleep 30; done; '${PYTHON}' '${SUMMARY}' --input-txt '${RUN_ROOT}/leaf_folders.snapshot.txt' --output-csv '${RUN_ROOT}/metric_summary.csv' --input-json-allowlist '${INPUT_ALLOWLIST}'; '${PYTHON}' '${VERIFY}' --baseline-list '${RUN_ROOT}/leaf_folders.snapshot.txt' --output '${RUN_ROOT}/verification.json' --input-json-allowlist '${INPUT_ALLOWLIST}'; '${PYTHON}' '${AUGMENT}'; '${PYTHON}' '${BASELINE_COMPARISON}'; '${PYTHON}' '${HEAD_ANALYSIS}'; exec bash"

for index in 0 1; do
  name="common_${index}"
  delay=$((index * 30))
  tmux new-window -d -t "${SESSION}" -n "${name}" \
    "sleep '${delay}'; METRIC_WORKER_THREADS=2 bash '${WORKER}' '${GPU_ID}' common '${name}' '${RUN_ROOT}' '${INPUT_ALLOWLIST}' '${MIN_FREE_MIB}' 10; exec bash"
done
for index in 0 1; do
  name="videophy2_${index}"
  delay=$((index * 30))
  tmux new-window -d -t "${SESSION}" -n "${name}" \
    "while [ \"\$(find '${RUN_ROOT}/state' -maxdepth 1 -name 'common_*.complete' -type f | wc -l)\" -lt 2 ]; do sleep 30; done; sleep '${delay}'; METRIC_WORKER_THREADS=2 bash '${WORKER}' '${GPU_ID}' videophy2 '${name}' '${RUN_ROOT}' '${INPUT_ALLOWLIST}' '${MIN_FREE_MIB}' 10; exec bash"
done
tmux new-window -d -t "${SESSION}" -n cosmos \
  "while [ \"\$(find '${RUN_ROOT}/state' -maxdepth 1 -name 'videophy2_*.complete' -type f | wc -l)\" -lt 2 ]; do sleep 30; done; METRIC_WORKER_THREADS=2 bash '${WORKER}' '${GPU_ID}' cosmos cosmos '${RUN_ROOT}' '${INPUT_ALLOWLIST}' '${MIN_FREE_MIB}' 10; exec bash"

tmux select-window -t "${SESSION}:monitor"
printf 'session=%s\nrun_root=%s\ngpu=%s\ntasks=%s (common=%s videophy2=%s cosmos=%s)\n' \
  "${SESSION}" "${RUN_ROOT}" "${GPU_ID}" "${TOTAL_TASKS}" \
  "${COMMON_TASKS}" "${VIDEOPHY2_TASKS}" "${COSMOS_TASKS}"
