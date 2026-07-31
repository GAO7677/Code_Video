#!/usr/bin/env bash
set -euo pipefail

# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_openvid_lora_head34_cpu_metrics_tmux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
WORKER="${SCRIPT_DIR}/run_head_role_incremental_metric_worker.sh"
SUMMARY=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/summarize_benchmark_txt_metrics.py
VERIFY=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/verify_bench_physiq_metrics.py
GENERATION_ROOT=/data/gaoya/agent-data/outputs/wan_dit_openvid_lora_head34/seed851/generation
RESULT_BASE="${GENERATION_ROOT}/openvid_lora_step10000/seed-000851"
INPUT_ALLOWLIST=/data/gaoya/agent-data/outputs/wan_dit_fulltoken_head_roles_50seeds/input_lists/test5_unique20.txt
METRIC_BASE=/data/gaoya/agent-data/outputs/wan_dit_openvid_lora_head34/metrics
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-${METRIC_BASE}/cpu_${RUN_TAG}}"
SESSION="${SESSION:-openvid_lora_head34_cpu_metrics_${RUN_TAG}}"
NUM_WORKERS="${NUM_WORKERS:-16}"
EXPECTED_ROOTS=34
EXPECTED_CASES=20

METRICS=(
  physics_iq_with_context
  physics_iq_without_context
  pmf_with_context
  pmf_without_context
)

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi
if (( NUM_WORKERS < 1 )); then
  echo "NUM_WORKERS must be >= 1" >&2
  exit 2
fi
test -x "${PYTHON}"
test -s "${INPUT_ALLOWLIST}"

mkdir -p "${RUN_ROOT}/queues" "${RUN_ROOT}/logs" \
  "${RUN_ROOT}/state" "${RUN_ROOT}/task_summaries"
find "${RESULT_BASE}" -mindepth 2 -maxdepth 2 -type d -name results \
  -print | sort > "${RUN_ROOT}/leaf_folders.snapshot.txt"

mapfile -t RESULT_ROOTS < "${RUN_ROOT}/leaf_folders.snapshot.txt"
if [[ "${#RESULT_ROOTS[@]}" -ne "${EXPECTED_ROOTS}" ]]; then
  echo "Expected ${EXPECTED_ROOTS} result roots, got ${#RESULT_ROOTS[@]}" >&2
  exit 2
fi
for result_root in "${RESULT_ROOTS[@]}"; do
  video_count="$(find "${result_root}" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
  if [[ "${video_count}" -ne "${EXPECTED_CASES}" ]]; then
    echo "Expected ${EXPECTED_CASES} videos under ${result_root}, got ${video_count}" >&2
    exit 2
  fi
done

: > "${RUN_ROOT}/queues/cpu.tsv"
: > "${RUN_ROOT}/completed_tasks.tsv"
: > "${RUN_ROOT}/failed_tasks.tsv"
printf '1\n' > "${RUN_ROOT}/queues/cpu.cursor"

task_index=0
for metric in "${METRICS[@]}"; do
  for result_root in "${RESULT_ROOTS[@]}"; do
    printf 'cpu-%04d\t%s\t%s\n' \
      "${task_index}" "${metric}" "${result_root}" \
      >> "${RUN_ROOT}/queues/cpu.tsv"
    task_index=$((task_index + 1))
  done
done
EXPECTED_TASKS="${task_index}"

tmux new-session -d -s "${SESSION}" -n monitor \
  "while true; do cursor=\$(cat '${RUN_ROOT}/queues/cpu.cursor'); complete=\$(wc -l < '${RUN_ROOT}/completed_tasks.tsv'); failed=\$(wc -l < '${RUN_ROOT}/failed_tasks.tsv'); workers=\$(find '${RUN_ROOT}/state' -maxdepth 1 -name '*.complete' -type f | wc -l); printf '[openvid-cpu-metrics] claimed=%s/${EXPECTED_TASKS} complete=%s failed=%s workers=%s/${NUM_WORKERS}\\n' \"\$((cursor - 1))\" \"\$complete\" \"\$failed\" \"\$workers\"; [ \"\$workers\" -eq '${NUM_WORKERS}' ] && break; sleep 30; done; '${PYTHON}' '${SUMMARY}' --input-txt '${RUN_ROOT}/leaf_folders.snapshot.txt' --output-csv '${RUN_ROOT}/metric_summary.csv' --input-json-allowlist '${INPUT_ALLOWLIST}'; '${PYTHON}' '${VERIFY}' --baseline-list '${RUN_ROOT}/leaf_folders.snapshot.txt' --output '${RUN_ROOT}/verification.json' --input-json-allowlist '${INPUT_ALLOWLIST}'; exec bash"

for worker_index in $(seq 0 $((NUM_WORKERS - 1))); do
  name="cpu_$(printf '%02d' "${worker_index}")"
  tmux new-window -d -t "${SESSION}" -n "${name}" \
    "CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      METRIC_WORKER_THREADS=2 bash '${WORKER}' -1 cpu '${name}' \
      '${RUN_ROOT}' '${INPUT_ALLOWLIST}' 0 0; exec bash"
done

tmux select-window -t "${SESSION}:monitor"
printf 'session=%s\nrun_root=%s\nresult_roots=%s\ntasks=%s\nworkers=%s\n' \
  "${SESSION}" "${RUN_ROOT}" "${#RESULT_ROOTS[@]}" \
  "${EXPECTED_TASKS}" "${NUM_WORKERS}"
