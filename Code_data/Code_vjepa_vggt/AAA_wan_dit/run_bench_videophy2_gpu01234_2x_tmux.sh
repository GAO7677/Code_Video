#!/usr/bin/env bash
set -euo pipefail

# Run:
#   bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_bench_videophy2_gpu01234_2x_tmux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${SCRIPT_DIR}/run_bench_v2v_wan_queue_worker.sh"
BASELINE_LIST="${BASELINE_LIST:-/data/gaoya/AAA_test_video/0623/test/v2v_wan/leaf_folders.txt}"
INPUT_ALLOWLIST="${INPUT_ALLOWLIST:-/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt}"
SESSION="${SESSION:-bench_v2v_wan_videophy2_gpu01234_2x_20260724}"
RUN_ROOT="${RUN_ROOT:-/data/gaoya/AAA_test_video/0623/test/v2v_wan/_bench_runs/${SESSION}}"
GPUS=(0 1 2 3 4)
WORKERS_PER_GPU=2

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi
if [[ ! -s "${BASELINE_LIST}" || ! -s "${INPUT_ALLOWLIST}" ]]; then
  echo "Missing baseline list or input allowlist" >&2
  exit 2
fi

mapfile -t RESULT_ROOTS < <(sed '/^[[:space:]]*$/d; /^[[:space:]]*#/d' "${BASELINE_LIST}")
if [[ "${#RESULT_ROOTS[@]}" -ne 32 ]]; then
  echo "Expected 32 result roots, got ${#RESULT_ROOTS[@]}" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}/queues" "${RUN_ROOT}/logs" "${RUN_ROOT}/state" "${RUN_ROOT}/task_summaries"
: > "${RUN_ROOT}/queues/videophy2.tsv"
: > "${RUN_ROOT}/completed_tasks.tsv"
: > "${RUN_ROOT}/failed_tasks.tsv"
printf '1\n' > "${RUN_ROOT}/queues/videophy2.cursor"
cp "${BASELINE_LIST}" "${RUN_ROOT}/leaf_folders.snapshot.txt"

for index in "${!RESULT_ROOTS[@]}"; do
  root="${RESULT_ROOTS[$index]}"
  if [[ ! -d "${root}" ]]; then
    echo "Missing result root: ${root}" >&2
    exit 2
  fi
  printf 'videophy2-%04d\tvideophy2\t%s\n' "${index}" "${root}" \
    >> "${RUN_ROOT}/queues/videophy2.tsv"
done

total_workers=$(( ${#GPUS[@]} * WORKERS_PER_GPU ))
tmux new-session -d -s "${SESSION}" -n coordinator \
  "while true; do done_count=\$(find '${RUN_ROOT}/state' -maxdepth 1 -name '*.complete' -type f | wc -l); cursor=\$(cat '${RUN_ROOT}/queues/videophy2.cursor'); printf '[coordinator] workers=%s/${total_workers} claimed=%s/32 completed=%s failed=%s\\n' \"\$done_count\" \"\$((cursor - 1))\" \"\$(wc -l < '${RUN_ROOT}/completed_tasks.tsv')\" \"\$(wc -l < '${RUN_ROOT}/failed_tasks.tsv')\"; [ \"\$done_count\" -eq '${total_workers}' ] && break; sleep 30; done; exec bash"

for gpu in "${GPUS[@]}"; do
  for worker_index in $(seq 0 $((WORKERS_PER_GPU - 1))); do
    name="g${gpu}_videophy${worker_index}"
    tmux new-window -t "${SESSION}" -n "${name}" \
      "bash '${WORKER}' '${gpu}' videophy2 '${name}' '${RUN_ROOT}' '${INPUT_ALLOWLIST}'"
  done
done

tmux select-window -t "${SESSION}:coordinator"
echo "tmux session: ${SESSION}"
echo "run root: ${RUN_ROOT}"
echo "workers: ${total_workers} (${WORKERS_PER_GPU} per GPU on 0,1,2,3,4)"
echo "tasks: 32 VideoPhy2 result roots"
