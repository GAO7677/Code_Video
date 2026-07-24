#!/usr/bin/env bash
set -euo pipefail

# Run:
#   bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_bench_fill_remaining_gpu01234_tmux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER="${SCRIPT_DIR}/run_bench_v2v_wan_queue_worker.sh"
FIND_INCOMPLETE="${SCRIPT_DIR}/find_incomplete_metric_roots.py"
PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
BASELINE_LIST="${BASELINE_LIST:-/data/gaoya/AAA_test_video/0623/test/v2v_wan/leaf_folders.txt}"
INPUT_ALLOWLIST="${INPUT_ALLOWLIST:-/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt}"
SESSION="${SESSION:-bench_v2v_wan_fill_remaining_gpu01234_20260724}"
RUN_ROOT="${RUN_ROOT:-/data/gaoya/AAA_test_video/0623/test/v2v_wan/_bench_runs/${SESSION}}"
WMREWARD_GPUS=(0 1 2 3)
VIDEOPHY2_GPUS=(1 2 3 4)

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}/queues" "${RUN_ROOT}/logs" "${RUN_ROOT}/state" "${RUN_ROOT}/task_summaries"
: > "${RUN_ROOT}/queues/wmreward_retry.tsv"
: > "${RUN_ROOT}/queues/videophy2_retry.tsv"
: > "${RUN_ROOT}/completed_tasks.tsv"
: > "${RUN_ROOT}/failed_tasks.tsv"
printf '1\n' > "${RUN_ROOT}/queues/wmreward_retry.cursor"
printf '1\n' > "${RUN_ROOT}/queues/videophy2_retry.cursor"

mapfile -t WMREWARD_ROOTS < <(
  "${PYTHON_BIN}" "${FIND_INCOMPLETE}" \
    --baseline-list "${BASELINE_LIST}" \
    --input-allowlist "${INPUT_ALLOWLIST}" \
    --metric wmreward
)
mapfile -t VIDEOPHY2_ROOTS < <(
  "${PYTHON_BIN}" "${FIND_INCOMPLETE}" \
    --baseline-list "${BASELINE_LIST}" \
    --input-allowlist "${INPUT_ALLOWLIST}" \
    --metric videophy2
)

for index in "${!WMREWARD_ROOTS[@]}"; do
  IFS=$'\t' read -r root missing total <<< "${WMREWARD_ROOTS[$index]}"
  printf 'wmretry-%04d\twmreward\t%s\n' "${index}" "${root}" \
    >> "${RUN_ROOT}/queues/wmreward_retry.tsv"
  printf 'wmreward root=%s missing=%s/%s\n' "${root}" "${missing}" "${total}" \
    >> "${RUN_ROOT}/initial_missing.txt"
done
for index in "${!VIDEOPHY2_ROOTS[@]}"; do
  IFS=$'\t' read -r root missing total <<< "${VIDEOPHY2_ROOTS[$index]}"
  printf 'vpretry-%04d\tvideophy2\t%s\n' "${index}" "${root}" \
    >> "${RUN_ROOT}/queues/videophy2_retry.tsv"
  printf 'videophy2 root=%s missing=%s/%s\n' "${root}" "${missing}" "${total}" \
    >> "${RUN_ROOT}/initial_missing.txt"
done

num_wm_tasks="${#WMREWARD_ROOTS[@]}"
num_vp_tasks="${#VIDEOPHY2_ROOTS[@]}"
total_workers=$(( ${#WMREWARD_GPUS[@]} + ${#VIDEOPHY2_GPUS[@]} ))
tmux new-session -d -s "${SESSION}" -n coordinator \
  "while true; do done_count=\$(find '${RUN_ROOT}/state' -maxdepth 1 -name '*.complete' -type f | wc -l); wm_cursor=\$(cat '${RUN_ROOT}/queues/wmreward_retry.cursor'); vp_cursor=\$(cat '${RUN_ROOT}/queues/videophy2_retry.cursor'); printf '[coordinator] workers=%s/${total_workers} wm_claimed=%s/${num_wm_tasks} vp_claimed=%s/${num_vp_tasks} completed=%s failed=%s\\n' \"\$done_count\" \"\$((wm_cursor - 1))\" \"\$((vp_cursor - 1))\" \"\$(wc -l < '${RUN_ROOT}/completed_tasks.tsv')\" \"\$(wc -l < '${RUN_ROOT}/failed_tasks.tsv')\"; [ \"\$done_count\" -eq '${total_workers}' ] && break; sleep 30; done; exec bash"

for worker_index in "${!WMREWARD_GPUS[@]}"; do
  gpu="${WMREWARD_GPUS[$worker_index]}"
  name="g${gpu}_wmretry${worker_index}"
  tmux new-window -t "${SESSION}" -n "${name}" \
    "bash '${WORKER}' '${gpu}' wmreward_retry '${name}' '${RUN_ROOT}' '${INPUT_ALLOWLIST}'"
done
for gpu in "${VIDEOPHY2_GPUS[@]}"; do
  name="g${gpu}_vpretry"
  tmux new-window -t "${SESSION}" -n "${name}" \
    "bash '${WORKER}' '${gpu}' videophy2_retry '${name}' '${RUN_ROOT}' '${INPUT_ALLOWLIST}'"
done

tmux select-window -t "${SESSION}:coordinator"
echo "tmux session: ${SESSION}"
echo "run root: ${RUN_ROOT}"
echo "workers: ${total_workers}"
echo "wmreward retry: ${num_wm_tasks} roots on GPUs 0,1,2,3"
echo "videophy2 retry: ${num_vp_tasks} roots on GPUs 1,2,3,4"
