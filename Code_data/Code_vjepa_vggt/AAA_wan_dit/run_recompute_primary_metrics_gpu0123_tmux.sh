#!/usr/bin/env bash
set -euo pipefail

# Run:
#   bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_recompute_primary_metrics_gpu0123_tmux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEO_WORKER="${SCRIPT_DIR}/run_recompute_videophy2_generated_only_worker.sh"
QUEUE_WORKER="${SCRIPT_DIR}/run_bench_v2v_wan_queue_worker.sh"
COORDINATOR="${SCRIPT_DIR}/run_recompute_primary_metrics_coordinator.sh"
PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python

RESULT_BASE=/data/gaoya/AAA_test_video/0623/test/v2v_wan
PRIMARY_LIST="${RESULT_BASE}/leaf_folders.txt"
PHYRVG_LIST="${RESULT_BASE}/PhyRVG/rvg_leaf_folders.txt"
INPUT_ALLOWLIST=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt
SESSION="${SESSION:-recompute_primary_metrics_gpu0123_20260725}"
RUN_ROOT="${RUN_ROOT:-${RESULT_BASE}/_bench_runs/${SESSION}}"
GPUS=(0 1 2 3)
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
GPU_MAX_USED_MIB="${GPU_MAX_USED_MIB:-26000}"
EXPECTED_ROOTS=32

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi
if [[ -e "${RUN_ROOT}" ]]; then
  echo "run root already exists: ${RUN_ROOT}" >&2
  exit 1
fi
if [[ ! -s "${PRIMARY_LIST}" || ! -s "${PHYRVG_LIST}" || ! -s "${INPUT_ALLOWLIST}" ]]; then
  echo "Missing result-root list or input allowlist" >&2
  exit 2
fi

mapfile -t ROOTS < <(sed '/^[[:space:]]*$/d; /^[[:space:]]*#/d' "${PRIMARY_LIST}")
if [[ "${#ROOTS[@]}" -ne "${EXPECTED_ROOTS}" ]]; then
  echo "Expected ${EXPECTED_ROOTS} primary roots, got ${#ROOTS[@]}" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}/queues" "${RUN_ROOT}/logs" "${RUN_ROOT}/state" "${RUN_ROOT}/task_summaries"
: > "${RUN_ROOT}/queues/videophy2.tsv"
: > "${RUN_ROOT}/queues/cosmos.tsv"
: > "${RUN_ROOT}/completed_tasks.tsv"
: > "${RUN_ROOT}/failed_tasks.tsv"
printf '1\n' > "${RUN_ROOT}/queues/videophy2.cursor"
printf '1\n' > "${RUN_ROOT}/queues/cosmos.cursor"
cp "${PRIMARY_LIST}" "${RUN_ROOT}/primary_leaf_folders.snapshot.txt"
cp "${PHYRVG_LIST}" "${RUN_ROOT}/physrvg_leaf_folders.snapshot.txt"
cp "${INPUT_ALLOWLIST}" "${RUN_ROOT}/input_allowlist.snapshot.txt"

XSSC_LIST="${RUN_ROOT}/xssc_leaf_folders.snapshot.txt"
printf '%s\n' "${ROOTS[@]}" | grep '/xssc/' > "${XSSC_LIST}"

for index in "${!ROOTS[@]}"; do
  printf 'videophy2-%04d\t%s\t0\n' "${index}" "${ROOTS[$index]}" \
    >> "${RUN_ROOT}/queues/videophy2.tsv"
done

cosmos_index=0
while IFS= read -r root; do
  if ! "${PYTHON_BIN}" "${SCRIPT_DIR}/verify_metric_completion.py" \
      --result-roots <(printf '%s\n' "${root}") \
      --input-json-allowlist "${INPUT_ALLOWLIST}" \
      --metric cosmos_reason1 \
      --required-field score \
      --expected-cases 67 \
      --output "${RUN_ROOT}/cosmos_probe.json" >/dev/null 2>&1; then
    printf 'cosmos-%04d\tcosmos_reason1\t%s\n' "${cosmos_index}" "${root}" \
      >> "${RUN_ROOT}/queues/cosmos.tsv"
    cosmos_index=$((cosmos_index + 1))
  fi
done < "${XSSC_LIST}"

VIDEO_WORKERS=$(( ${#GPUS[@]} * WORKERS_PER_GPU ))
COSMOS_WORKERS="${VIDEO_WORKERS}"
touch "${RUN_ROOT}/video.start.ready"

tmux new-session -d -s "${SESSION}" -n coordinator \
  "bash '${COORDINATOR}' '${RUN_ROOT}' '${PRIMARY_LIST}' '${XSSC_LIST}' '${PHYRVG_LIST}' '${INPUT_ALLOWLIST}' '${VIDEO_WORKERS}' '${COSMOS_WORKERS}' '${RESULT_BASE}'; exec bash"

for gpu in "${GPUS[@]}"; do
  for worker_index in $(seq 0 $((WORKERS_PER_GPU - 1))); do
    video_name="g${gpu}_vp${worker_index}"
    tmux new-window -t "${SESSION}" -n "${video_name}" \
      "bash '${VIDEO_WORKER}' '${gpu}' '${video_name}' '${RUN_ROOT}' '${INPUT_ALLOWLIST}' 67 '${GPU_MAX_USED_MIB}' '${RUN_ROOT}/video.start.ready' '${RUN_ROOT}/unused_prior_state' 0; exec bash"

    cosmos_name="g${gpu}_cosmos${worker_index}"
    tmux new-window -t "${SESSION}" -n "${cosmos_name}" \
      "while [[ ! -f '${RUN_ROOT}/cosmos.start.ready' ]]; do sleep 30; done; bash '${QUEUE_WORKER}' '${gpu}' cosmos '${cosmos_name}' '${RUN_ROOT}' '${INPUT_ALLOWLIST}'; exec bash"
  done
done

tmux select-window -t "${SESSION}:coordinator"
echo "tmux session: ${SESSION}"
echo "run root: ${RUN_ROOT}"
echo "VideoPhy2: ${#ROOTS[@]} roots, ${VIDEO_WORKERS} workers on GPUs 0,1,2,3"
echo "Cosmos: ${cosmos_index} incomplete roots, ${COSMOS_WORKERS} workers after VideoPhy2"
