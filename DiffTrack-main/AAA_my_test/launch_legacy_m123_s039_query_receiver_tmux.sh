#!/usr/bin/env bash
set -euo pipefail

SESSION=${1:-m123_s039_query_receiver_capture}
RUNNER=/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/run_legacy_m123_s039_query_receiver_gpu.sh
PILOT_COMPLETE=/data/gaoya/agent-data/outputs/object_query_attention_overlays/m123_head_scope_s039_query_receiver_v1/0613pybullet_sample_001460_w002/seed_47326/single_object__object_A__outgoing_only__top100/complete.json

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 2
fi

wait_gpu() {
  local gpu=$1
  printf '%s' "while true; do used=\$(nvidia-smi --id=${gpu} --query-gpu=memory.used --format=csv,noheader,nounits | xargs); if [[ \"\${used}\" =~ ^[0-9]+$ ]] && (( used < 4096 )); then break; fi; echo \"[\$(date -u +%FT%TZ)] waiting GPU${gpu}: \${used} MiB\"; sleep 30; done"
}

GPU2_COMMAND="$(wait_gpu 2); bash ${RUNNER} 2 0 2 8 && exec bash ${RUNNER} 2 0 2"
GPU3_COMMAND="$(wait_gpu 3); while [[ ! -f ${PILOT_COMPLETE} ]]; do echo \"[\$(date -u +%FT%TZ)] waiting M3 pilot\"; sleep 30; done; exec bash ${RUNNER} 3 1 2"

tmux new-session -d -s "${SESSION}" -n gpu2 "bash -lc '${GPU2_COMMAND}'"
tmux new-window -d -t "${SESSION}" -n gpu3 "bash -lc '${GPU3_COMMAND}'"

echo "started ${SESSION}: wait GPU2/3, validate M3 pilot task 8, then run all 108"
