#!/usr/bin/env bash
set -euo pipefail

# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/status_configured_head_ablation_pipeline.sh \
#   /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/config_head_ablation_all_blocks_test5.sh

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 CONFIG" >&2
  exit 2
fi

CONFIG="$(realpath "$1")"
# shellcheck source=/dev/null
source "${CONFIG}"

queue="${RUN_ROOT}/generation/queue.tsv"
task_state="${RUN_ROOT}/generation/task_state"
total="$(wc -l < "${queue}")"
complete="$(find "${task_state}" -maxdepth 1 -name '*.complete' -type f | wc -l)"
failed="$(find "${task_state}" -maxdepth 1 -name '*.failed' -type f | wc -l)"
videos="$(find "${OUTPUT_BASE}" -type f -name '*.mp4' | wc -l)"
jsons="$(find "${OUTPUT_BASE}" -type f -name '*.json' | wc -l)"

printf 'session=%s\nconfigs=%s/%s\nfailed_configs=%s\nvideos=%s\njsons=%s\n' \
  "${SESSION}" "${complete}" "${total}" "${failed}" "${videos}" "${jsons}"
if [[ -f "${RUN_ROOT}/metrics.ready" ]]; then
  metric_done="$(wc -l < "${RUN_ROOT}/metrics/completed_tasks.tsv")"
  metric_failed="$(wc -l < "${RUN_ROOT}/metrics/failed_tasks.tsv")"
  metric_total="$(
    find "${RUN_ROOT}/metrics/queues" -name '*.tsv' -type f -print0 \
      | xargs -0 cat | wc -l
  )"
  printf 'metrics=%s/%s\nfailed_metrics=%s\n' \
    "${metric_done}" "${metric_total}" "${metric_failed}"
else
  echo "metrics=waiting_for_generation"
fi
if [[ -f "${RUN_ROOT}/pipeline.complete" ]]; then
  echo "pipeline=complete"
elif [[ -f "${RUN_ROOT}/generation.failed" || -f "${RUN_ROOT}/metrics.failed" ]]; then
  echo "pipeline=failed"
else
  echo "pipeline=running"
fi

read -r -a gpu_array <<< "${GPUS}"
nvidia-smi -i "$(IFS=,; echo "${gpu_array[*]}")" \
  --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
du -sh "${OUTPUT_BASE}"
df -h "${OUTPUT_BASE}" | tail -1
