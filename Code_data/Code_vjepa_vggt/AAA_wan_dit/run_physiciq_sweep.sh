#!/usr/bin/env bash
set -euo pipefail

# Full default sweep:
# GPU_IDS=0,1,2,3 \
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_physiciq_sweep.sh
#
# Smaller pilot:
# GPU_IDS=0,1 BLOCK_IDS="0 5 10 15 20 25 29" INCLUDE_OBJECT_CROSS_ATTN=0 \
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_physiciq_sweep.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ONE="${SCRIPT_DIR}/run_physiciq_one.sh"
GPU_IDS_TEXT="${GPU_IDS:-0}"
BLOCK_IDS_TEXT="${BLOCK_IDS:-$(seq 0 29 | tr '\n' ' ')}"
INCLUDE_BASELINE="${INCLUDE_BASELINE:-1}"
INCLUDE_OBJECT_CROSS_ATTN="${INCLUDE_OBJECT_CROSS_ATTN:-1}"

IFS=',' read -r -a GPU_IDS_ARRAY <<< "${GPU_IDS_TEXT}"
read -r -a BLOCK_IDS_ARRAY <<< "${BLOCK_IDS_TEXT}"
NUM_WORKERS="${#GPU_IDS_ARRAY[@]}"

if (( NUM_WORKERS == 0 )); then
  echo "GPU_IDS is empty" >&2
  exit 2
fi

for block_id in "${BLOCK_IDS_ARRAY[@]}"; do
  if [[ ! "${block_id}" =~ ^([0-9]|[12][0-9])$ ]]; then
    echo "Invalid block id in BLOCK_IDS: ${block_id}" >&2
    exit 2
  fi
done

run_worker() {
  local worker_id="$1"
  local gpu_id="$2"
  local job_index=0

  run_if_assigned() {
    local model="$1"
    local mode="$2"
    local block="$3"
    if (( job_index % NUM_WORKERS == worker_id )); then
      echo "[worker ${worker_id}] gpu=${gpu_id} model=${model} mode=${mode} block=${block}"
      bash "${RUN_ONE}" "${model}" "${mode}" "${block}" "${gpu_id}"
    fi
    job_index=$((job_index + 1))
  }

  if [[ "${INCLUDE_BASELINE}" == "1" ]]; then
    run_if_assigned wan_lora baseline none
    run_if_assigned xssc baseline none
  fi

  local block_id
  for block_id in "${BLOCK_IDS_ARRAY[@]}"; do
    run_if_assigned wan_lora whole_block "${block_id}"
    run_if_assigned wan_lora self_attn_zero "${block_id}"
    run_if_assigned xssc whole_block "${block_id}"
    run_if_assigned xssc self_attn_zero "${block_id}"
    if [[ "${INCLUDE_OBJECT_CROSS_ATTN}" == "1" ]]; then
      run_if_assigned xssc object_cross_attn "${block_id}"
    fi
  done
}

pids=()
for worker_id in "${!GPU_IDS_ARRAY[@]}"; do
  gpu_id="${GPU_IDS_ARRAY[$worker_id]}"
  gpu_id="${gpu_id//[[:space:]]/}"
  if [[ -z "${gpu_id}" ]]; then
    echo "Empty GPU id at worker ${worker_id}" >&2
    exit 2
  fi
  run_worker "${worker_id}" "${gpu_id}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
