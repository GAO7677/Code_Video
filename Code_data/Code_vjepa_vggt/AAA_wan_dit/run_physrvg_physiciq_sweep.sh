#!/usr/bin/env bash
set -euo pipefail

# One-case pilot over the recommended sparse layers:
# LIMIT=1 GPU_IDS=0,1 \
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_physrvg_physiciq_sweep.sh
#
# Full 67-case sparse sweep:
# GPU_IDS=0,1,2,3 \
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_physrvg_physiciq_sweep.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ONE="${SCRIPT_DIR}/run_physrvg_physiciq_one.sh"
GPU_IDS_TEXT="${GPU_IDS:-0}"
BLOCK_IDS_TEXT="${BLOCK_IDS:-0 5 11 17 19 29}"
MODES_TEXT="${MODES:-whole_block self_attn_zero text_cross_attn_zero ffn_zero lora_off}"
INCLUDE_BASELINE="${INCLUDE_BASELINE:-1}"

IFS=',' read -r -a GPU_IDS_ARRAY <<< "${GPU_IDS_TEXT}"
read -r -a BLOCK_IDS_ARRAY <<< "${BLOCK_IDS_TEXT}"
read -r -a MODES_ARRAY <<< "${MODES_TEXT}"
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

for mode in "${MODES_ARRAY[@]}"; do
  case "${mode}" in
    whole_block|self_attn_zero|text_cross_attn_zero|ffn_zero|lora_off) ;;
    *)
      echo "Invalid mode in MODES: ${mode}" >&2
      exit 2
      ;;
  esac
done

run_worker() {
  local worker_id="$1"
  local gpu_id="$2"
  local job_index=0

  run_if_assigned() {
    local mode="$1"
    local block="$2"
    if (( job_index % NUM_WORKERS == worker_id )); then
      echo "[worker ${worker_id}] gpu=${gpu_id} mode=${mode} block=${block}"
      bash "${RUN_ONE}" "${mode}" "${block}" "${gpu_id}"
    fi
    job_index=$((job_index + 1))
  }

  if [[ "${INCLUDE_BASELINE}" == "1" ]]; then
    run_if_assigned baseline none
  fi

  local block_id mode
  for block_id in "${BLOCK_IDS_ARRAY[@]}"; do
    for mode in "${MODES_ARRAY[@]}"; do
      run_if_assigned "${mode}" "${block_id}"
    done
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

