#!/usr/bin/env bash
set -euo pipefail

# Wait until one GPU in a candidate set is free, then run the given command on it.
#
# Typical tmux foreground usage:
# tmux new-window -n gpuwait \
#   'bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/wait_gpu012_and_run.sh \
#      --gpus 0,1,2 \
#      -- bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/run_formal_compare_morpheus_physiciq_wan_base_two_loras_ti2v_t2v_gpu023.sh'
#
# Same wrapper, but only launch a filtered subset:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/wait_gpu012_and_run.sh \
#   --gpus 0,1,2 \
#   -- env TARGET_DATASETS=physicIQ TARGET_MODES=ti2v OVERWRITE=1 \
#      bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/run_formal_compare_morpheus_physiciq_wan_base_two_loras_ti2v_t2v_gpu023.sh

GPUS="0,1,2"
POLL_SECONDS=30
MAX_MEMORY_MB=1024
MAX_UTILIZATION=10
MAX_WAIT_SECONDS=0

usage() {
  cat <<'EOF'
Usage:
  bash wait_gpu012_and_run.sh [options] -- <command> [args...]

Options:
  --gpus 0,1,2              Candidate GPUs. Default: 0,1,2
  --poll-seconds 30         Poll interval. Default: 30
  --max-memory-mb 1024      Treat GPU as busy if memory.used is above this threshold.
  --max-utilization 10      Treat GPU as busy if utilization.gpu is above this threshold.
  --max-wait-seconds 0      0 means wait forever. Default: 0
  -h, --help                Show help.

The selected GPU id is exported as:
  CUDA_VISIBLE_DEVICES=<gpu>
and the command is then executed in the foreground with exec.
EOF
}

while (($# > 0)); do
  case "$1" in
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --poll-seconds)
      POLL_SECONDS="$2"
      shift 2
      ;;
    --max-memory-mb)
      MAX_MEMORY_MB="$2"
      shift 2
      ;;
    --max-utilization)
      MAX_UTILIZATION="$2"
      shift 2
      ;;
    --max-wait-seconds)
      MAX_WAIT_SECONDS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if (($# == 0)); then
  echo "Missing command after --" >&2
  usage >&2
  exit 2
fi

IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
for gpu in "${GPU_LIST[@]}"; do
  if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
    echo "Invalid gpu id in --gpus: ${gpu}" >&2
    exit 2
  fi
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found" >&2
  exit 127
fi

START_TS="$(date +%s)"

gpu_is_free() {
  local gpu="$1"
  local row
  row="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits -i "${gpu}" 2>/dev/null | head -n 1 || true)"
  if [[ -z "${row}" ]]; then
    return 1
  fi

  local index
  local memory_used
  local utilization
  IFS=',' read -r index memory_used utilization <<< "${row}"
  index="$(echo "${index}" | xargs)"
  memory_used="$(echo "${memory_used}" | xargs)"
  utilization="$(echo "${utilization}" | xargs)"

  local app_count
  app_count="$(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i "${gpu}" 2>/dev/null \
      | sed '/^[[:space:]]*$/d' \
      | wc -l
  )"

  if (( app_count > 0 )); then
    return 1
  fi
  if (( memory_used > MAX_MEMORY_MB )); then
    return 1
  fi
  if (( utilization > MAX_UTILIZATION )); then
    return 1
  fi
  return 0
}

describe_gpu() {
  local gpu="$1"
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits -i "${gpu}" 2>/dev/null \
    | head -n 1 \
    | sed 's/^[[:space:]]*//'
}

find_free_gpu() {
  local gpu
  for gpu in "${GPU_LIST[@]}"; do
    if gpu_is_free "${gpu}"; then
      echo "${gpu}"
      return 0
    fi
  done
  return 1
}

while true; do
  if FREE_GPU="$(find_free_gpu)"; then
    echo "[gpu_wait] selected_gpu=${FREE_GPU} status=$(describe_gpu "${FREE_GPU}")"
    echo "[gpu_wait] exec: CUDA_VISIBLE_DEVICES=${FREE_GPU} $*"
    export CUDA_VISIBLE_DEVICES="${FREE_GPU}"
    exec "$@"
  fi

  local_now="$(date '+%F %T')"
  echo "[gpu_wait] ${local_now} no_free_gpu yet among ${GPUS}; thresholds memory<=${MAX_MEMORY_MB}MB util<=${MAX_UTILIZATION}% poll=${POLL_SECONDS}s"
  for gpu in "${GPU_LIST[@]}"; do
    echo "[gpu_wait] busy: $(describe_gpu "${gpu}")"
  done

  if (( MAX_WAIT_SECONDS > 0 )); then
    now_ts="$(date +%s)"
    waited="$((now_ts - START_TS))"
    if (( waited >= MAX_WAIT_SECONDS )); then
      echo "[gpu_wait] timeout after ${waited}s without a free gpu in ${GPUS}" >&2
      exit 124
    fi
  fi

  sleep "${POLL_SECONDS}"
done
