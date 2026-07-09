#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER_SCRIPT="${SCRIPT_DIR}/run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh"
DEFAULT_NEGATIVE_PROMPT="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

usage() {
  cat <<'EOF'
Usage:
  bash run_parallel_infer_from_txt.sh \
    --input-txt /abs/path/to/list.txt \
    --weights-root /abs/path/to/step-xxxx \
    --output-root /abs/path/to/output_root \
    --method-name some_name \
    --gpus 0,2,3 \
    [--ctx 8] \
    [--output-frames 49] \
    [--negative-prompt default|empty|<text>] \
    [--disable-object-branch]

Notes:
  - The txt file will be evenly sharded by the number of GPUs.
  - Each shard is launched as a separate single-GPU worker.
  - --negative-prompt default means use DEFAULT_NEGATIVE_PROMPT.
  - --negative-prompt empty means pass an empty string.
EOF
}

INPUT_TXT=""
WEIGHTS_ROOT=""
OUTPUT_ROOT=""
METHOD_NAME=""
GPU_LIST=""
CTX="8"
OUTPUT_FRAMES="49"
NEGATIVE_PROMPT_SPEC="default"
DISABLE_OBJECT_BRANCH="0"
SHARD_ROOT_BASE="/data/gaoya/agent-data/outputs"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input-txt)
      INPUT_TXT="$2"
      shift 2
      ;;
    --weights-root)
      WEIGHTS_ROOT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --method-name)
      METHOD_NAME="$2"
      shift 2
      ;;
    --gpus)
      GPU_LIST="$2"
      shift 2
      ;;
    --ctx)
      CTX="$2"
      shift 2
      ;;
    --output-frames)
      OUTPUT_FRAMES="$2"
      shift 2
      ;;
    --negative-prompt)
      NEGATIVE_PROMPT_SPEC="$2"
      shift 2
      ;;
    --disable-object-branch)
      DISABLE_OBJECT_BRANCH="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${INPUT_TXT}" || -z "${WEIGHTS_ROOT}" || -z "${OUTPUT_ROOT}" || -z "${METHOD_NAME}" || -z "${GPU_LIST}" ]]; then
  echo "Missing required arguments." >&2
  usage >&2
  exit 1
fi

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "No GPUs parsed from --gpus ${GPU_LIST}" >&2
  exit 1
fi

for gpu in "${GPUS[@]}"; do
  gpu="$(echo "${gpu}" | xargs)"
  if [[ -z "${gpu}" ]]; then
    echo "Empty GPU id in --gpus ${GPU_LIST}" >&2
    exit 1
  fi
  if [[ "${gpu}" == "4" ]]; then
    echo "gpu4 is forbidden" >&2
    exit 1
  fi
done

INPUT_TXT="$(realpath "${INPUT_TXT}")"
WEIGHTS_ROOT="$(realpath "${WEIGHTS_ROOT}")"
OUTPUT_ROOT="$(realpath -m "${OUTPUT_ROOT}")"

if [[ ! -f "${INPUT_TXT}" ]]; then
  echo "input txt not found: ${INPUT_TXT}" >&2
  exit 1
fi
if [[ ! -d "${WEIGHTS_ROOT}" ]]; then
  echo "weights root not found: ${WEIGHTS_ROOT}" >&2
  exit 1
fi

case "${NEGATIVE_PROMPT_SPEC}" in
  default)
    NEGATIVE_PROMPT_VALUE="${DEFAULT_NEGATIVE_PROMPT}"
    ;;
  empty)
    NEGATIVE_PROMPT_VALUE=""
    ;;
  *)
    NEGATIVE_PROMPT_VALUE="${NEGATIVE_PROMPT_SPEC}"
    ;;
esac

txt_stem="$(basename "${INPUT_TXT}")"
txt_stem="${txt_stem%.*}"
step_name="$(basename "${WEIGHTS_ROOT}")"
gpu_tag="$(echo "${GPU_LIST}" | tr ',' '_')"
SHARD_ROOT="${SHARD_ROOT_BASE}/${txt_stem}_${step_name}_ctx${CTX}_gpus${gpu_tag}_shards"
mkdir -p "${SHARD_ROOT}"
rm -f "${SHARD_ROOT}"/shard_*.txt

gpu_count="${#GPUS[@]}"
awk -v mod="${gpu_count}" -v out_root="${SHARD_ROOT}" '
  NF && $0 !~ /^#/ {
    out = sprintf("%s/shard_%d.txt", out_root, n % mod)
    print > out
    n++
  }
' "${INPUT_TXT}"

echo "[split]"
for ((i=0; i<gpu_count; ++i)); do
  shard="${SHARD_ROOT}/shard_${i}.txt"
  count=0
  if [[ -f "${shard}" ]]; then
    count="$(wc -l < "${shard}")"
  fi
  echo "  shard_${i}.txt ${count}"
done

declare -a pids=()
for ((i=0; i<gpu_count; ++i)); do
  gpu="$(echo "${GPUS[$i]}" | xargs)"
  shard="${SHARD_ROOT}/shard_${i}.txt"
  if [[ ! -s "${shard}" ]]; then
    echo "[skip] gpu=${gpu} shard_${i}.txt is empty"
    continue
  fi

  (
    VISIBLE_GPU_IDS="${gpu}" \
    TEST_JSON_TXT="${shard}" \
    WEIGHTS_ROOT="${WEIGHTS_ROOT}" \
    METHOD_NAME="${METHOD_NAME}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    OUTPUT_FRAMES="${OUTPUT_FRAMES}" \
    CTX="${CTX}" \
    NEGATIVE_PROMPT="${NEGATIVE_PROMPT_VALUE}" \
    DISABLE_OBJECT_BRANCH="${DISABLE_OBJECT_BRANCH}" \
    bash "${INNER_SCRIPT}"
  ) &
  pids+=("$!")
  echo "[launch] gpu=${gpu} shard=${shard}"
done

if [[ "${#pids[@]}" -eq 0 ]]; then
  echo "No shard process launched." >&2
  exit 1
fi

wait "${pids[@]}"
echo "[done] all shard workers finished"
