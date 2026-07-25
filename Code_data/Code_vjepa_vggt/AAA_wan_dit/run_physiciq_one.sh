#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_physiciq_one.sh MODEL MODE BLOCK GPU_ID [HEAD]
#
# Examples:
#   bash run_physiciq_one.sh wan_lora baseline none 0
#   bash run_physiciq_one.sh wan_lora whole_block 7 0
#   bash run_physiciq_one.sh xssc self_attn_zero 12 1
#   bash run_physiciq_one.sh xssc object_cross_attn 12 1

if [[ "$#" -lt 4 || "$#" -gt 5 ]]; then
  echo "Usage: $0 MODEL MODE BLOCK GPU_ID [HEAD]" >&2
  echo "MODEL: wan_lora | xssc" >&2
  echo "MODE: baseline | whole_block | self_attn_zero | self_attn_head_zero | object_cross_attn" >&2
  echo "BLOCK: none for baseline, otherwise 0-29" >&2
  exit 2
fi

MODEL="$1"
ABLATION_MODE="$2"
BLOCK_TEXT="$3"
GPU_ID="$4"
HEAD_TEXT="${5:-none}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
TRAIN0419_ROOT=/home/gaoya/Code_Video/Code_data/Code_train/train_0419
PYTHON="${PYTHON:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"

INPUT_LIST="${INPUT_LIST:-/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt}"
OUTPUT_BASE="${OUTPUT_BASE:-/data/gaoya/AAA_test_video/0623/test/v2v_wan}"
WAN_ROOT="${WAN_ROOT:-/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B}"
WAN_LORA_ROOT="${WAN_LORA_ROOT:-/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500}"
XSSC_WEIGHTS_ROOT="${XSSC_WEIGHTS_ROOT:-/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/offcial_xSSC/train_xssc_context_slots/checkpoints/step-001500}"
XSSC_ROOT="${XSSC_ROOT:-/home/gaoya/Code_Video/xSSC-main}"
XSSC_CONFIG="${XSSC_CONFIG:-${XSSC_ROOT}/config-randsfq/rsfq2_r-ytvis.py}"
XSSC_CHECKPOINT="${XSSC_CHECKPOINT:-/data/gaoya/ckpt/xSSC/rsfq2_r-ytvis/42-0130.pth}"

HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-896}"
NUM_FRAMES="${NUM_FRAMES:-49}"
CONTEXT_FRAMES="${CONTEXT_FRAMES:-8}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
CFG_SCALE="${CFG_SCALE:-5.0}"
FPS="${FPS:-30}"
SEED="${SEED:-42}"
LIMIT="${LIMIT:-}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-模糊，低质量，变形，伪影，文字，水印，过曝，欠曝，颜色异常，几何扭曲，物体融化，物理不合理}"

if [[ -n "${LIMIT}" ]]; then
  if [[ ! "${LIMIT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "LIMIT must be a positive integer, got ${LIMIT}" >&2
    exit 2
  fi
fi

case "${MODEL}" in
  wan_lora|xssc) ;;
  *)
    echo "Unsupported MODEL=${MODEL}; expected wan_lora or xssc" >&2
    exit 2
    ;;
esac

case "${ABLATION_MODE}" in
  baseline)
    if [[ "${BLOCK_TEXT}" != "none" || "${HEAD_TEXT}" != "none" ]]; then
      echo "baseline requires BLOCK=none and HEAD=none" >&2
      exit 2
    fi
    ABLATION_TAG=baseline
    ABLATION_ARGS=(--dit-ablation-mode baseline)
    ;;
  whole_block|self_attn_zero|object_cross_attn)
    if [[ ! "${BLOCK_TEXT}" =~ ^([0-9]|[12][0-9])$ ]]; then
      echo "BLOCK must be an integer in [0, 29], got ${BLOCK_TEXT}" >&2
      exit 2
    fi
    if [[ "${MODEL}" == "wan_lora" && "${ABLATION_MODE}" == "object_cross_attn" ]]; then
      echo "object_cross_attn is only valid for MODEL=xssc" >&2
      exit 2
    fi
    if [[ "${HEAD_TEXT}" != "none" ]]; then
      echo "${ABLATION_MODE} requires HEAD=none" >&2
      exit 2
    fi
    printf -v BLOCK_PADDED "%02d" "$((10#${BLOCK_TEXT}))"
    ABLATION_TAG="${ABLATION_MODE}_block${BLOCK_PADDED}"
    ABLATION_ARGS=(
      --dit-ablation-mode "${ABLATION_MODE}"
      --dit-ablation-block "${BLOCK_TEXT}"
    )
    ;;
  self_attn_head_zero)
    if [[ ! "${BLOCK_TEXT}" =~ ^([0-9]|[12][0-9])$ ]]; then
      echo "BLOCK must be an integer in [0, 29], got ${BLOCK_TEXT}" >&2
      exit 2
    fi
    if [[ ! "${HEAD_TEXT}" =~ ^([0-9]|1[0-9]|2[0-3])$ ]]; then
      echo "HEAD must be an integer in [0, 23], got ${HEAD_TEXT}" >&2
      exit 2
    fi
    printf -v BLOCK_PADDED "%02d" "$((10#${BLOCK_TEXT}))"
    printf -v HEAD_PADDED "%02d" "$((10#${HEAD_TEXT}))"
    ABLATION_TAG="${ABLATION_MODE}_block${BLOCK_PADDED}_head${HEAD_PADDED}"
    ABLATION_ARGS=(
      --dit-ablation-mode "${ABLATION_MODE}"
      --dit-ablation-block "${BLOCK_TEXT}"
      --dit-ablation-head "${HEAD_TEXT}"
    )
    ;;
  *)
    echo "Unsupported MODE=${ABLATION_MODE}" >&2
    exit 2
    ;;
esac

if [[ ! -s "${INPUT_LIST}" ]]; then
  echo "Input list not found or empty: ${INPUT_LIST}" >&2
  exit 2
fi

EXPERIMENT_ROOT="${OUTPUT_BASE}/${MODEL}/${ABLATION_TAG}"
mkdir -p "${EXPERIMENT_ROOT}"
EFFECTIVE_INPUT_LIST="${INPUT_LIST}"
if [[ -n "${LIMIT}" ]]; then
  EFFECTIVE_INPUT_LIST="${EXPERIMENT_ROOT}/input_first_${LIMIT}.txt"
  sed -n "1,${LIMIT}p" "${INPUT_LIST}" > "${EFFECTIVE_INPUT_LIST}"
  if [[ ! -s "${EFFECTIVE_INPUT_LIST}" ]]; then
    echo "Limited input list is empty: ${EFFECTIVE_INPUT_LIST}" >&2
    exit 2
  fi
fi
{
  echo "model=${MODEL}"
  echo "ablation_mode=${ABLATION_MODE}"
  echo "block=${BLOCK_TEXT}"
  echo "head=${HEAD_TEXT}"
  echo "gpu_id=${GPU_ID}"
  echo "input_list=${INPUT_LIST}"
  echo "effective_input_list=${EFFECTIVE_INPUT_LIST}"
  echo "wan_root=${WAN_ROOT}"
  echo "wan_lora_root=${WAN_LORA_ROOT}"
  echo "xssc_weights_root=${XSSC_WEIGHTS_ROOT}"
  echo "xssc_config=${XSSC_CONFIG}"
  echo "xssc_checkpoint=${XSSC_CHECKPOINT}"
  echo "xssc_preprocess_mode=center_crop"
  echo "xssc_slot_temporal_mode=full"
  echo "height=${HEIGHT}"
  echo "width=${WIDTH}"
  echo "num_frames=${NUM_FRAMES}"
  echo "context_frames=${CONTEXT_FRAMES}"
  echo "num_inference_steps=${NUM_INFERENCE_STEPS}"
  echo "cfg_scale=${CFG_SCALE}"
  echo "fps=${FPS}"
  echo "seed=${SEED}"
  echo "limit=${LIMIT:-all}"
  echo "negative_prompt=${NEGATIVE_PROMPT}"
} > "${EXPERIMENT_ROOT}/ablation_config.txt"

COMMON_ENV=(
  PYTHONPATH="${PROJECT_ROOT}:${DIFFSYNTH_ROOT}:${TRAIN0419_ROOT}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
)

if [[ "${MODEL}" == "wan_lora" ]]; then
  exec env "${COMMON_ENV[@]}" \
    "${PYTHON}" "${SCRIPT_DIR}/infer_wan_lora_dit_ablation.py" \
    "${ABLATION_ARGS[@]}" \
    --weights-root "${WAN_LORA_ROOT}" \
    --input-json-list-path "${EFFECTIVE_INPUT_LIST}" \
    --model-name "wan_lora_${ABLATION_TAG}" \
    --wan-root "${WAN_ROOT}" \
    --output-root "${EXPERIMENT_ROOT}" \
    --runtime-root "${EXPERIMENT_ROOT}/_runtime" \
    --device cuda \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --num-frames "${NUM_FRAMES}" \
    --context-frames "${CONTEXT_FRAMES}" \
    --conditioning-mode context_aware \
    --context-resize-mode crop \
    --num-inference-steps "${NUM_INFERENCE_STEPS}" \
    --cfg-scale "${CFG_SCALE}" \
    --fps "${FPS}" \
    --seed "${SEED}" \
    --negative-prompt "${NEGATIVE_PROMPT}"
fi

exec env "${COMMON_ENV[@]}" \
  XSSC_ROOT="${XSSC_ROOT}" \
  XSSC_CONFIG="${XSSC_CONFIG}" \
  XSSC_CHECKPOINT="${XSSC_CHECKPOINT}" \
  XSSC_PREPROCESS_MODE=center_crop \
  XSSC_SLOT_TEMPORAL_MODE=full \
  "${PYTHON}" "${SCRIPT_DIR}/infer_xssc_dit_ablation.py" \
  "${ABLATION_ARGS[@]}" \
  --weights-root "${XSSC_WEIGHTS_ROOT}" \
  --input-json-list-path "${EFFECTIVE_INPUT_LIST}" \
  --model-name "xssc_${ABLATION_TAG}" \
  --output-root "${EXPERIMENT_ROOT}" \
  --step-output-dir-name results \
  --wan-root "${WAN_ROOT}" \
  --lora-checkpoint "${WAN_LORA_ROOT}/checkpoint.safetensors" \
  --device cuda:0 \
  --aux-device cuda:0 \
  --inference-devices cuda:0,cuda:0 \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --num-frames "${NUM_FRAMES}" \
  --context-frames "${CONTEXT_FRAMES}" \
  --sampling-mode prefix \
  --num-inference-steps "${NUM_INFERENCE_STEPS}" \
  --cfg-scale "${CFG_SCALE}" \
  --fps "${FPS}" \
  --seed "${SEED}" \
  --negative-prompt "${NEGATIVE_PROMPT}"
