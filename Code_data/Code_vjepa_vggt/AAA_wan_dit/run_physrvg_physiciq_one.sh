#!/usr/bin/env bash
set -euo pipefail

# Run:
# LIMIT=1 bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_physrvg_physiciq_one.sh baseline none 0
# LIMIT=1 bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_physrvg_physiciq_one.sh self_attn_zero 5 0
#
# Usage: run_physrvg_physiciq_one.sh MODE BLOCK GPU_ID

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 MODE BLOCK GPU_ID" >&2
  echo "MODE: baseline | whole_block | self_attn_zero | text_cross_attn_zero | ffn_zero | lora_off" >&2
  echo "BLOCK: none for baseline, otherwise 0-29" >&2
  exit 2
fi

MODE="$1"
BLOCK_TEXT="$2"
GPU_ID="$3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHYSRVG_ROOT="${PHYSRVG_ROOT:-/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_phys_papers_compare/PhysRVG-main}"
PYTHON="${PYTHON:-/data/gaoya/miniconda3/envs/vjepa2/bin/python}"

INPUT_LIST="${INPUT_LIST:-/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt}"
OUTPUT_BASE="${OUTPUT_BASE:-/data/gaoya/AAA_test_video/0623/test/v2v_wan/PhyRVG}"
MODEL_ID="${MODEL_ID:-/data/gaoya/ckpt/HappyP4nda-PhysRVG/Wan2.2-TI2V-5B-Diffusers}"
DIT_CHECKPOINT="${DIT_CHECKPOINT:-/data/gaoya/ckpt/HappyP4nda-PhysRVG/dit/diffusion_pytorch_model.safetensors}"
LORA_CHECKPOINT="${LORA_CHECKPOINT:-/data/gaoya/ckpt/HappyP4nda-PhysRVG/lora/checkpoint}"

# Matched to the previous xSSC ablation run, except PhysRVG keeps CFG disabled.
HEIGHT=512
WIDTH=896
NUM_FRAMES=49
CONTEXT_FRAMES=8
NUM_INFERENCE_STEPS=40
GUIDANCE_SCALE=5.0
DO_CFG=0
FPS=30
SEED=42
LIMIT="${LIMIT:-}"
NEGATIVE_PROMPT="模糊，低质量，变形，伪影，文字，水印，过曝，欠曝，颜色异常，几何扭曲，物体融化，物理不合理"

case "${MODE}" in
  baseline)
    if [[ "${BLOCK_TEXT}" != "none" ]]; then
      echo "baseline requires BLOCK=none" >&2
      exit 2
    fi
    TAG=baseline
    ABLATION_ARGS=(--physrvg-ablation-mode baseline)
    ;;
  whole_block|self_attn_zero|text_cross_attn_zero|ffn_zero|lora_off)
    if [[ ! "${BLOCK_TEXT}" =~ ^([0-9]|[12][0-9])$ ]]; then
      echo "BLOCK must be an integer in [0, 29], got ${BLOCK_TEXT}" >&2
      exit 2
    fi
    printf -v BLOCK_PADDED "%02d" "$((10#${BLOCK_TEXT}))"
    TAG="${MODE}_block${BLOCK_PADDED}"
    ABLATION_ARGS=(
      --physrvg-ablation-mode "${MODE}"
      --physrvg-ablation-block "${BLOCK_TEXT}"
    )
    ;;
  *)
    echo "Unsupported MODE=${MODE}" >&2
    exit 2
    ;;
esac

if [[ ! -s "${INPUT_LIST}" ]]; then
  echo "Input list not found or empty: ${INPUT_LIST}" >&2
  exit 2
fi
if [[ ! -s "${DIT_CHECKPOINT}" ]]; then
  echo "PhysRVG DiT checkpoint is not downloaded yet: ${DIT_CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -s "${LORA_CHECKPOINT}/adapter_model.safetensors" ]]; then
  echo "PhysRVG LoRA checkpoint not found: ${LORA_CHECKPOINT}" >&2
  exit 2
fi

EXPERIMENT_ROOT="${OUTPUT_BASE}/${TAG}"
mkdir -p "${EXPERIMENT_ROOT}"
EFFECTIVE_INPUT_LIST="${INPUT_LIST}"
if [[ -n "${LIMIT}" ]]; then
  if [[ ! "${LIMIT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "LIMIT must be a positive integer, got ${LIMIT}" >&2
    exit 2
  fi
  EFFECTIVE_INPUT_LIST="${EXPERIMENT_ROOT}/input_first_${LIMIT}.txt"
  sed -n "1,${LIMIT}p" "${INPUT_LIST}" > "${EFFECTIVE_INPUT_LIST}"
fi

{
  echo "model=PhysRVG"
  echo "ablation_mode=${MODE}"
  echo "block=${BLOCK_TEXT}"
  echo "gpu_id=${GPU_ID}"
  echo "input_list=${INPUT_LIST}"
  echo "effective_input_list=${EFFECTIVE_INPUT_LIST}"
  echo "model_id=${MODEL_ID}"
  echo "dit_checkpoint=${DIT_CHECKPOINT}"
  echo "lora_checkpoint=${LORA_CHECKPOINT}"
  echo "height=${HEIGHT}"
  echo "width=${WIDTH}"
  echo "num_frames=${NUM_FRAMES}"
  echo "context_frames=${CONTEXT_FRAMES}"
  echo "context_policy=all_8_json_input_video_frames"
  echo "prompt_policy=input_caption_from_physiciq_json"
  echo "config_policy=matched_to_previous_xssc_except_cfg"
  echo "num_inference_steps=${NUM_INFERENCE_STEPS}"
  echo "guidance_scale=${GUIDANCE_SCALE}"
  echo "do_cfg=${DO_CFG}"
  echo "fps=${FPS}"
  echo "seed=${SEED}"
  echo "negative_prompt=${NEGATIVE_PROMPT}"
  echo "limit=${LIMIT:-all}"
} > "${EXPERIMENT_ROOT}/ablation_config.txt"

exec env \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  PYTHONPATH="${PHYSRVG_ROOT}:${SCRIPT_DIR}" \
  PYTHONNOUSERSITE=0 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PYTHON}" "${SCRIPT_DIR}/infer_physrvg_dit_ablation.py" \
  "${ABLATION_ARGS[@]}" \
  --expected-context-frames "${CONTEXT_FRAMES}" \
  --physrvg-root "${PHYSRVG_ROOT}" \
  --input-json-list-paths "${EFFECTIVE_INPUT_LIST}" \
  --output-root "${EXPERIMENT_ROOT}" \
  --model-id "${MODEL_ID}" \
  --dit-checkpoint "${DIT_CHECKPOINT}" \
  --lora-checkpoint "${LORA_CHECKPOINT}" \
  --device cuda:0 \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --num-frames "${NUM_FRAMES}" \
  --fps "${FPS}" \
  --num-inference-steps "${NUM_INFERENCE_STEPS}" \
  --guidance-scale "${GUIDANCE_SCALE}" \
  --seed "${SEED}"
