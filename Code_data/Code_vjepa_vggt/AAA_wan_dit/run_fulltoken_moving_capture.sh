#!/usr/bin/env bash
set -euo pipefail

# Preflight:
# MODEL=wan_lora GPU=3 BLOCKS=0,13,23,29 STEPS=5 INPUT_LIST=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/fulltoken_moving_preflight_case.txt OUTPUT_ROOT=/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/preflight bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_fulltoken_moving_capture.sh
# Full pilot:
# MODEL=wan_lora GPU=3 bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_fulltoken_moving_capture.sh
# MODEL=xssc GPU=3 bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_fulltoken_moving_capture.sh
# MODEL=physrvg GPU=3 bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_fulltoken_moving_capture.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
TRAIN0419_ROOT=/home/gaoya/Code_Video/Code_data/Code_train/train_0419
PHYSRVG_ROOT="${PROJECT_ROOT}/code_phys_papers_compare/PhysRVG-main"
WAN_PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
PHYSRVG_PYTHON=/data/gaoya/miniconda3/envs/vjepa2/bin/python

MODEL="${MODEL:?set MODEL to wan_lora, xssc, or physrvg}"
GPU="${GPU:-3}"
SEED="${SEED:-851}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/capture/${MODEL}}"
INPUT_LIST="${INPUT_LIST:-${SCRIPT_DIR}/fulltoken_moving_pilot_cases.txt}"
BLOCKS="${BLOCKS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29}"
STEPS="${STEPS:-5,15,25,35}"
QUERY_CHUNK="${QUERY_CHUNK:-64}"
QUERY_MAP="${QUERY_MAP:-/data/gaoya/agent-data/outputs/wan_dit_paired_query_50seeds/query_maps/${MODEL}/seed-000851/query_map.json}"

WAN_ROOT=/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B
WAN_LORA_ROOT=/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500
XSSC_WEIGHTS_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/offcial_xSSC/train_xssc_context_slots/checkpoints/step-001500
XSSC_ROOT=/home/gaoya/Code_Video/xSSC-main
XSSC_CONFIG="${XSSC_ROOT}/config-randsfq/rsfq2_r-ytvis.py"
XSSC_CHECKPOINT=/data/gaoya/ckpt/xSSC/rsfq2_r-ytvis/42-0130.pth
MODEL_ID=/data/gaoya/ckpt/HappyP4nda-PhysRVG/Wan2.2-TI2V-5B-Diffusers
DIT_CHECKPOINT=/data/gaoya/ckpt/HappyP4nda-PhysRVG/dit/diffusion_pytorch_model.safetensors
LORA_CHECKPOINT=/data/gaoya/ckpt/HappyP4nda-PhysRVG/lora/checkpoint
NEGATIVE_PROMPT="模糊，低质量，变形，伪影，文字，水印，过曝，欠曝，颜色异常，几何扭曲，物体融化，物理不合理"

mkdir -p "${OUTPUT_ROOT}/logs"
test -s "${INPUT_LIST}"
test -s "${QUERY_MAP}"
ATTENTION_ARGS=(
  --attention-output-root "${OUTPUT_ROOT}"
  --attention-blocks "${BLOCKS}"
  --attention-steps "${STEPS}"
  --attention-query-map "${QUERY_MAP}"
  --attention-query-chunk "${QUERY_CHUNK}"
)
if [[ -n "${CASE_FILTER:-}" ]]; then
  ATTENTION_ARGS+=(--attention-case-filter "${CASE_FILTER}")
fi

if [[ "${MODEL}" == "wan_lora" ]]; then
  env PYTHONNOUSERSITE=1 \
    PYTHONPATH="${PROJECT_ROOT}:${DIFFSYNTH_ROOT}:${TRAIN0419_ROOT}" \
    CUDA_VISIBLE_DEVICES="${GPU}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${WAN_PYTHON}" "${SCRIPT_DIR}/capture_fulltoken_moving_wan_lora.py" \
    "${ATTENTION_ARGS[@]}" --weights-root "${WAN_LORA_ROOT}" \
    --input-json-list-path "${INPUT_LIST}" \
    --model-name wan_lora_fulltoken_moving --wan-root "${WAN_ROOT}" \
    --output-root "${OUTPUT_ROOT}/generated/wan_lora" \
    --runtime-root "${OUTPUT_ROOT}/generated/wan_lora/_runtime" \
    --device cuda --height 512 --width 896 --num-frames 49 \
    --context-frames 8 --conditioning-mode context_aware \
    --context-resize-mode crop --num-inference-steps 40 --cfg-scale 5.0 \
    --fps 30 --seed "${SEED}" --negative-prompt "${NEGATIVE_PROMPT}" --overwrite
elif [[ "${MODEL}" == "xssc" ]]; then
  env PYTHONNOUSERSITE=1 \
    PYTHONPATH="${PROJECT_ROOT}:${DIFFSYNTH_ROOT}:${TRAIN0419_ROOT}" \
    CUDA_VISIBLE_DEVICES="${GPU}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    XSSC_ROOT="${XSSC_ROOT}" XSSC_CONFIG="${XSSC_CONFIG}" \
    XSSC_CHECKPOINT="${XSSC_CHECKPOINT}" \
    "${WAN_PYTHON}" "${SCRIPT_DIR}/capture_fulltoken_moving_xssc.py" \
    "${ATTENTION_ARGS[@]}" --weights-root "${XSSC_WEIGHTS_ROOT}" \
    --input-json-list-path "${INPUT_LIST}" \
    --model-name xssc_fulltoken_moving \
    --output-root "${OUTPUT_ROOT}/generated/xssc" \
    --step-output-dir-name results --wan-root "${WAN_ROOT}" \
    --lora-checkpoint "${WAN_LORA_ROOT}/checkpoint.safetensors" \
    --device cuda:0 --aux-device cuda:0 --inference-devices cuda:0,cuda:0 \
    --height 512 --width 896 --num-frames 49 --context-frames 8 \
    --sampling-mode prefix --num-inference-steps 40 --cfg-scale 5.0 \
    --fps 30 --seed "${SEED}" --negative-prompt "${NEGATIVE_PROMPT}" --overwrite
elif [[ "${MODEL}" == "physrvg" ]]; then
  env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="${GPU}" \
    PYTHONPATH="${PHYSRVG_ROOT}:${SCRIPT_DIR}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PHYSRVG_PYTHON}" "${SCRIPT_DIR}/capture_fulltoken_moving_physrvg.py" \
    "${ATTENTION_ARGS[@]}" --physrvg-root "${PHYSRVG_ROOT}" \
    --input-json-list-paths "${INPUT_LIST}" \
    --output-root "${OUTPUT_ROOT}/generated/physrvg" \
    --model-id "${MODEL_ID}" --dit-checkpoint "${DIT_CHECKPOINT}" \
    --lora-checkpoint "${LORA_CHECKPOINT}" --device cuda:0 \
    --height 512 --width 896 --num-frames 49 --fps 30 \
    --num-inference-steps 40 --guidance-scale 5.0 --seed "${SEED}" --force
else
  echo "unsupported MODEL=${MODEL}" >&2
  exit 2
fi
