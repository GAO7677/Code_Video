#!/usr/bin/env bash
set -euo pipefail

# Run one model per invocation:
# MODEL=wan_lora GPU=3 bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_allblock_ball_query_case001460.sh
# MODEL=xssc GPU=4 bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_allblock_ball_query_case001460.sh
# MODEL=physrvg GPU=3 bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_allblock_ball_query_case001460.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
TRAIN0419_ROOT=/home/gaoya/Code_Video/Code_data/Code_train/train_0419
PHYSRVG_ROOT="${PROJECT_ROOT}/code_phys_papers_compare/PhysRVG-main"
WAN_PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
PHYSRVG_PYTHON=/data/gaoya/miniconda3/envs/vjepa2/bin/python

MODEL="${MODEL:?set MODEL to wan_lora, xssc, or physrvg}"
GPU="${GPU:?set GPU to one physical GPU id}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/agent-data/outputs/wan_dit_ball_query_attention/case001460_frame08_allblocks}"
INPUT_LIST="${SCRIPT_DIR}/ball_query_case001460.txt"
CASE=0613pybullet_sample_001460_w002
BLOCKS="${BLOCKS:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29}"
STEPS="${STEPS:-5,15,25,35}"
QUERY_COORDS="2:6:13,2:6:14,2:7:13,2:7:14"

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

BASELINE_ROOT=/data/gaoya/agent-data/outputs/wan_dit_block17_self_attention/test5_first5/generated
case "${MODEL}" in
  wan_lora)
    BASELINE_VIDEO="${BASELINE_ROOT}/wan_lora/${CASE}.mp4"
    ;;
  xssc)
    BASELINE_VIDEO="${BASELINE_ROOT}/xssc/results/${CASE}.mp4"
    ;;
  physrvg)
    BASELINE_VIDEO="${BASELINE_ROOT}/physrvg/input_first5_unique/physRVG_steps40_512x896_08_49f/${CASE}.mp4"
    ;;
  *)
    echo "unsupported MODEL=${MODEL}" >&2
    exit 2
    ;;
esac

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/query_previews"
test -s "${INPUT_LIST}"
test -s "${BASELINE_VIDEO}"
PREVIEW="${OUTPUT_ROOT}/query_previews/${MODEL}.png"
if [[ ! -s "${PREVIEW}" ]]; then
  PYTHONNOUSERSITE=1 "${WAN_PYTHON}" "${SCRIPT_DIR}/prepare_ball_query_preview.py" \
    --video "${BASELINE_VIDEO}" \
    --frame 8 \
    --query-coords "${QUERY_COORDS}" \
    --label "${MODEL}" \
    --output "${PREVIEW}"
fi

ATTENTION_ARGS=(
  --attention-output-root "${OUTPUT_ROOT}"
  --attention-blocks "${BLOCKS}"
  --attention-steps "${STEPS}"
  --attention-query-coords "${QUERY_COORDS}"
  --attention-query-video-frame 8
  --attention-query-preview "${PREVIEW}"
)

if [[ "${MODEL}" == "wan_lora" ]]; then
  env \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH="${PROJECT_ROOT}:${DIFFSYNTH_ROOT}:${TRAIN0419_ROOT}" \
    CUDA_VISIBLE_DEVICES="${GPU}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${WAN_PYTHON}" "${SCRIPT_DIR}/capture_allblocks_wan_lora_ball_query.py" \
    "${ATTENTION_ARGS[@]}" \
    --weights-root "${WAN_LORA_ROOT}" \
    --input-json-list-path "${INPUT_LIST}" \
    --model-name wan_lora_allblock_ball_query_attention \
    --wan-root "${WAN_ROOT}" \
    --output-root "${OUTPUT_ROOT}/generated/wan_lora" \
    --runtime-root "${OUTPUT_ROOT}/generated/wan_lora/_runtime" \
    --device cuda --height 512 --width 896 --num-frames 49 \
    --context-frames 8 --conditioning-mode context_aware \
    --context-resize-mode crop --num-inference-steps 40 --cfg-scale 5.0 \
    --fps 30 --seed 42 --negative-prompt "${NEGATIVE_PROMPT}" --overwrite
elif [[ "${MODEL}" == "xssc" ]]; then
  env \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH="${PROJECT_ROOT}:${DIFFSYNTH_ROOT}:${TRAIN0419_ROOT}" \
    CUDA_VISIBLE_DEVICES="${GPU}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    XSSC_ROOT="${XSSC_ROOT}" XSSC_CONFIG="${XSSC_CONFIG}" \
    XSSC_CHECKPOINT="${XSSC_CHECKPOINT}" \
    "${WAN_PYTHON}" "${SCRIPT_DIR}/capture_allblocks_xssc_ball_query.py" \
    "${ATTENTION_ARGS[@]}" \
    --weights-root "${XSSC_WEIGHTS_ROOT}" \
    --input-json-list-path "${INPUT_LIST}" \
    --model-name xssc_allblock_ball_query_attention \
    --output-root "${OUTPUT_ROOT}/generated/xssc" \
    --step-output-dir-name results --wan-root "${WAN_ROOT}" \
    --lora-checkpoint "${WAN_LORA_ROOT}/checkpoint.safetensors" \
    --device cuda:0 --aux-device cuda:0 --inference-devices cuda:0,cuda:0 \
    --height 512 --width 896 --num-frames 49 --context-frames 8 \
    --sampling-mode prefix --num-inference-steps 40 --cfg-scale 5.0 \
    --fps 30 --seed 42 --negative-prompt "${NEGATIVE_PROMPT}" --overwrite
else
  env \
    PYTHONNOUSERSITE=1 \
    CUDA_VISIBLE_DEVICES="${GPU}" \
    PYTHONPATH="${PHYSRVG_ROOT}:${SCRIPT_DIR}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PHYSRVG_PYTHON}" "${SCRIPT_DIR}/capture_allblocks_physrvg_ball_query.py" \
    "${ATTENTION_ARGS[@]}" \
    --physrvg-root "${PHYSRVG_ROOT}" \
    --input-json-list-paths "${INPUT_LIST}" \
    --output-root "${OUTPUT_ROOT}/generated/physrvg" \
    --model-id "${MODEL_ID}" --dit-checkpoint "${DIT_CHECKPOINT}" \
    --lora-checkpoint "${LORA_CHECKPOINT}" --device cuda:0 \
    --height 512 --width 896 --num-frames 49 --fps 30 \
    --num-inference-steps 40 --guidance-scale 5.0 --seed 42 --force
fi

echo "model=${MODEL} output=${OUTPUT_ROOT}"
