#!/usr/bin/env bash
set -euo pipefail

# Run:
# GPU_WAN=0 GPU_XSSC=1 GPU_PHYRVG=2 \
# ATTENTION_BLOCK=17 \
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_ball_query_attention_case001460.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
TRAIN0419_ROOT=/home/gaoya/Code_Video/Code_data/Code_train/train_0419
PHYSRVG_ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_phys_papers_compare/PhysRVG-main
WAN_PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
PHYSRVG_PYTHON=/data/gaoya/miniconda3/envs/vjepa2/bin/python

INPUT_LIST="${SCRIPT_DIR}/ball_query_case001460.txt"
CASE=0613pybullet_sample_001460_w002
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/agent-data/outputs/wan_dit_ball_query_attention/case001460_frame08}"
ATTENTION_BLOCK="${ATTENTION_BLOCK:-17}"
GPU_WAN="${GPU_WAN:-0}"
GPU_XSSC="${GPU_XSSC:-1}"
GPU_PHYRVG="${GPU_PHYRVG:-2}"
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
WAN_BASELINE="${BASELINE_ROOT}/wan_lora/${CASE}.mp4"
XSSC_BASELINE="${BASELINE_ROOT}/xssc/results/${CASE}.mp4"
PHYSRVG_BASELINE="${BASELINE_ROOT}/physrvg/input_first5_unique/physRVG_steps40_512x896_08_49f/${CASE}.mp4"

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/query_previews"
for path in "${INPUT_LIST}" "${WAN_BASELINE}" "${XSSC_BASELINE}" "${PHYSRVG_BASELINE}"; do
  test -s "${path}"
done

prepare_preview() {
  local model="$1"
  local video="$2"
  "${WAN_PYTHON}" "${SCRIPT_DIR}/prepare_ball_query_preview.py" \
    --video "${video}" \
    --frame 8 \
    --query-coords "${QUERY_COORDS}" \
    --label "${model}" \
    --output "${OUTPUT_ROOT}/query_previews/${model}.png"
}
prepare_preview wan_lora "${WAN_BASELINE}"
prepare_preview xssc "${XSSC_BASELINE}"
prepare_preview physrvg "${PHYSRVG_BASELINE}"

common_attention_args() {
  local model="$1"
  printf '%s\n' \
    --attention-output-root "${OUTPUT_ROOT}/matrices" \
    --attention-block "${ATTENTION_BLOCK}" \
    --attention-steps 5,15,25,35 \
    --attention-query-coords "${QUERY_COORDS}" \
    --attention-query-video-frame 8 \
    --attention-query-preview "${OUTPUT_ROOT}/query_previews/${model}.png"
}

mapfile -t WAN_ATTN < <(common_attention_args wan_lora)
mapfile -t XSSC_ATTN < <(common_attention_args xssc)
mapfile -t PHYSRVG_ATTN < <(common_attention_args physrvg)

run_wan_lora() {
  env \
    PYTHONPATH="${PROJECT_ROOT}:${DIFFSYNTH_ROOT}:${TRAIN0419_ROOT}" \
    CUDA_VISIBLE_DEVICES="${GPU_WAN}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${WAN_PYTHON}" "${SCRIPT_DIR}/visualize_wan_lora_ball_query_attention.py" \
    "${WAN_ATTN[@]}" \
    --weights-root "${WAN_LORA_ROOT}" \
    --input-json-list-path "${INPUT_LIST}" \
    --model-name "wan_lora_block$(printf '%02d' "${ATTENTION_BLOCK}")_ball_query_attention" \
    --wan-root "${WAN_ROOT}" \
    --output-root "${OUTPUT_ROOT}/generated/wan_lora" \
    --runtime-root "${OUTPUT_ROOT}/generated/wan_lora/_runtime" \
    --device cuda --height 512 --width 896 --num-frames 49 \
    --context-frames 8 --conditioning-mode context_aware \
    --context-resize-mode crop --num-inference-steps 40 --cfg-scale 5.0 \
    --fps 30 --seed 42 --negative-prompt "${NEGATIVE_PROMPT}" --overwrite
}

run_xssc() {
  env \
    PYTHONPATH="${PROJECT_ROOT}:${DIFFSYNTH_ROOT}:${TRAIN0419_ROOT}" \
    CUDA_VISIBLE_DEVICES="${GPU_XSSC}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    XSSC_ROOT="${XSSC_ROOT}" XSSC_CONFIG="${XSSC_CONFIG}" \
    XSSC_CHECKPOINT="${XSSC_CHECKPOINT}" XSSC_PREPROCESS_MODE=center_crop \
    XSSC_SLOT_TEMPORAL_MODE=full \
    "${WAN_PYTHON}" "${SCRIPT_DIR}/visualize_xssc_ball_query_attention.py" \
    "${XSSC_ATTN[@]}" \
    --weights-root "${XSSC_WEIGHTS_ROOT}" \
    --input-json-list-path "${INPUT_LIST}" \
    --model-name "xssc_block$(printf '%02d' "${ATTENTION_BLOCK}")_ball_query_attention" \
    --output-root "${OUTPUT_ROOT}/generated/xssc" \
    --step-output-dir-name results --wan-root "${WAN_ROOT}" \
    --lora-checkpoint "${WAN_LORA_ROOT}/checkpoint.safetensors" \
    --device cuda:0 --aux-device cuda:0 --inference-devices cuda:0,cuda:0 \
    --height 512 --width 896 --num-frames 49 --context-frames 8 \
    --sampling-mode prefix --num-inference-steps 40 --cfg-scale 5.0 \
    --fps 30 --seed 42 --negative-prompt "${NEGATIVE_PROMPT}" --overwrite
}

run_physrvg() {
  env \
    CUDA_VISIBLE_DEVICES="${GPU_PHYRVG}" \
    PYTHONPATH="${PHYSRVG_ROOT}:${SCRIPT_DIR}" \
    PYTHONNOUSERSITE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PHYSRVG_PYTHON}" "${SCRIPT_DIR}/visualize_physrvg_ball_query_attention.py" \
    "${PHYSRVG_ATTN[@]}" --physrvg-root "${PHYSRVG_ROOT}" \
    --input-json-list-paths "${INPUT_LIST}" \
    --output-root "${OUTPUT_ROOT}/generated/physrvg" \
    --model-id "${MODEL_ID}" --dit-checkpoint "${DIT_CHECKPOINT}" \
    --lora-checkpoint "${LORA_CHECKPOINT}" --device cuda:0 \
    --height 512 --width 896 --num-frames 49 --fps 30 \
    --num-inference-steps 40 --guidance-scale 5.0 --seed 42 --force
}

run_wan_lora > "${OUTPUT_ROOT}/logs/wan_lora.log" 2>&1 &
pid_wan=$!
run_xssc > "${OUTPUT_ROOT}/logs/xssc.log" 2>&1 &
pid_xssc=$!
run_physrvg > "${OUTPUT_ROOT}/logs/physrvg.log" 2>&1 &
pid_physrvg=$!

status=0
for pid in "${pid_wan}" "${pid_xssc}" "${pid_physrvg}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if (( status != 0 )); then
  echo "At least one ball-query attention job failed; inspect ${OUTPUT_ROOT}/logs" >&2
  exit "${status}"
fi

"${WAN_PYTHON}" "${SCRIPT_DIR}/build_ball_query_attention_gallery.py" \
  --root "${OUTPUT_ROOT}/matrices" \
  --case "${CASE}" \
  --output "${OUTPUT_ROOT}/_gallery"
echo "gallery=${OUTPUT_ROOT}/_gallery/index.html"
