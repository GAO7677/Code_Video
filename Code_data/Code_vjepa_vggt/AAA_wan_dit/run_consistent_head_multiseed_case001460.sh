#!/usr/bin/env bash
set -euo pipefail

# Run:
# GPU=3 bash run_consistent_head_multiseed_case001460.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
TRAIN0419_ROOT=/home/gaoya/Code_Video/Code_data/Code_train/train_0419
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
GPU="${GPU:-3}"

NEW_ROOT=/data/gaoya/agent-data/outputs/wan_dit_consistent_category_head_ablation/case001460_wan_lora_multiseed
LEGACY_ROOT=/data/gaoya/agent-data/outputs/wan_dit_consistent_category_head_ablation/case001460_wan_lora
LEGACY_BASELINE=/data/gaoya/agent-data/outputs/wan_dit_ball_query_attention/case001460_frame08/generated/wan_lora/0613pybullet_sample_001460_w002.mp4
CLASSIFICATION_METADATA=/data/gaoya/agent-data/outputs/wan_dit_allblock_head_roles/case001460_latent_aligned_wan_lora/metadata.json
INPUT_LIST="${SCRIPT_DIR}/ball_query_case001460.txt"
SOURCE_JSON=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/0613pybullet_sample_001460_w002.json
WAN_ROOT=/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B
WEIGHTS_ROOT=/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500
GALLERY_OUTPUT=/data/gaoya/agent-data/outputs/wan_dit_allblock_head_roles/case001460_consistent_category_head_zero_wan_lora_multiseed
NEGATIVE_PROMPT="模糊，低质量，变形，伪影，文字，水印，过曝，欠曝，颜色异常，几何扭曲，物体融化，物理不合理"
LOG="${NEW_ROOT}/multiseed.log"

mkdir -p "${NEW_ROOT}"
env \
  PYTHONPATH="${PROJECT_ROOT}:${DIFFSYNTH_ROOT}:${TRAIN0419_ROOT}" \
  CUDA_VISIBLE_DEVICES="${GPU}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONUNBUFFERED=1 \
  TOKENIZERS_PARALLELISM=false \
  "${PYTHON}" "${SCRIPT_DIR}/infer_wan_lora_consistent_head_multiseed.py" \
  --seeds 43 44 45 46 \
  --output-root "${NEW_ROOT}" \
  --classification-metadata "${CLASSIFICATION_METADATA}" \
  --weights-root "${WEIGHTS_ROOT}" \
  --input-json-list-path "${INPUT_LIST}" \
  --wan-root "${WAN_ROOT}" \
  --device cuda --height 512 --width 896 --num-frames 49 \
  --context-frames 8 --num-inference-steps 40 --cfg-scale 5.0 \
  --fps 30 --quality 5 --negative-prompt "${NEGATIVE_PROMPT}" \
  2>&1 | tee "${LOG}"

"${PYTHON}" "${SCRIPT_DIR}/verify_consistent_head_multiseed.py" \
  --new-root "${NEW_ROOT}" \
  --legacy-ablation-root "${LEGACY_ROOT}" \
  --legacy-baseline-video "${LEGACY_BASELINE}" \
  --classification-metadata "${CLASSIFICATION_METADATA}" \
  --seeds 42 43 44 45 46 \
  --output "${NEW_ROOT}/verification_report.json"

PYTHONPATH="${SCRIPT_DIR}" \
"${PYTHON}" "${SCRIPT_DIR}/build_consistent_head_multiseed_gallery.py" \
  --new-root "${NEW_ROOT}" \
  --legacy-ablation-root "${LEGACY_ROOT}" \
  --legacy-baseline-video "${LEGACY_BASELINE}" \
  --source-json "${SOURCE_JSON}" \
  --verification-report "${NEW_ROOT}/verification_report.json" \
  --output "${GALLERY_OUTPUT}" \
  --seeds 42 43 44 45 46

echo "gallery=${GALLERY_OUTPUT}/index.html"
