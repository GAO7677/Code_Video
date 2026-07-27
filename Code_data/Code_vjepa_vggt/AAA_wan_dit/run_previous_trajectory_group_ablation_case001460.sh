#!/usr/bin/env bash
set -euo pipefail

# Run:
# GPU=3 bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_previous_trajectory_group_ablation_case001460.sh A
# GPU=3 bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_previous_trajectory_group_ablation_case001460.sh B

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 A|B" >&2
  exit 2
fi

GROUP="${1^^}"
case "${GROUP}" in A) EXPECTED=47 ;; B) EXPECTED=38 ;; *) exit 2 ;; esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
TRAIN0419_ROOT=/home/gaoya/Code_Video/Code_data/Code_train/train_0419
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
GPU="${GPU:-3}"

INPUT_LIST="${SCRIPT_DIR}/ball_query_case001460.txt"
TARGETS_CSV=/data/gaoya/agent-data/outputs/wan_dit_allblock_head_roles/case001460_latent_aligned_wan_lora/previous_trajectory_heads.csv
OUTPUT_BASE=/data/gaoya/agent-data/outputs/wan_dit_previous_trajectory_group_ablation/case001460
OUTPUT_ROOT="${OUTPUT_BASE}/prev_${GROUP,,}"
WAN_ROOT=/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B
WAN_LORA_ROOT=/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500
NEGATIVE_PROMPT="模糊，低质量，变形，伪影，文字，水印，过曝，欠曝，颜色异常，几何扭曲，物体融化，物理不合理"

mkdir -p "${OUTPUT_ROOT}/logs"
test -s "${TARGETS_CSV}"
{
  echo "group=${GROUP}"
  echo "expected_targets=${EXPECTED}"
  echo "targets_csv=${TARGETS_CSV}"
  echo "gpu=${GPU}"
  echo "context_frames=8"
  echo "num_inference_steps=40"
  echo "cfg_scale=5.0"
  echo "seed=42"
} > "${OUTPUT_ROOT}/ablation_config.txt"

env \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="${PROJECT_ROOT}:${DIFFSYNTH_ROOT}:${TRAIN0419_ROOT}:${SCRIPT_DIR}" \
  CUDA_VISIBLE_DEVICES="${GPU}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PYTHON}" "${SCRIPT_DIR}/infer_wan_lora_grouped_head_ablation.py" \
  --grouped-head-targets-csv "${TARGETS_CSV}" \
  --grouped-head-object "${GROUP}" \
  --expected-target-count "${EXPECTED}" \
  --weights-root "${WAN_LORA_ROOT}" \
  --input-json-list-path "${INPUT_LIST}" \
  --model-name "wan_lora_prev_${GROUP,,}_heads_zero" \
  --wan-root "${WAN_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --runtime-root "${OUTPUT_ROOT}/_runtime" \
  --device cuda --height 512 --width 896 --num-frames 49 \
  --context-frames 8 --conditioning-mode context_aware \
  --context-resize-mode crop --num-inference-steps 40 --cfg-scale 5.0 \
  --fps 30 --seed 42 --negative-prompt "${NEGATIVE_PROMPT}" --overwrite \
  2>&1 | tee "${OUTPUT_ROOT}/logs/run.log"
