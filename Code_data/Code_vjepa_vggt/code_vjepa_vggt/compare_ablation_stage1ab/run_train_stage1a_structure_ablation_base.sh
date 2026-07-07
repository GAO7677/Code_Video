#!/usr/bin/env bash
set -euo pipefail

GPU_SET="${GPU_SET:-0,2,3,5}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
STRUCTURE_ABLATION_TYPE="${STRUCTURE_ABLATION_TYPE:-wo_jepa}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-vjepa_vggt_wan_stage1ab}"
ABLATION_TAG="${ABLATION_TAG:-stage1a_${STRUCTURE_ABLATION_TYPE}}"
WANDB_NAME="${WANDB_NAME:-${ABLATION_TAG}}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train0705_ablation_stage1ab/${ABLATION_TAG}}"

if [[ ",${GPU_SET}," == *",4,"* ]]; then
  echo "ERROR: gpu4 故障, 禁止使用。当前 GPU_SET=${GPU_SET}" >&2
  exit 1
fi

ACCELERATE_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/accelerate
PROJ=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
CONFIG="${PROJ}/code_vjepa_vggt/compare_ablation_stage1ab/config_stage1a_full_token_structure_ablation.yaml"

mkdir -p "${OUTPUT_DIR}"

CMD=(
  env
  PYTHONPATH="${PROJ}"
  CUDA_VISIBLE_DEVICES="${GPU_SET}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
)

if [[ "${NUM_PROCESSES}" == "1" ]]; then
  CMD+=("${ACCELERATE_BIN}" launch --num_processes 1 --num_machines 1 --mixed_precision bf16)
else
  CMD+=("${ACCELERATE_BIN}" launch --multi_gpu --num_processes "${NUM_PROCESSES}" --num_machines 1 --mixed_precision bf16)
fi

CMD+=(
  -m
  code_vjepa_vggt.compare_ablation_stage1ab.train_stage1a_full_token_structure_ablation
  --config "${CONFIG}"
  --structure_ablation_type "${STRUCTURE_ABLATION_TYPE}"
  --output_dir "${OUTPUT_DIR}"
  --experiment_name "${ABLATION_TAG}"
  --wandb_project "${WANDB_PROJECT}"
  --wandb_run_name "${WANDB_NAME}"
)

CMD+=("$@")

echo "[启动] stage1a structure_ablation=${STRUCTURE_ABLATION_TYPE} GPU_SET=${GPU_SET} NUM_PROCESSES=${NUM_PROCESSES}"
echo "[启动] 输出=${OUTPUT_DIR}"
echo "[启动] 命令: ${CMD[*]}"
exec "${CMD[@]}"
