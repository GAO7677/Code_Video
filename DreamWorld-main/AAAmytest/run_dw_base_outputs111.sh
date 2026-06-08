#!/usr/bin/env bash

set -euo pipefail

GPU_ID="${1:-1}"
OUTDIR="/home/gaoya/Code_Video/DreamWorld-main/outputs/111"
LOGDIR="${OUTDIR}/logs"
mkdir -p "${OUTDIR}" "${LOGDIR}"

export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/home/gaoya/Code_Video/DreamWorld-main:${PYTHONPATH:-}
export WANDB_MODE=disabled
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export FINETRAINERS_LOG_LEVEL=DEBUG

source /home/gaoya/miniconda3/etc/profile.d/conda.sh
conda activate wan-cu128

cd /home/gaoya/Code_Video/DreamWorld-main

run_case() {
  local output_name="$1"
  local dataset_file="$2"
  local port="$3"
  local final_file="${OUTDIR}/${output_name}.mp4"

  rm -f "${final_file}"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_backend c10d \
    --rdzv_endpoint "localhost:${port}" \
    /home/gaoya/Code_Video/DreamWorld-main/script/inference/inference.py \
    --parallel_backend ptd \
    --pp_degree 1 \
    --dp_degree 1 \
    --dp_shards 1 \
    --cp_degree 1 \
    --tp_degree 1 \
    --model_name wan \
    --pretrained_model_name_or_path /data/gaoya/ckpt/Wan-AI-Wan2.1-T2V-1.3B-Diffusers \
    --enable_slicing \
    --enable_tiling \
    --dino_in_channels 8 \
    --dino_out_channels 8 \
    --vggt_in_channels 8 \
    --vggt_out_channels 8 \
    --flow_in_channels 16 \
    --flow_out_channels 16 \
    --inference_type text_to_video \
    --dataset_file "${dataset_file}" \
    --attn_provider native \
    --allow_tf32 \
    --float32_matmul_precision high \
    --seed 42 \
    --tracker_name dreamworld-inference \
    --output_dir "${OUTDIR}" \
    --init_timeout 600 \
    --nccl_timeout 600 \
    --report_to none

  local latest
  latest="$(find "${OUTDIR}" -maxdepth 1 -type f -name "inference-0-2-${output_name}-*.mp4" -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)"
  if [[ -z "${latest}" ]]; then
    echo "Failed to find generated file for ${output_name}" >&2
    exit 1
  fi
  mv -f "${latest}" "${final_file}"
}

run_case \
  "wan21_base_dreamworldentry_ball_block_seed42_832x480_81f_50s" \
  "/home/gaoya/Code_Video/DreamWorld-main/script/inference/inference_ball_block_base_official_480p.json" \
  "19347"

run_case \
  "wan21_base_dreamworldentry_corgi_seed42_832x480_81f_50s" \
  "/home/gaoya/Code_Video/DreamWorld-main/script/inference/inference_corgi_base_official_480p.json" \
  "19348"
