#!/usr/bin/env bash

set -euo pipefail

GPU_ID="${1:-3}"
OUTDIR="/home/gaoya/Code_Video/DreamWorld-main/outputs/111"
LOGDIR="${OUTDIR}/logs"
mkdir -p "${OUTDIR}" "${LOGDIR}"

source /home/gaoya/miniconda3/etc/profile.d/conda.sh
conda activate wan-cu128

cd /home/gaoya/Code_Video/WAN_2p2/Wan2.1-main

CUDA_VISIBLE_DEVICES="${GPU_ID}" python generate.py \
  --task t2v-1.3B \
  --size "832*480" \
  --frame_num 81 \
  --ckpt_dir /data/gaoya/ckpt/Wan-AI-Wan2.1-T2V-1.3B \
  --offload_model True \
  --t5_cpu \
  --sample_steps 50 \
  --sample_shift 8 \
  --sample_guide_scale 6 \
  --base_seed 42 \
  --prompt "A red ball rolls forward and collides with a small wooden block on a tabletop. The block slides from the impact. Realistic motion, stable lighting, natural colors." \
  --save_file "${OUTDIR}/wan21_officialdemo_ball_block_seed42_832x480_81f_50s.mp4"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python generate.py \
  --task t2v-1.3B \
  --size "832*480" \
  --frame_num 81 \
  --ckpt_dir /data/gaoya/ckpt/Wan-AI-Wan2.1-T2V-1.3B \
  --offload_model True \
  --t5_cpu \
  --sample_steps 50 \
  --sample_shift 8 \
  --sample_guide_scale 6 \
  --base_seed 42 \
  --prompt "A small corgi runs across a sunny grass field, realistic motion, natural lighting." \
  --save_file "${OUTDIR}/wan21_officialdemo_corgi_seed42_832x480_81f_50s.mp4"
