#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
PROJECT=${ROOT}/wan_phyco_train0716
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
ACCELERATE=/home/gaoya/miniconda3/envs/wan-cu128/bin/accelerate
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/gaoya/agent-data/checkpoints/wan_phyco_train0716/smoke_${RUN_TAG}}"
TMP_ROOT="${TMP_ROOT:-/data/gaoya/agent-data/cache/wan_phyco_train0716/smoke_${RUN_TAG}}"
VISIBLE_GPU_IDS="${VISIBLE_GPU_IDS:-6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
mkdir -p "${OUTPUT_DIR}" "${TMP_ROOT}" /data/gaoya/agent-data/cache/wandb

env \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="${ROOT}:${DIFFSYNTH_ROOT}" \
  CUDA_VISIBLE_DEVICES="${VISIBLE_GPU_IDS}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  TMPDIR="${TMP_ROOT}" TMP="${TMP_ROOT}" TEMP="${TMP_ROOT}" \
  WANDB_DIR=/data/gaoya/agent-data/cache/wandb \
  "${ACCELERATE}" launch --num_processes "${NUM_PROCESSES}" --num_machines 1 --mixed_precision bf16 \
  "${PROJECT}/train.py" \
    --diffsynth_root "${DIFFSYNTH_ROOT}" \
    --wan_root /data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B \
    --dataset_type replay_preserve_mix \
    --pybullet_raw_root /data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/raw_v1/industrial_s1_scale2_merged_h264_batch1500 \
    --pybullet_raw_split train --pybullet_raw_sampling_strategy prefix --pybullet_raw_window_starts 0 \
    --pybullet_raw_init_scan_limit 8 \
    --kubric_root /data/gaoya/dataset/nnsriram97-phyco_kubric --kubric_split train \
    --kubric_cache_root /data/gaoya/agent-data/cache/kubric_no_gt_box_dataset \
    --kubric_sampling_strategy prefix --kubric_replay_index_num_frames 69 \
    --kubric_replay_index_num_context_frames 20 --kubric_init_scan_limit 8 \
    --openvid_root /data/gaoya/dataset/mvp-lab-OpenVidHD-0.4M-720p-48fps/train --openvid_max_samples 8 \
    --mixture_pybullet_ratio 0.30 --mixture_kubric_ratio 0.30 --mixture_openvid_ratio 0.40 \
    --height 512 --width 896 --num_frames 49 \
    --fixed_num_context_frames 8 --ctx_max_length 8 \
    --min_context_frames 0 --max_context_ratio 1.0 --no_context_ratio 0.0 \
    --min_timestep_boundary 0.01 --max_timestep_boundary 1.0 \
    --trainable_models dit --extra_inputs input_image \
    --phyco_hidden_dim 128 --phyco_block_ids 3,8,13,18,23,28 --phyco_map_downsample 8 \
    --max_train_steps "${MAX_TRAIN_STEPS:-2}" --num_epochs 2 \
    --dataset_num_workers 0 --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 --weight_decay 0.01 --optimizer_type paged_adamw8bit \
    --max_grad_norm 1.0 --find_unused_parameters --fail_on_nonfinite_train_values \
    --save_steps 1 --max_checkpoints_keep 0 \
    --remove_prefix_in_ckpt pipe.dit. --output_path "${OUTPUT_DIR}" \
    --report_to wandb --wandb_project vjepa_vggt_wan \
    --wandb_name "wan_phyco_train0716_smoke_${RUN_TAG}" --wandb_mode disabled

echo "smoke output: ${OUTPUT_DIR}"

