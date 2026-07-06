#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-3}"

if [ "$GPU" = "4" ]; then
  echo "ERROR: gpu4 故障, 禁止使用。请指定其他 GPU。" >&2
  exit 1
fi

ACCELERATE_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/accelerate
TRAIN_SCRIPT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0706_wan1p3b/train_v_newtrain.py
DATASET_CONFIG=${DATASET_CONFIG:-/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0706_wan1p3b/dataset_mix_config.json}
WAN_ROOT=/data/gaoya/ckpt/Wan-AI-Wan2.1-T2V-1.3B
OUTPUT_DIR=${OUTPUT_DIR:-/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints_wan21_13b/smoke/openvid_mixed_ctx81_384x672_lora}
USE_SAMPLE_FULL_VIDEO_LENGTH=${USE_SAMPLE_FULL_VIDEO_LENGTH:-1}
TRAIN_NUM_FRAMES=${TRAIN_NUM_FRAMES:-81}
SAMPLE_FULL_VIDEO_MAX_FRAMES=${SAMPLE_FULL_VIDEO_MAX_FRAMES:-81}

mkdir -p "${OUTPUT_DIR}"

CMD=(
  env
  CUDA_VISIBLE_DEVICES="${GPU}"
  "${ACCELERATE_BIN}" launch --num_processes 1 --num_machines 1 "${TRAIN_SCRIPT}" \
  --diffsynth_root /home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
  --wan_root "${WAN_ROOT}" \
  --dataset_base_path "${DATASET_CONFIG}" \
  --dataset_metadata_path "" \
  --height 384 \
  --width 672 \
  --num_frames "${TRAIN_NUM_FRAMES}" \
  --use_sample_full_video_length \
  --max_train_steps 2 \
  --context_sampling_profile mixed_modes \
  --min_context_frames 1 \
  --max_context_ratio 0.5 \
  --context_reference_frames 49 \
  --context_reference_prefixes 1,4,8,12,16 \
  --prefix_context_ratio 0.55 \
  --first_frame_context_ratio 0.20 \
  --sparse_context_ratio 0.15 \
  --random_context_ratio 0.05 \
  --no_context_ratio 0.05 \
  --dataset_repeat 1 \
  --dataset_num_workers 0 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --num_epochs 1 \
  --gradient_accumulation_steps 1 \
  --save_steps 1 \
  --max_checkpoints_keep 3 \
  --remove_prefix_in_ckpt pipe.dit. \
  --output_path "${OUTPUT_DIR}" \
  --lora_base_model dit \
  --lora_target_modules q,k,v,o,ffn.0,ffn.2 \
  --lora_rank 32 \
  --report_to wandb \
  --wandb_project openvid-movid-genesis-wan21_13b \
  --wandb_mode online
)

if [[ "${USE_SAMPLE_FULL_VIDEO_LENGTH}" != "1" ]]; then
  FILTERED_CMD=()
  for arg in "${CMD[@]}"; do
    if [[ "${arg}" == "--use_sample_full_video_length" ]]; then
      continue
    fi
    FILTERED_CMD+=("${arg}")
  done
  CMD=("${FILTERED_CMD[@]}")
fi

if [[ -n "${SAMPLE_FULL_VIDEO_MAX_FRAMES}" ]]; then
  CMD+=(--sample_full_video_max_frames "${SAMPLE_FULL_VIDEO_MAX_FRAMES}")
fi

exec "${CMD[@]}"
