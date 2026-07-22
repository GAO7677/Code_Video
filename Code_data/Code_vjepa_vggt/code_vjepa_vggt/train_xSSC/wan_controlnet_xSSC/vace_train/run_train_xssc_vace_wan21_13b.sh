#!/usr/bin/env bash
set -euo pipefail

export DIFFSYNTH_ROOT="${DIFFSYNTH_ROOT:-/home/gaoya/Code_Video/DiffSynth-Studio-main}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_xssc_vace_condition.py"

DATASET_BASE_PATH="${DATASET_BASE_PATH:-data/diffsynth_example_dataset/wanvideo/Wan2.1-VACE-1.3B}"
DATASET_METADATA_PATH="${DATASET_METADATA_PATH:-${DATASET_BASE_PATH}/metadata.csv}"
OUTPUT_PATH="${OUTPUT_PATH:-/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/wan_vace_xssc_condition/wan21_13b_ctx9}"

XSSC_ROOT="${XSSC_ROOT:-/home/gaoya/Code_Video/xSSC-main}"
XSSC_CONFIG="${XSSC_CONFIG:-${XSSC_ROOT}/config-randsfq/rsfq2_r-ytvis.py}"
XSSC_CHECKPOINT="${XSSC_CHECKPOINT:-/data/gaoya/ckpt/xSSC/rsfq2_r-ytvis/42-0130.pth}"

accelerate launch "${TRAIN_SCRIPT}" \
  --dataset_base_path "${DATASET_BASE_PATH}" \
  --dataset_metadata_path "${DATASET_METADATA_PATH}" \
  --data_file_keys "video" \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat "${DATASET_REPEAT:-100}" \
  --model_id_with_origin_paths "${MODEL_ID_WITH_ORIGIN_PATHS:-Wan-AI/Wan2.1-VACE-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-1.3B:Wan2.1_VAE.pth}" \
  --learning_rate "${LEARNING_RATE:-5e-5}" \
  --num_epochs "${NUM_EPOCHS:-2}" \
  --save_steps "${SAVE_STEPS:-500}" \
  --output_path "${OUTPUT_PATH}" \
  --trainable_models "vace" \
  --xssc_root "${XSSC_ROOT}" \
  --xssc_config "${XSSC_CONFIG}" \
  --xssc_checkpoint "${XSSC_CHECKPOINT}" \
  --xssc_condition_frames "${XSSC_CONDITION_FRAMES:-9}" \
  --xssc_reference_frames "${XSSC_REFERENCE_FRAMES:-9}" \
  --xssc_input_size "${XSSC_INPUT_SIZE:-256}" \
  --xssc_vae_temporal_stride "${XSSC_VAE_TEMPORAL_STRIDE:-4}" \
  --xssc_slot_dropout "${XSSC_SLOT_DROPOUT:-0.0}" \
  --xssc_query_dim "${XSSC_QUERY_DIM:-256}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --use_gradient_checkpointing_offload
