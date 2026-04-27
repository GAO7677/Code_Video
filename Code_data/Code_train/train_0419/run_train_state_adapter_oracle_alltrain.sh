#!/usr/bin/env bash
set -euo pipefail

source /home/gaoya/miniconda3/etc/profile.d/conda.sh
conda activate wan

ACCELERATE_BIN=/data/gaoya/miniconda3/envs/wan/bin/accelerate

DIFFSYNTH_ROOT=/home/gaoya/Code_Video/DiffSynth-Studio-main
TRAIN_ROOT=/home/gaoya/Code_Video/Code_data/Code_train/train_0419
STATE_ADAPTER_ROOT=${TRAIN_ROOT}/state_adapter
WAN_ROOT=/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B

RAW_DATASET_ROOT=/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases
WINDOW_ROOT=${RAW_DATASET_ROOT}/preprocess_v1/oracle_wan_ctx8_fut5_9_13_alltrain

TV2V_ROOT=/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_ctx49_736x1280_lora
OUTPUT_ROOT=/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/oracle_state_adapter_ctx8_fut5_9_13_alltrain

mkdir -p "${OUTPUT_ROOT}"

if [ ! -f "${WINDOW_ROOT}/manifest.json" ]; then
  python3 ${STATE_ADAPTER_ROOT}/build_stage1_oracle_windows.py \
    --dataset_root ${RAW_DATASET_ROOT} \
    --out_root ${WINDOW_ROOT} \
    --count_buckets count_01,count_02,count_03_04 \
    --context_len 8 \
    --future_lengths 5,9,13 \
    --contact_mode none \
    --future_main_visibility_threshold 0.5
fi

CUDA_VISIBLE_DEVICES=2,3 "${ACCELERATE_BIN}" launch \
  --multi_gpu \
  --num_processes 2 \
  --num_machines 1 \
  ${STATE_ADAPTER_ROOT}/train_state_adapter.py \
  --diffsynth_root ${DIFFSYNTH_ROOT} \
  --wan_root ${WAN_ROOT} \
  --dataset_root ${WINDOW_ROOT} \
  --preset_tv2v_root ${TV2V_ROOT} \
  --height 736 \
  --width 1280 \
  --dataset_repeat 1 \
  --dataset_num_workers 4 \
  --learning_rate 2e-4 \
  --weight_decay 0.01 \
  --num_epochs 10 \
  --gradient_accumulation_steps 1 \
  --save_steps 1000 \
  --output_path ${OUTPUT_ROOT} \
  --report_to wandb \
  --wandb_project wan22-oracle-state-adapter \
  --wandb_mode online
