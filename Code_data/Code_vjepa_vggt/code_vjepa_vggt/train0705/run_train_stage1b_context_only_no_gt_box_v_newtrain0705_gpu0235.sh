#!/usr/bin/env bash
# =============================================================================
# Stage1B context-only *no-GT-box* 多卡训练启动脚本 (train0705, gpu0235)
#
# 说明:
#   - 基于 train0705/run_train_stage1b_context_only_no_gt_box_v_newtrain0705.sh
#   - 训练框架保持为 WAN_2p2/DiffSynth-Studio-main/diffsynth
#   - 冻结模块 / 可训练模块定义不变，只调整为 accelerate 多卡启动
#   - 使用物理 GPU 0,2,3,5；明确禁用 gpu4
#   - 每 500 step 保存一次权重
#   - 前台运行，不使用 nohup / 后台
# =============================================================================
set -euo pipefail

GPU_SET="${GPU_SET:-0,2,3,5}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
RESUME="${RESUME:-none}"
NUM_FRAMES="${NUM_FRAMES:-24}"
FIXED_NUM_CONTEXT_FRAMES="${FIXED_NUM_CONTEXT_FRAMES:-8}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-20000}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-896}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-vjepa_vggt_wan}"
WANDB_NAME="${WANDB_NAME:-pybullet0629_teacher_student_stage1b_context_only_no_gt_box_v_newtrain0705_gpu0235_20260703}"

if [[ ",${GPU_SET}," == *",4,"* ]]; then
  echo "ERROR: gpu4 故障, 禁止使用。当前 GPU_SET=${GPU_SET}" >&2
  exit 1
fi

ACCELERATE_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/accelerate
PROJ=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
TRAIN_SCRIPT="${PROJ}/code_vjepa_vggt/train0705/train_stage1b_context_only_no_gt_box_v_newtrain.py"

WAN_ROOT=/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B
BASE_LORA=/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors
STAGE1A_CKPT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt
DATASET_ROOT=/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/episodes_v1/industrial_s1_scale2_256x144_s8_f16_n6_h264_batch1500
OUTPUT_DIR="${OUTPUT_DIR:-/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703}"
DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-0}"
GROUNDING_GT_BOX_QUERY_REPAIR="${GROUNDING_GT_BOX_QUERY_REPAIR:-0}"
GROUNDING_GT_BOX_OVERSAMPLE_FACTOR="${GROUNDING_GT_BOX_OVERSAMPLE_FACTOR:-4}"
GROUNDING_GT_BOX_MIN_VISIBLE_RATIO="${GROUNDING_GT_BOX_MIN_VISIBLE_RATIO:-0.60}"
GROUNDING_GT_BOX_MIN_IN_BOX_RATIO="${GROUNDING_GT_BOX_MIN_IN_BOX_RATIO:-0.60}"

mkdir -p "${OUTPUT_DIR}"

RESUME_ARGS=()
if [ "${RESUME}" != "none" ]; then
  RESUME_ARGS=(--stage2_resume_from "${RESUME}")
  echo "[resume] 断点续训: ${RESUME}"
else
  echo "[fresh] 从头开始训练"
fi

CMD=(
  env
  PYTHONPATH="${PROJ}:${DIFFSYNTH_ROOT}"
  CUDA_VISIBLE_DEVICES="${GPU_SET}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  "${ACCELERATE_BIN}" launch --multi_gpu --num_processes "${NUM_PROCESSES}" --num_machines 1 --mixed_precision bf16
  "${TRAIN_SCRIPT}"
  --diffsynth_root "${DIFFSYNTH_ROOT}"
  --wan_root "${WAN_ROOT}"
  --dataset_type phys_state_episode
  --phys_state_root "${DATASET_ROOT}"
  --phys_state_split train
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --num_frames "${NUM_FRAMES}"
  --fixed_num_context_frames "${FIXED_NUM_CONTEXT_FRAMES}"
  --max_train_steps "${MAX_TRAIN_STEPS}"
  --num_epochs "${NUM_EPOCHS}"
  --dataset_num_workers "${DATASET_NUM_WORKERS}"
  --learning_rate 1e-4
  --weight_decay 0.01
  --gradient_accumulation_steps 1
  --optimizer_type paged_adamw8bit
  --max_grad_norm 1.0
  --find_unused_parameters
  --save_steps 500
  --max_checkpoints_keep 10
  --remove_prefix_in_ckpt pipe.dit.
  --output_path "${OUTPUT_DIR}"
  --lora_base_model dit
  --lora_target_modules q,k,v,o,ffn.0,ffn.2
  --lora_rank 32
  --lora_alpha 32
  --lora_checkpoint "${BASE_LORA}"
  --extra_inputs input_image
  --enable_object_branch
  --freeze_non_object_trainables
  --train_object_adapter
  --train_object_dit_branch
  --object_num_queries 8
  --aux_max_objects 4
  --jepa_ckpt_path /data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth
  --jepa_input_size 384
  --jepa_patch_size 16
  --jepa_tubelet_size 2
  --cotracker_checkpoint /data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth
  --cotracker_input_h 384
  --cotracker_input_w 512
  --cotracker_window_len 60
  --vggt_model_path /data/gaoya/ckpt/facebook-VGGT-1B
  --vggt_input_h 420
  --vggt_input_w 728
  --object_pooler_latent_dim 16
  --cond_proj_dim 4096
  --jepa_window_radius 1
  --latent_window_radius 1
  --object_gate_init 0.1
  --lambda_main 1.0
  --lambda_track_aux 0.0
  --lambda_box_aux 0.0
  --lambda_depth_aux 0.0
  --stage1a_init_from "${STAGE1A_CKPT}"
  --grounding_proposal_source gdino_only
  --grounding_motion_score_ratio 0.15
  --grounding_text_prompt "box . cube . block . cylinder . capsule . sphere . ball ."
  --grounding_disable_caption_terms
  --grounding_gdino_box_threshold 0.20
  --grounding_gdino_text_threshold 0.15
  --grounding_prompt_frame_mode first
  --grounding_track_dedupe_iou_threshold 0.75
  --grounding_container_suppress_ratio_threshold 0.95
  --grounding_container_suppress_min_contained 2
  --grounding_container_suppress_min_area_ratio 1.5
  --grounding_container_suppress_small_iou_threshold 0.7
  --sam2_segment_len 8
  --report_to wandb
  --wandb_project "${WANDB_PROJECT}"
  --wandb_name "${WANDB_NAME}"
  --wandb_mode "${WANDB_MODE}"
)

if [ "${GROUNDING_GT_BOX_QUERY_REPAIR}" = "1" ]; then
  CMD+=(
    --grounding_gt_box_query_repair
    --grounding_gt_box_oversample_factor "${GROUNDING_GT_BOX_OVERSAMPLE_FACTOR}"
    --grounding_gt_box_min_visible_ratio "${GROUNDING_GT_BOX_MIN_VISIBLE_RATIO}"
    --grounding_gt_box_min_in_box_ratio "${GROUNDING_GT_BOX_MIN_IN_BOX_RATIO}"
  )
fi

CMD+=("${RESUME_ARGS[@]}")

echo "[启动] GPU_SET=${GPU_SET} NUM_PROCESSES=${NUM_PROCESSES} 输出=${OUTPUT_DIR}"
echo "[启动] 命令: ${CMD[*]}"
exec "${CMD[@]}"
