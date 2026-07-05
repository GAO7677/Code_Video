#!/usr/bin/env bash
set -euo pipefail

GPU_SET="${GPU_SET:-3,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
RESUME="${RESUME:-none}"

if [[ ",${GPU_SET}," == *",4,"* ]]; then
  echo "ERROR: gpu4 故障, 禁止使用。当前 GPU_SET=${GPU_SET}" >&2
  exit 1
fi

ACCELERATE_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/accelerate
PROJ=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
TRAIN_SCRIPT="${PROJ}/code_vjepa_vggt/train0706_wan1p3b/train_stage1b_context_only_no_gt_box_v_newtrain.py"

WAN_ROOT=/data/gaoya/ckpt/Wan-AI-Wan2.1-T2V-1.3B
BASE_LORA=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints_wan21_13b/raw_phys_state_wan21_13b_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors
STAGE1A_CKPT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints_wan21_13b/pybullet_teacher_student/stage1a_full_token/step_0005000.pt
DATASET_ROOT=/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/episodes_v1/industrial_s1_scale2_256x144_s8_f16_n6_h264_batch1500
OUTPUT_DIR="${OUTPUT_DIR:-/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints_wan21_13b/train_stage1b_diffsynth_native0706_wan21_13b/run_gpu0235_$(date +%Y%m%d)}"
DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-0}"

mkdir -p "${OUTPUT_DIR}"

RESUME_ARGS=()
if [ "${RESUME}" != "none" ]; then
  RESUME_ARGS=(--stage2_resume_from "${RESUME}")
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
  --height 512
  --width 896
  --num_frames 24
  --fixed_num_context_frames 8
  --max_train_steps 20000
  --num_epochs 100
  --dataset_num_workers "${DATASET_NUM_WORKERS}"
  --learning_rate 1e-4
  --weight_decay 0.01
  --gradient_accumulation_steps 1
  --optimizer_type paged_adamw8bit
  --max_grad_norm 1.0
  --find_unused_parameters
  --save_steps 500
  --max_checkpoints_keep 20
  --remove_prefix_in_ckpt pipe.dit.
  --output_path "${OUTPUT_DIR}"
  --lora_base_model dit
  --lora_target_modules q,k,v,o,ffn.0,ffn.2
  --lora_rank 32
  --lora_alpha 32
  --lora_checkpoint "${BASE_LORA}"
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
  --wandb_project vjepa_vggt_wan
  --wandb_name pybullet0629_teacher_student_stage1b_context_only_no_gt_box_v_newtrain0706_wan21_13b_gpu0235
  --wandb_mode online
)

CMD+=("${RESUME_ARGS[@]}")

echo "[启动] GPU_SET=${GPU_SET} NUM_PROCESSES=${NUM_PROCESSES} 输出=${OUTPUT_DIR}"
echo "[启动] 命令: ${CMD[*]}"
exec "${CMD[@]}"
