#!/usr/bin/env bash
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
SAVE_STEPS="${SAVE_STEPS:-500}"
MAX_CHECKPOINTS_KEEP="${MAX_CHECKPOINTS_KEEP:-10}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-vjepa_vggt_wan_stage1ab}"
STRUCTURE_ABLATION_TYPE="${STRUCTURE_ABLATION_TYPE:-wo_jepa}"
ABLATION_TAG="${ABLATION_TAG:-stage1b_${STRUCTURE_ABLATION_TYPE}}"
WANDB_NAME="${WANDB_NAME:-${ABLATION_TAG}_gpu0235}"

if [[ ",${GPU_SET}," == *",4,"* ]]; then
  echo "ERROR: gpu4 故障, 禁止使用。当前 GPU_SET=${GPU_SET}" >&2
  exit 1
fi

ACCELERATE_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/accelerate
PROJ=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
TRAIN_SCRIPT="${PROJ}/code_vjepa_vggt/compare_ablation_stage1ab/structure_ablation_train_stage1b_context_only_no_gt_box_v_newtrain.py"

WAN_ROOT=/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B
BASE_LORA=/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors
DATASET_ROOT=/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/episodes_v1/industrial_s1_scale2_256x144_s8_f16_n6_h264_batch1500
STAGE1A_CKPT="${STAGE1A_CKPT:-/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train0705_ablation_stage1ab/stage1a_${STRUCTURE_ABLATION_TYPE}/step_0005000.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train0705_ablation_stage1ab/${ABLATION_TAG}}"
DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-0}"

mkdir -p "${OUTPUT_DIR}"

RESUME_ARGS=()
if [[ "${RESUME}" != "none" ]]; then
  RESUME_ARGS=(--stage2_resume_from "${RESUME}")
fi

CMD=(
  env
  PYTHONPATH="${PROJ}:${DIFFSYNTH_ROOT}"
  CUDA_VISIBLE_DEVICES="${GPU_SET}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
)

if [[ "${NUM_PROCESSES}" == "1" ]]; then
  CMD+=("${ACCELERATE_BIN}" launch --num_processes 1 --num_machines 1 --mixed_precision bf16)
else
  CMD+=("${ACCELERATE_BIN}" launch --multi_gpu --num_processes "${NUM_PROCESSES}" --num_machines 1 --mixed_precision bf16)
fi

CMD+=(
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
  --save_steps "${SAVE_STEPS}"
  --max_checkpoints_keep "${MAX_CHECKPOINTS_KEEP}"
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
  --structure_ablation_type "${STRUCTURE_ABLATION_TYPE}"
  --stage1a_init_from "${STAGE1A_CKPT}"
)

CMD+=("${RESUME_ARGS[@]}")
CMD+=("$@")

echo "[启动] stage1b structure_ablation=${STRUCTURE_ABLATION_TYPE} GPU_SET=${GPU_SET} NUM_PROCESSES=${NUM_PROCESSES}"
echo "[启动] Stage1A init=${STAGE1A_CKPT}"
echo "[启动] 输出=${OUTPUT_DIR}"
echo "[启动] 命令: ${CMD[*]}"
exec "${CMD[@]}"
