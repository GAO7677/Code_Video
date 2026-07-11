#!/usr/bin/env bash
# Stage1B stability-v2: all-layer residual guard, stronger bounded gate/context
# regularization, slot subset training, and full-slot emphasis.
set -euo pipefail

VISIBLE_GPU_IDS="${VISIBLE_GPU_IDS:-0,1,2,3,5,6}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
OBJECT_AUX_DEVICES="${OBJECT_AUX_DEVICES:-cuda:4,cuda:4,cuda:5,cuda:5}"
RESUME="${RESUME:-/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-003500/training_state.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/train_stage1b_kubric0708_stability_v2_resume3500_20260711}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-28569}"
SAVE_STEPS="${SAVE_STEPS:-500}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_NAME="${WANDB_NAME:-stage1b_kubric0708_stability_v2_resume3500_vis012356}"

IFS=',' read -r -a VISIBLE_GPU_ARRAY <<< "${VISIBLE_GPU_IDS}"
if [ "${#VISIBLE_GPU_ARRAY[@]}" -lt 6 ]; then
  echo "ERROR: 需要至少 6 张可见卡: 前 4 张训练, 后 2 张给 object aux。" >&2
  exit 1
fi
for GPU in "${VISIBLE_GPU_ARRAY[@]}"; do
  if [ "${GPU}" = "4" ]; then
    echo "ERROR: gpu4 故障, 禁止使用。" >&2
    exit 1
  fi
done

PROJ=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
TRAIN_SCRIPT="${PROJ}/code_vjepa_vggt/train0705_kubric_no_gt_box/train_stage1b_context_only_no_gt_box_v_newtrain_kubric.py"
ACCELERATE_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/accelerate
WAN_ROOT=/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B
BASE_LORA=/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors
STAGE1A_CKPT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt
WANDB_DIR=/data/gaoya/agent-data/cache/wandb

mkdir -p "${OUTPUT_DIR}" "${WANDB_DIR}"
LOG_FILE="${OUTPUT_DIR}/train_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

CMD=(
  env PYTHONNOUSERSITE=1 PYTHONPATH="${PROJ}:${DIFFSYNTH_ROOT}"
  CUDA_VISIBLE_DEVICES="${VISIBLE_GPU_IDS}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  WANDB_DIR="${WANDB_DIR}"
  "${ACCELERATE_BIN}" launch --num_processes "${NUM_PROCESSES}" --num_machines 1 --mixed_precision bf16
  "${TRAIN_SCRIPT}"
  --diffsynth_root "${DIFFSYNTH_ROOT}"
  --wan_root "${WAN_ROOT}"
  --dataset_type kubric_no_gt_box
  --kubric_root /data/gaoya/dataset/nnsriram97-phyco_kubric
  --kubric_split train
  --kubric_cache_root /data/gaoya/agent-data/cache/kubric_no_gt_box_dataset
  --kubric_sampling_strategy prefix
  --height 512 --width 896 --num_frames 69
  --fixed_num_context_frames 20 --ctx_max_length 20
  --min_context_frames 0 --max_context_ratio 1.0
  --context_length_sampling short_biased --no_context_ratio 0.0
  --max_train_steps "${MAX_TRAIN_STEPS}" --num_epochs 100 --dataset_num_workers 4
  --learning_rate 1e-4 --weight_decay 0.01 --gradient_accumulation_steps 1
  --optimizer_type paged_adamw8bit --max_grad_norm 1.0 --find_unused_parameters
  --save_steps "${SAVE_STEPS}" --max_checkpoints_keep 20
  --remove_prefix_in_ckpt pipe.dit. --output_path "${OUTPUT_DIR}"
  --lora_base_model dit --lora_target_modules q,k,v,o,ffn.0,ffn.2
  --lora_rank 32 --lora_alpha 32 --lora_checkpoint "${BASE_LORA}"
  --extra_inputs input_image --enable_object_branch --freeze_non_object_trainables
  --train_object_adapter --train_object_dit_branch
  --object_num_queries 8 --aux_max_objects 4
  --jepa_ckpt_path /data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth
  --jepa_input_size 384 --jepa_patch_size 16 --jepa_tubelet_size 2
  --cotracker_checkpoint /data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth
  --cotracker_input_h 384 --cotracker_input_w 512 --cotracker_window_len 60
  --vggt_model_path /data/gaoya/ckpt/facebook-VGGT-1B
  --vggt_input_h 420 --vggt_input_w 728 --object_aux_devices "${OBJECT_AUX_DEVICES}"
  --object_pooler_latent_dim 16 --cond_proj_dim 4096
  --jepa_window_radius 1 --latent_window_radius 1 --object_gate_init 0.1
  --lambda_main 1.0 --lambda_track_aux 0.0 --lambda_box_aux 0.0 --lambda_depth_aux 0.0
  --lambda_object_context_reg 5e-2
  --lambda_object_gate_reg 1e-1 --object_gate_reg_target 0.08
  --object_slot_dropout_prob 0.35 --full_slot_loss_weight 1.5
  --object_branch_train_trace
  --object_branch_ratio_guard_max_ratio 0.15 --object_branch_ratio_guard_max_block_id -1
  --debug_print_object_regularization
  --stage1a_init_from "${STAGE1A_CKPT}"
  --grounding_proposal_source gdino_only --grounding_motion_score_ratio 0.15
  --grounding_text_prompt "box . cube . block . cylinder . capsule . sphere . ball ."
  --grounding_disable_caption_terms
  --grounding_gdino_box_threshold 0.20 --grounding_gdino_text_threshold 0.15
  --grounding_prompt_frame_mode first --grounding_track_dedupe_iou_threshold 0.75
  --grounding_container_suppress_ratio_threshold 0.95
  --grounding_container_suppress_min_contained 2
  --grounding_container_suppress_min_area_ratio 1.5
  --grounding_container_suppress_small_iou_threshold 0.7 --sam2_segment_len 8
  --report_to wandb --wandb_project vjepa_vggt_wan
  --wandb_name "${WANDB_NAME}" --wandb_mode "${WANDB_MODE}"
  --stage2_resume_from "${RESUME}"
)

echo "[train] stability-v2 resume=${RESUME}"
echo "[train] output=${OUTPUT_DIR} log=${LOG_FILE}"
echo "[train] command: ${CMD[*]}"
exec "${CMD[@]}"
