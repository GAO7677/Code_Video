#!/usr/bin/env bash
# Resume the best stability-v3 Stage1B checkpoint on raw PyBullet start-0
# 49-frame windows. Global step, optimizer, scheduler, and RNG resume at 4500.
set -euo pipefail

VISIBLE_GPU_IDS="${VISIBLE_GPU_IDS:-GPU-34579b7b-23fc-35ea-539f-1eac72fb7fa5,GPU-74468333-11e0-dfa6-ef16-584a42fa5a02,GPU-994fb224-27dc-b1e0-759d-0226b0c0d775,GPU-05862376-967b-f129-f129-835daf8158cf,GPU-99e4d61a-1169-14e0-d90c-364fdbe30065,GPU-7f6fbc40-3594-2c34-8557-422621355ff9}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
OBJECT_AUX_DEVICES="${OBJECT_AUX_DEVICES:-cuda:4,cuda:4,cuda:5,cuda:5}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-500}"
MAX_CHECKPOINTS_KEEP="${MAX_CHECKPOINTS_KEEP:-10}"
DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-4}"
RAW_ROOT="${RAW_ROOT:-/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/raw_v1/industrial_s1_scale2_merged_h264_batch1500}"
RAW_SPLIT="${RAW_SPLIT:-train}"
FIXED_NUM_CONTEXT_FRAMES="${FIXED_NUM_CONTEXT_FRAMES:-8}"
RESUME_FROM="${RESUME_FROM:-/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708_stability_v3_from_scratch_20260711T144000Z/checkpoints/step-004500}"
WANDB_MODE="${WANDB_MODE:-online}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_pybullet_raw49f_start0_stability_v3_resume4500_${RUN_TAG}}"
WANDB_NAME="${WANDB_NAME:-stage1b_pybullet_raw49f_start0_stability_v3_resume4500_${RUN_TAG}}"
TMP_ROOT="${TMP_ROOT:-/data/gaoya/agent-data/cache/tmp/pybullet_raw49f_resume4500_${RUN_TAG}}"

IFS=',' read -r -a VISIBLE_GPU_ARRAY <<< "${VISIBLE_GPU_IDS}"
if [ "${#VISIBLE_GPU_ARRAY[@]}" -lt 6 ]; then
  echo "ERROR: at least 6 visible GPUs are required: 4 train + 2 object aux." >&2
  exit 1
fi
for GPU in "${VISIBLE_GPU_ARRAY[@]}"; do
  if [ "${GPU}" = "4" ] || [ "${GPU}" = "GPU-4a8abb69-6a43-4b79-5713-31979b8d6d75" ]; then
    echo "ERROR: faulty physical GPU4 must not be used." >&2
    exit 1
  fi
done
if [ ! -d "${RAW_ROOT}/${RAW_SPLIT}" ]; then
  echo "ERROR: raw PyBullet split directory not found: ${RAW_ROOT}/${RAW_SPLIT}" >&2
  exit 1
fi
if [ ! -f "${RESUME_FROM}/checkpoint.safetensors" ] || [ ! -f "${RESUME_FROM}/training_state.pt" ]; then
  echo "ERROR: resume bundle is incomplete: ${RESUME_FROM}" >&2
  exit 1
fi

PROJ=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
TRAIN_SCRIPT="${PROJ}/code_vjepa_vggt/train0705_kubric_no_gt_box/train_stage1b_context_only_no_gt_box_v_newtrain_kubric.py"
ACCELERATE_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/accelerate
WAN_ROOT=/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B
BASE_LORA=/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors
STAGE1A_CKPT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt
WANDB_DIR=/data/gaoya/agent-data/cache/wandb

mkdir -p "${OUTPUT_DIR}" "${WANDB_DIR}" "${TMP_ROOT}" "${TMP_ROOT}/torchinductor"
LOG_FILE="${OUTPUT_DIR}/train_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

CMD=(
  env PYTHONNOUSERSITE=1 PYTHONPATH="${PROJ}:${DIFFSYNTH_ROOT}"
  CUDA_VISIBLE_DEVICES="${VISIBLE_GPU_IDS}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  WANDB_DIR="${WANDB_DIR}" WANDB__DISABLE_STATS=true
  TMPDIR="${TMP_ROOT}" TMP="${TMP_ROOT}" TEMP="${TMP_ROOT}"
  TORCHINDUCTOR_CACHE_DIR="${TMP_ROOT}/torchinductor"
  "${ACCELERATE_BIN}" launch --num_processes "${NUM_PROCESSES}" --num_machines 1 --mixed_precision bf16
  "${TRAIN_SCRIPT}"
  --diffsynth_root "${DIFFSYNTH_ROOT}" --wan_root "${WAN_ROOT}"
  --dataset_type pybullet_raw_no_gt_box
  --pybullet_raw_root "${RAW_ROOT}" --pybullet_raw_split "${RAW_SPLIT}"
  --pybullet_raw_sampling_strategy prefix --pybullet_raw_window_starts 0
  --height 512 --width 896 --num_frames 49
  --fixed_num_context_frames "${FIXED_NUM_CONTEXT_FRAMES}" --ctx_max_length 8
  --min_context_frames 0 --max_context_ratio 1.0
  --context_length_sampling short_biased --no_context_ratio 0.0
  --max_train_steps "${MAX_TRAIN_STEPS}" --num_epochs 100
  --dataset_num_workers "${DATASET_NUM_WORKERS}"
  --learning_rate 1e-4 --weight_decay 0.01 --gradient_accumulation_steps 1
  --optimizer_type paged_adamw8bit --max_grad_norm 1.0 --find_unused_parameters
  --save_steps "${SAVE_STEPS}" --max_checkpoints_keep "${MAX_CHECKPOINTS_KEEP}"
  --remove_prefix_in_ckpt pipe.dit. --output_path "${OUTPUT_DIR}"
  --lora_base_model dit --lora_target_modules q,k,v,o,ffn.0,ffn.2
  --lora_rank 32 --lora_alpha 32 --lora_checkpoint "${BASE_LORA}"
  --stage2_resume_from "${RESUME_FROM}"
  --extra_inputs input_image --enable_object_branch --freeze_non_object_trainables
  --train_object_adapter --train_object_dit_branch
  --object_num_queries 8 --aux_max_objects 4 --compact_object_context_slots
  --jepa_ckpt_path /data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth
  --jepa_input_size 384 --jepa_patch_size 16 --jepa_tubelet_size 2
  --cotracker_checkpoint /data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth
  --cotracker_input_h 384 --cotracker_input_w 512 --cotracker_window_len 60
  --vggt_model_path /data/gaoya/ckpt/facebook-VGGT-1B
  --vggt_input_h 420 --vggt_input_w 728 --object_aux_devices "${OBJECT_AUX_DEVICES}"
  --object_pooler_latent_dim 16 --cond_proj_dim 4096
  --jepa_window_radius 1 --latent_window_radius 1 --object_gate_init 0.1
  --lambda_main 1.0 --lambda_track_aux 0.0 --lambda_box_aux 0.0 --lambda_depth_aux 0.0
  --lambda_object_context_reg 1e-2
  --lambda_object_gate_reg 1e-1 --object_gate_reg_target 0.08
  --lambda_object_adapter_mlp_reg 1e-1
  --object_adapter_mlp_reg_target 2.5 --object_adapter_mlp_residual_max_ratio 3.0
  --object_slot_dropout_prob 0.35 --full_slot_loss_weight 1.0
  --object_branch_train_trace
  --object_branch_ratio_guard_max_ratio 0.30 --object_branch_ratio_guard_max_block_id -1
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
)

echo "[train] stability-v3 resume step-004500 on raw PyBullet start0 49f"
echo "[train] dataset=${RAW_ROOT}/${RAW_SPLIT} samples=1200 raw_frames=0-48"
echo "[train] resume=${RESUME_FROM} max_global_step=${MAX_TRAIN_STEPS}"
echo "[train] output=${OUTPUT_DIR} log=${LOG_FILE}"
echo "[train] command: ${CMD[*]}"
exec "${CMD[@]}"
