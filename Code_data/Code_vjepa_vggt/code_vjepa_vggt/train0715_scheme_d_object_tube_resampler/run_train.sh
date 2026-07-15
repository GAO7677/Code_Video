#!/usr/bin/env bash
set -euo pipefail

BASE=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
PROJECT=${BASE}/code_vjepa_vggt/train0715_scheme_d_object_tube_resampler
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
ACCELERATE=/home/gaoya/miniconda3/envs/wan-cu128/bin/accelerate

VISIBLE_GPU_IDS="${VISIBLE_GPU_IDS:-GPU-34579b7b-23fc-35ea-539f-1eac72fb7fa5,GPU-74468333-11e0-dfa6-ef16-584a42fa5a02,GPU-994fb224-27dc-b1e0-759d-0226b0c0d775,GPU-05862376-967b-f129-f129-835daf8158cf,GPU-7f6fbc40-3594-2c34-8557-422621355ff9,GPU-558afaa4-0f43-84f7-d4f9-281fe35e4c64}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
OBJECT_AUX_DEVICES="${OBJECT_AUX_DEVICES:-cuda:4,cuda:4,cuda:5,cuda:5}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_scheme_d_object_tube_fresh_${RUN_TAG}}"
TMP_ROOT="${TMP_ROOT:-/data/gaoya/agent-data/cache/t/scheme_d_${RUN_TAG}}"
WANDB_DIR=/data/gaoya/agent-data/cache/wandb
mkdir -p "${OUTPUT_DIR}" "${TMP_ROOT}" "${WANDB_DIR}"

LOG_FILE="${OUTPUT_DIR}/train_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

env \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="${BASE}:${DIFFSYNTH_ROOT}" \
  CUDA_VISIBLE_DEVICES="${VISIBLE_GPU_IDS}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  WANDB_DIR="${WANDB_DIR}" WANDB__DISABLE_STATS=true \
  TMPDIR="${TMP_ROOT}" TMP="${TMP_ROOT}" TEMP="${TMP_ROOT}" \
  "${ACCELERATE}" launch --num_processes "${NUM_PROCESSES}" --num_machines 1 --mixed_precision bf16 \
  "${PROJECT}/train.py" \
    --diffsynth_root "${DIFFSYNTH_ROOT}" \
    --wan_root /data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B \
    --dataset_type replay_preserve_mix \
    --pybullet_raw_root /data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/raw_v1/industrial_s1_scale2_merged_h264_batch1500 \
    --pybullet_raw_split train --pybullet_raw_sampling_strategy prefix --pybullet_raw_window_starts 0 \
    --kubric_root /data/gaoya/dataset/nnsriram97-phyco_kubric --kubric_split train \
    --kubric_cache_root /data/gaoya/agent-data/cache/kubric_no_gt_box_dataset \
    --kubric_sampling_strategy prefix --kubric_replay_index_num_frames 69 \
    --kubric_replay_index_num_context_frames 20 \
    --openvid_root /data/gaoya/dataset/mvp-lab-OpenVidHD-0.4M-720p-48fps/train \
    --mixture_pybullet_ratio 0.30 --mixture_kubric_ratio 0.30 --mixture_openvid_ratio 0.40 \
    --height 512 --width 896 --num_frames 49 \
    --fixed_num_context_frames 8 --replay_fixed_context_frames 8 --ctx_max_length 8 \
    --min_context_frames 0 --max_context_ratio 1.0 --no_context_ratio 0.0 \
    --max_train_steps "${MAX_TRAIN_STEPS:-3500}" --num_epochs 100 \
    --dataset_num_workers "${DATASET_NUM_WORKERS:-4}" \
    --learning_rate "${LEARNING_RATE:-1e-5}" --weight_decay 0.01 \
    --gradient_accumulation_steps 1 --optimizer_type paged_adamw8bit \
    --max_grad_norm 1.0 --find_unused_parameters --fail_on_nonfinite_train_values \
    --save_steps "${SAVE_STEPS:-500}" --max_checkpoints_keep 0 \
    --remove_prefix_in_ckpt pipe.dit. --output_path "${OUTPUT_DIR}" \
    --lora_base_model dit --lora_target_modules q,k,v,o,ffn.0,ffn.2 \
    --lora_rank 32 --lora_alpha 32 \
    --lora_checkpoint /data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors \
    --extra_inputs input_image --enable_object_branch --freeze_non_object_trainables \
    --train_object_pooler --train_object_adapter --train_object_dit_branch \
    --object_num_queries 8 --aux_max_objects 4 --compact_object_context_slots \
    --jepa_ckpt_path /data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth \
    --jepa_input_size 384 --jepa_patch_size 16 --jepa_tubelet_size 2 \
    --cotracker_checkpoint /data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth \
    --cotracker_input_h 384 --cotracker_input_w 512 --cotracker_window_len 60 \
    --object_aux_devices "${OBJECT_AUX_DEVICES}" \
    --object_pooler_latent_dim 48 --cond_proj_dim 4096 --object_gate_init 0.1 \
    --tube_num_tokens "${TUBE_NUM_TOKENS:-4}" --tube_hidden_dim "${TUBE_HIDDEN_DIM:-512}" \
    --tube_num_heads 8 --tube_num_layers 2 --tube_latent_dim 48 \
    --tube_modality_dropout_prob 0.10 --object_block_ids 8,11,14,17,20,23 \
    --lambda_main 1.0 --lambda_object_context_reg 1e-2 \
    --lambda_object_gate_reg 1e-1 --object_gate_reg_target 0.08 \
    --lambda_object_adapter_mlp_reg 1e-1 --object_adapter_mlp_reg_target 2.5 \
    --object_adapter_mlp_residual_max_ratio 3.0 \
    --object_slot_dropout_prob 0.35 --full_slot_loss_weight 1.0 \
    --object_branch_dropout_prob 0.20 --openvid_object_branch_dropout_prob 0.50 \
    --pybullet_teacher_preservation_lambda 0.0 --kubric_teacher_preservation_lambda 0.0 \
    --openvid_teacher_preservation_lambda 0.05 \
    --teacher_preservation_every_n_steps 4 --openvid_teacher_preservation_every_n_steps 1 \
    --teacher_preservation_unbiased_interval_scale \
    --object_branch_train_trace --object_branch_ratio_guard_max_ratio 0.30 \
    --object_branch_ratio_guard_max_block_id -1 \
    --entity_binding_sources pybullet,kubric --entity_binding_bottleneck_dim 256 \
    --entity_binding_gate_init 0.5 --entity_binding_dropout_prob 0.20 \
    --entity_binding_residual_max_ratio 0.5 \
    --grounding_proposal_source gdino_only --grounding_text_prompt "" \
    --grounding_enable_caption_terms --grounding_caption_prompt_mode physical_noun_phrases \
    --grounding_caption_max_phrases 4 --grounding_caption_min_score 4.0 \
    --grounding_gdino_box_threshold 0.20 --grounding_gdino_text_threshold 0.15 \
    --grounding_prompt_frame_mode first --sam2_segment_len 8 \
    --report_to wandb --wandb_project vjepa_vggt_wan \
    --wandb_name "scheme_d_object_tube_fresh_${RUN_TAG}" --wandb_mode online

echo "training output: ${OUTPUT_DIR}"
