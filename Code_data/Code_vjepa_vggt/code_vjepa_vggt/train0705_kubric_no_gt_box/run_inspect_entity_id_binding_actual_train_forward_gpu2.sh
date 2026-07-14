#!/usr/bin/env bash
set -euo pipefail

PROJ=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
ROOT=${PROJ}/code_vjepa_vggt/train0705_kubric_no_gt_box
DIFFSYNTH=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
OUTPUT_DIR=${OUTPUT_DIR:-/data/gaoya/agent-data/outputs/AAA_physv/entity_id_binding_train_forward_smoke_20260714_v4}
GPU=${GPU:-2}

mkdir -p /data/gaoya/agent-data/cache/entity_binding_smoke_tmp

env PYTHONNOUSERSITE=1 \
  PYTHONPATH=${PROJ}:${DIFFSYNTH} \
  CUDA_VISIBLE_DEVICES=${GPU} \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  TMPDIR=/data/gaoya/agent-data/cache/entity_binding_smoke_tmp \
  ${PYTHON} ${ROOT}/inspect_entity_id_binding_actual_train_forward_overlay.py \
  --diffsynth_root ${DIFFSYNTH} \
  --wan_root /data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B \
  --dataset_type pybullet_raw_no_gt_box \
  --pybullet_raw_root /data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/raw_v1/industrial_s1_scale2_merged_h264_batch1500 \
  --pybullet_raw_split train --pybullet_raw_sampling_strategy prefix --pybullet_raw_window_starts 0 \
  --height 512 --width 896 --num_frames 49 \
  --fixed_num_context_frames 8 --replay_fixed_context_frames 8 \
  --lora_base_model dit --lora_target_modules q,k,v,o,ffn.0,ffn.2 --lora_rank 32 --lora_alpha 32 \
  --lora_checkpoint /data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors \
  --stage1a_init_from /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt \
  --stage2_resume_from /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708_stability_v3_from_scratch_20260711T144000Z/checkpoints/step-003500 \
  --extra_inputs input_image --enable_object_branch --freeze_non_object_trainables \
  --train_object_adapter --train_object_dit_branch \
  --object_num_queries 8 --aux_max_objects 4 --compact_object_context_slots \
  --jepa_ckpt_path /data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth \
  --jepa_input_size 384 --jepa_patch_size 16 --jepa_tubelet_size 2 \
  --cotracker_checkpoint /data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth \
  --cotracker_input_h 384 --cotracker_input_w 512 --cotracker_window_len 60 \
  --vggt_model_path /data/gaoya/ckpt/facebook-VGGT-1B \
  --vggt_input_h 420 --vggt_input_w 728 --object_aux_devices cuda:0 \
  --object_pooler_latent_dim 16 --cond_proj_dim 4096 \
  --jepa_window_radius 1 --latent_window_radius 1 --object_gate_init 0.1 \
  --lambda_main 1.0 --lambda_track_aux 0.0 --lambda_box_aux 0.0 --lambda_depth_aux 0.0 \
  --lambda_object_context_reg 1e-2 \
  --lambda_object_gate_reg 1e-1 --object_gate_reg_target 0.08 \
  --lambda_object_adapter_mlp_reg 1e-1 --object_adapter_mlp_reg_target 2.5 \
  --object_adapter_mlp_residual_max_ratio 3.0 \
  --object_slot_dropout_prob 0.35 --full_slot_loss_weight 1.0 \
  --object_branch_dropout_prob 0.0 --openvid_object_branch_dropout_prob 0.50 \
  --lambda_teacher_preservation 0.05 \
  --pybullet_teacher_preservation_lambda 0.0 --kubric_teacher_preservation_lambda 0.0 \
  --openvid_teacher_preservation_lambda 0.05 \
  --teacher_preservation_every_n_steps 4 --openvid_teacher_preservation_every_n_steps 1 \
  --teacher_preservation_unbiased_interval_scale \
  --object_branch_train_trace \
  --object_branch_ratio_guard_max_ratio 0.30 --object_branch_ratio_guard_max_block_id -1 \
  --grounding_proposal_source gdino_only --grounding_motion_score_ratio 0.15 \
  --grounding_text_prompt "box . cube . block . cylinder . capsule . sphere . ball . person . car . vehicle . container ." \
  --grounding_disable_caption_terms \
  --grounding_gdino_box_threshold 0.20 --grounding_gdino_text_threshold 0.15 \
  --grounding_prompt_frame_mode first --grounding_track_dedupe_iou_threshold 0.75 \
  --grounding_container_suppress_ratio_threshold 0.95 --grounding_container_suppress_min_contained 2 \
  --grounding_container_suppress_min_area_ratio 1.5 \
  --grounding_container_suppress_small_iou_threshold 0.7 --sam2_segment_len 8 \
  --entity_binding_sources pybullet,kubric --entity_binding_bottleneck_dim 256 \
  --entity_binding_gate_init 0.1 --entity_binding_dropout_prob 0.2 \
  --entity_binding_residual_max_ratio 0.1 \
  --inspect_num_samples 1 --inspect_min_object_count 1 --inspect_seed 42 --inspect_fps 24 \
  --inspect_output_dir ${OUTPUT_DIR}
