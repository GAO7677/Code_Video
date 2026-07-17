#!/usr/bin/env bash
set -euo pipefail

BASE=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
PROJECT=${BASE}/code_vjepa_vggt/train0717_scheme_e_v4_grounded_self_attention
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
ACCELERATE=/home/gaoya/miniconda3/envs/wan-cu128/bin/accelerate

MAIN_GPU_IDS="${MAIN_GPU_IDS:-5,6}"
AUX_GPU_IDS="${AUX_GPU_IDS:-2,3}"
OBJECT_AUX_DEVICES="${OBJECT_AUX_DEVICES:-cuda:2,cuda:3}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/gaoya/agent-data/checkpoints/scheme_e_v4_grounded_smoke_gpu56_aux23_${RUN_TAG}}"
TMP_ROOT="${TMP_ROOT:-/data/gaoya/agent-data/cache/t/scheme_e_v4_smoke_${RUN_TAG}}"
mkdir -p "${OUTPUT_DIR}" "${TMP_ROOT}"

LOG_FILE="${OUTPUT_DIR}/smoke.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

env \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="${BASE}:${DIFFSYNTH_ROOT}" \
  CUDA_VISIBLE_DEVICES="${MAIN_GPU_IDS},${AUX_GPU_IDS}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  TMPDIR="${TMP_ROOT}" TMP="${TMP_ROOT}" TEMP="${TMP_ROOT}" \
  "${ACCELERATE}" launch --num_processes 2 --num_machines 1 --mixed_precision bf16 \
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
    --openvid_max_samples 1 \
    --mixture_pybullet_ratio 1.0 --mixture_kubric_ratio 0.0 --mixture_openvid_ratio 0.0 \
    --height 512 --width 896 --num_frames 49 \
    --min_timestep_boundary 0.01 --max_timestep_boundary 1.0 \
    --fixed_num_context_frames 8 --replay_fixed_context_frames 8 --ctx_max_length 8 \
    --min_context_frames 0 --max_context_ratio 1.0 --no_context_ratio 0.0 \
    --max_train_steps "${MAX_TRAIN_STEPS:-2}" --num_epochs 2 \
    --dataset_num_workers 0 --learning_rate 1e-5 --weight_decay 0.01 \
    --gradient_accumulation_steps 1 --optimizer_type adamw --max_grad_norm 1.0 \
    --find_unused_parameters --fail_on_nonfinite_train_values \
    --save_steps 2 --max_checkpoints_keep 0 \
    --remove_prefix_in_ckpt pipe.dit. --output_path "${OUTPUT_DIR}" \
    --lora_base_model dit --lora_target_modules q,k,v,o,ffn.0,ffn.2 \
    --lora_rank 32 --lora_alpha 32 \
    --lora_checkpoint /data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors \
    --extra_inputs input_image --enable_object_branch --freeze_non_object_trainables \
    --train_object_pooler --train_object_dit_branch \
    --object_num_queries 8 --aux_max_objects 4 \
    --jepa_ckpt_path /data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth \
    --jepa_input_size 384 --jepa_patch_size 16 --jepa_tubelet_size 2 \
    --cotracker_checkpoint /data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth \
    --cotracker_input_h 384 --cotracker_input_w 512 --cotracker_window_len 60 \
    --object_aux_devices "${OBJECT_AUX_DEVICES}" \
    --object_pooler_latent_dim 48 --cond_proj_dim 256 --object_gate_init 0.0 \
    --tube_num_tokens 4 --tube_hidden_dim 256 --tube_num_heads 8 --tube_num_layers 2 \
    --tube_motion_tokens 4 --tube_motion_fourier_bands 4 --tube_latent_dim 48 \
    --tube_object_attn_dim 256 --tube_object_attn_heads 8 \
    --tube_modality_dropout_prob 0.05 --object_block_ids 14 \
    --grounded_gate_init 0.01 --noun_key_gate_init 0.1 \
    --assignment_loss_weight 0.1 --spatial_bias_strength 0.5 \
    --spatial_bias_dropout_prob 0.25 --evidence_rms_reference 0.01 \
    --evidence_active_threshold 0.001 \
    --lambda_main 1.0 --lambda_object_context_reg 0.0 \
    --lambda_object_gate_reg 0.0 --lambda_object_adapter_mlp_reg 0.0 \
    --object_slot_dropout_prob 0.0 --full_slot_loss_weight 1.0 \
    --object_branch_dropout_prob 0.0 --openvid_object_branch_dropout_prob 1.0 \
    --pybullet_teacher_preservation_lambda 0.0 --kubric_teacher_preservation_lambda 0.0 \
    --openvid_teacher_preservation_lambda 0.0 \
    --object_branch_ratio_guard_max_ratio 0.0 --object_branch_ratio_guard_max_block_id -1 \
    --disable_entity_id_binding \
    --grounding_proposal_source gdino_only --grounding_text_prompt "" \
    --grounding_enable_caption_terms --grounding_caption_prompt_mode physical_noun_phrases \
    --grounding_caption_max_phrases 4 --grounding_caption_min_score 4.0 \
    --grounding_gdino_box_threshold 0.20 --grounding_gdino_text_threshold 0.15 \
    --grounding_prompt_frame_mode first --sam2_segment_len 8 \
    --debug_print_tube_shapes --tube_shape_trace_path "${OUTPUT_DIR}/shape_trace.jsonl" \
    --report_to none

echo "smoke output: ${OUTPUT_DIR}"
