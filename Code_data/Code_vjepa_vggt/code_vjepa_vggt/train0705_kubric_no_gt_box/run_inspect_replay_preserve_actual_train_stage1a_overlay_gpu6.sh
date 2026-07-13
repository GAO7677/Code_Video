#!/usr/bin/env bash
set -euo pipefail

PROJ=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
SCRIPT="${PROJ}/code_vjepa_vggt/train0705_kubric_no_gt_box/inspect_replay_preserve_actual_train_stage1a_overlay.py"
OUTPUT_DIR="${OUTPUT_DIR:-/data/gaoya/agent-data/outputs/replay_preserve_actual_train_stage1a_overlay_gpu6_20260713}"
TRAIN_TMUX="${TRAIN_TMUX:-stage1b_replay_preserve_0713}"
NEW_RUN_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_raw49f_kubric_openvid_replay_preserve_init3500_20260713T061500Z
FALLBACK_STATE=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708_stability_v3_from_scratch_20260711T144000Z/checkpoints/step-003500/training_state.pt

while tmux has-session -t "${TRAIN_TMUX}" 2>/dev/null; do
  echo "[wait] ${TRAIN_TMUX} is still using physical GPU6; retrying in 60 seconds."
  sleep 60
done

STAGE2_STATE="${NEW_RUN_ROOT}/checkpoints/step-000300/training_state.pt"
if [ ! -f "${STAGE2_STATE}" ]; then
  STAGE2_STATE="${NEW_RUN_ROOT}/checkpoints/step-000150/training_state.pt"
fi
if [ ! -f "${STAGE2_STATE}" ]; then
  STAGE2_STATE="${FALLBACK_STATE}"
fi

mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/run.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[run] physical_gpu=6 uuid=GPU-7f6fbc40-3594-2c34-8557-422621355ff9"
echo "[run] stage2_state=${STAGE2_STATE}"
echo "[run] output=${OUTPUT_DIR}"

env \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="${PROJ}:${DIFFSYNTH_ROOT}" \
  CUDA_VISIBLE_DEVICES=GPU-7f6fbc40-3594-2c34-8557-422621355ff9 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PYTHON}" "${SCRIPT}" \
  --diffsynth_root "${DIFFSYNTH_ROOT}" \
  --wan_root /data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B \
  --dataset_type replay_preserve_mix \
  --pybullet_raw_root /data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/raw_v1/industrial_s1_scale2_merged_h264_batch1500 \
  --pybullet_raw_split train --pybullet_raw_sampling_strategy prefix --pybullet_raw_window_starts 0 \
  --kubric_root /data/gaoya/dataset/nnsriram97-phyco_kubric --kubric_split train \
  --kubric_cache_root /data/gaoya/agent-data/cache/kubric_no_gt_box_dataset \
  --kubric_sampling_strategy prefix \
  --kubric_replay_index_num_frames 69 --kubric_replay_index_num_context_frames 20 \
  --openvid_root /data/gaoya/dataset/mvp-lab-OpenVidHD-0.4M-720p-48fps/train \
  --mixture_pybullet_ratio 0.30 --mixture_kubric_ratio 0.30 --mixture_openvid_ratio 0.40 \
  --height 512 --width 896 --num_frames 49 \
  --fixed_num_context_frames 8 --ctx_max_length 8 \
  --min_context_frames 0 --max_context_ratio 1.0 --context_length_sampling short_biased --no_context_ratio 0.0 \
  --lora_base_model dit --lora_target_modules q,k,v,o,ffn.0,ffn.2 \
  --lora_rank 32 --lora_alpha 32 \
  --lora_checkpoint /data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors \
  --stage1a_init_from /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt \
  --stage2_resume_from "${STAGE2_STATE}" \
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
  --object_adapter_mlp_residual_max_ratio 3.0 \
  --object_slot_dropout_prob 0.35 --full_slot_loss_weight 1.0 \
  --object_branch_dropout_prob 0.20 --lambda_teacher_preservation 0.05 \
  --teacher_preservation_every_n_steps 4 \
  --grounding_proposal_source gdino_only --grounding_motion_score_ratio 0.15 \
  --grounding_text_prompt "box . cube . block . cylinder . capsule . sphere . ball . person . car . vehicle . container ." \
  --grounding_disable_caption_terms \
  --grounding_gdino_box_threshold 0.20 --grounding_gdino_text_threshold 0.15 \
  --grounding_prompt_frame_mode first --grounding_track_dedupe_iou_threshold 0.75 \
  --grounding_container_suppress_ratio_threshold 0.95 \
  --grounding_container_suppress_min_contained 2 \
  --grounding_container_suppress_min_area_ratio 1.5 \
  --grounding_container_suppress_small_iou_threshold 0.7 --sam2_segment_len 8 \
  --inspect_indices 0,1200,115476,1,1201,115477,2,1202,115478,3,1203,115479 \
  --inspect_num_samples 6 --inspect_seed 42 --inspect_min_object_count 1 \
  --inspect_fps 30 --inspect_output_dir "${OUTPUT_DIR}" \
  --report_to none
