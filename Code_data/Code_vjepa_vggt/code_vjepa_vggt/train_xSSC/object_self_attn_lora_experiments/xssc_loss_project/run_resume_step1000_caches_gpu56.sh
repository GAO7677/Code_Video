#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT=/data/gaoya/agent-data/checkpoints/xssc_feature_loss/full_sa_object_slot_dedup_xssc50k_xssc_loss_dinov3_movic_step50000/resume_step000500_pybullet100_gpu56_20260814T101910Z
ORIGINAL_COMMAND=${OUTPUT_ROOT}/launch_command.txt
ARCHIVED_COMMAND=${OUTPUT_ROOT}/launch_command_step000500_to_step001000_before_caches.txt
VAE_CACHE=/data/gaoya/AAA_test_video/Dataset_physV/0717pybullet_5000_vbenchtop5/vae_latents_wan22_512x896_49f_prefix_bf16
PROMPT_CACHE=/data/gaoya/AAA_test_video/Dataset_physV/0717pybullet_5000_vbenchtop5/prompt_embeddings_wan22_umt5_bf16
OLD_RESUME=/data/gaoya/agent-data/checkpoints/xssc_feature_loss/full_sa_object_slot_dedup_xssc50k_xssc_loss_dinov3_movic_step50000/formal_gpu01/checkpoints/step-000500/training_state.pt
NEW_RESUME=${OUTPUT_ROOT}/checkpoints/step-001000/training_state.pt
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-2}

if [[ ! -f ${ARCHIVED_COMMAND} ]]; then
  cp --preserve=all "${ORIGINAL_COMMAND}" "${ARCHIVED_COMMAND}"
fi

command=$(<"${ARCHIVED_COMMAND}")
command=${command/--train_batch_size 1/--train_batch_size ${TRAIN_BATCH_SIZE}}
command=${command/--gradient_accumulation_steps 4/--gradient_accumulation_steps ${GRAD_ACCUM_STEPS}}
command=${command/--stage2_resume_from ${OLD_RESUME}/--stage2_resume_from ${NEW_RESUME}}
command=${command/--generation-prompt-suffix Maintain consistent object identity and count throughout the video./--generation-prompt-suffix 'Maintain consistent object identity and count throughout the video.'}
command+=" --pybullet0713_vae_cache_dir ${VAE_CACHE}"
command+=" --pybullet0713_prompt_cache_dir ${PROMPT_CACHE}"

exec env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=5,6 bash -c "${command}"
