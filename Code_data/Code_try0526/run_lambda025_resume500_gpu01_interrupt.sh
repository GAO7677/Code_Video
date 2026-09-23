#!/usr/bin/env bash
set -euo pipefail

REPO=/home/gaoya/code_V2V_baselines/PhysRVG-main
PYTHON=/data/gaoya/agent-data/envs/physrvg-full-sa/bin/python
OUT=/data/gaoya/agent-data/checkpoints/generic2175_initial0907_context_noise_20260923
RUN=lambda025-initial0907-seed42-resume500-gpu01-interrupt
LOG=/data/gaoya/agent-data/outputs/context_noise_interpolation_20260923/${RUN}.log
MODEL=/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B-Diffusers
DIT=/data/gaoya/agent-data/weights/physrvg-diffusers-d8caf2/dit/diffusion_pytorch_model.safetensors
DATA=/data/gaoya/dataset/new_data0826/my_process
RESUME=$OUT/lambda025-initial0907-seed42/checkpoints/step-000500

mkdir -p "$OUT" "$(dirname "$LOG")"
echo "[launch] run=$RUN CUDA_VISIBLE_DEVICES=0,1 resume_step=500" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=0,1 \
WANDB_MODE=online WANDB_PROJECT=physrvg-full-sa WANDB_NAME="$RUN" \
PYTHONPATH="$REPO" \
"$PYTHON" -m torch.distributed.run --standalone --master_port 29635 --nproc_per_node=2 \
  "$REPO/scripts_mytrain/train/train_full_sa_pybullet_rl_lora_raw.py" \
  --case_prompt_variant base \
  --expected_trainable_params 80609280 \
  --gradient_checkpointing \
  --lora_alpha 64.0 \
  --model_id "$MODEL" \
  --output_dir "$OUT" \
  --physrvg_dit_checkpoint "$DIT" \
  --prompt_cache_dir "$DATA/train_cache/prompt_embeddings_initial0907" \
  --pybullet_root "$DATA/asset_generic_full_49f" \
  --run_name "$RUN" \
  --sample_list "$DATA/case_indexes/generic_full_2175/case_json_paths.txt" \
  --vae_cache_dir "$DATA/train_cache/vae_latents_wan22_512x896_49f_prefix_bf16" \
  --vggt_scene_patch_size 4 4 4 \
  --caption_field caption_initial0907 \
  --max_train_steps 2500 \
  --resume_lora_checkpoint "$RESUME" \
  --resume_global_step 500 \
  --context_noise_scale 0.25 \
  >> "$LOG" 2>&1
