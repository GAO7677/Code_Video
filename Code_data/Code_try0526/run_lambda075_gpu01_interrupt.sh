#!/usr/bin/env bash
set -euo pipefail

REPO=/home/gaoya/code_V2V_baselines/PhysRVG-main
PYTHON=/data/gaoya/agent-data/envs/physrvg-full-sa/bin/python
OUT=/data/gaoya/agent-data/checkpoints/generic2175_initial0907_context_noise_20260923
RUN=lambda075-initial0907-seed42-gpu01-interrupt
RUN_DIR="$OUT/$RUN"
LOG=/data/gaoya/agent-data/outputs/context_noise_interpolation_20260923/${RUN}.log
MODEL=/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B-Diffusers
DIT=/data/gaoya/agent-data/weights/physrvg-diffusers-d8caf2/dit/diffusion_pytorch_model.safetensors
DATA=/data/gaoya/dataset/new_data0826/my_process

if pgrep -af -- "--run_name $RUN" >/dev/null; then
  echo "[refuse] training is already running: $RUN" >&2
  exit 3
fi
if [[ -e "$RUN_DIR/resolved_config.json" ]]; then
  echo "[refuse] run already initialized: $RUN_DIR" >&2
  exit 4
fi

mkdir -p "$OUT" "$(dirname "$LOG")"
echo "[$(date -u +%FT%TZ)] [launch] run=$RUN CUDA_VISIBLE_DEVICES=0,1 context_noise_scale=0.75" | tee -a "$LOG"
CUDA_VISIBLE_DEVICES=0,1 \
WANDB_MODE=online WANDB_PROJECT=physrvg-full-sa WANDB_NAME="$RUN" \
PYTHONPATH="$REPO" \
"$PYTHON" -m torch.distributed.run --standalone --master_port 29637 --nproc_per_node=2 \
  "$REPO/scripts_mytrain/train/train_full_sa_pybullet_rl_lora_raw.py" \
  --case_prompt_variant base \
  --expected_trainable_params 80609280 \
  --gradient_checkpointing \
  --lora_rank 32 \
  --lora_alpha 64.0 \
  --lora_dropout 0.0 \
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
  --seed 42 \
  --train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-4 \
  --optimizer paged_adamw8bit \
  --lr_scheduler constant_with_warmup \
  --lr_warmup_steps 0 \
  --checkpointing_steps 500 \
  --checkpoints_total_limit 10 \
  --max_train_steps 2500 \
  --context_noise_scale 0.75 \
  >> "$LOG" 2>&1

"$PYTHON" /home/gaoya/Code_Video/Code_data/Code_try0526/configure_generic2175_context_noise_test70.py \
  --lambda-value 0.75 --apply >> "$LOG" 2>&1
tmux new-session -d -s lambda075-test70-gpu0 \
  "exec bash /home/gaoya/Code_Video/Code_data/Code_try0526/run_context_noise_test70_gpu_queue.sh 0 075"
tmux new-session -d -s lambda075-test70-gpu1 \
  "exec bash /home/gaoya/Code_Video/Code_data/Code_try0526/run_context_noise_test70_gpu_queue.sh 1 075"
