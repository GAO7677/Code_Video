#!/usr/bin/env bash
set -euo pipefail

REPO=/home/gaoya/code_V2V_baselines/PhysRVG-main
PYTHON=/data/gaoya/agent-data/envs/physrvg-full-sa/bin/python
OUT=/data/gaoya/agent-data/checkpoints/generic2175_initial0907_context_noise_20260923
MODEL=/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B-Diffusers
DIT=/data/gaoya/agent-data/weights/physrvg-diffusers-d8caf2/dit/diffusion_pytorch_model.safetensors
DATA=/data/gaoya/dataset/new_data0826/my_process

while true; do
    busy=0
    for proc in /proc/[0-9]*; do
        pid=${proc##*/}
        cmd=$(tr '\0' ' ' < "$proc/cmdline" 2>/dev/null || true)
        case "$cmd" in
            *infer_full_sa_lora_mixed_clean_context_json_list.py*)
                env=$(tr '\0' '\n' < "$proc/environ" 2>/dev/null || true)
                case "$env" in
            *CUDA_VISIBLE_DEVICES=5*|*CUDA_VISIBLE_DEVICES=6*) busy=1;;
                esac
                ;;
        esac
    done
    if [[ "$busy" == 0 ]]; then
        break
    fi
    echo "[wait] GPU5/GPU6 inference is still active"
    sleep 60
done

echo "[launch] lambda=0.75 run=lambda075-initial0907-seed42 CUDA_VISIBLE_DEVICES=5,6"
CUDA_VISIBLE_DEVICES=5,6 \
WANDB_MODE=online WANDB_PROJECT=physrvg-full-sa WANDB_NAME=lambda075-initial0907-seed42 \
PYTHONPATH="$REPO" \
"$PYTHON" -m torch.distributed.run --standalone --master_port 29627 --nproc_per_node=2 \
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
  --run_name lambda075-initial0907-seed42 \
  --sample_list "$DATA/case_indexes/generic_full_2175/case_json_paths.txt" \
  --vae_cache_dir "$DATA/train_cache/vae_latents_wan22_512x896_49f_prefix_bf16" \
  --vggt_scene_patch_size 4 4 4 \
  --caption_field caption_initial0907 \
  --max_train_steps 2500 \
  --context_noise_scale 0.75
