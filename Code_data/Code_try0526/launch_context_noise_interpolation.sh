#!/usr/bin/env bash
set -euo pipefail

REPO=/home/gaoya/code_V2V_baselines/PhysRVG-main
PYTHON=/data/gaoya/agent-data/envs/physrvg-full-sa/bin/python
OUT=/data/gaoya/agent-data/checkpoints/generic2175_initial0907_context_noise_20260923
LOG=/data/gaoya/agent-data/outputs/context_noise_interpolation_20260923
MODEL=/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B-Diffusers
DIT=/data/gaoya/agent-data/weights/physrvg-diffusers-d8caf2/dit/diffusion_pytorch_model.safetensors
DATA=/data/gaoya/dataset/new_data0826/my_process

mkdir -p "$OUT" "$LOG"

run_one() {
    local lambda_value=$1
    local run_name=$2
    local visible_gpus=$3
    local port=$4
    echo "[launch] lambda=${lambda_value} run=${run_name} CUDA_VISIBLE_DEVICES=${visible_gpus}"
    CUDA_VISIBLE_DEVICES="$visible_gpus" \
    WANDB_MODE=online WANDB_PROJECT=physrvg-full-sa WANDB_NAME="$run_name" \
    PYTHONPATH="$REPO" \
    "$PYTHON" -m torch.distributed.run --standalone --master_port "$port" --nproc_per_node=2 \
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
      --run_name "$run_name" \
      --sample_list "$DATA/case_indexes/generic_full_2175/case_json_paths.txt" \
      --vae_cache_dir "$DATA/train_cache/vae_latents_wan22_512x896_49f_prefix_bf16" \
      --vggt_scene_patch_size 4 4 4 \
      --caption_field caption_initial0907 \
      --max_train_steps 2500 \
      --context_noise_scale "$lambda_value"
}

wait_for_inference_gpus_03() {
    while true; do
        busy=$(
            for proc in /proc/[0-9]*; do
                pid=${proc##*/}
                cmd=$(tr '\0' ' ' < "$proc/cmdline" 2>/dev/null || true)
                case "$cmd" in
                    *infer_full_sa_lora_mixed_clean_context_json_list.py*)
                        env=$(tr '\0' '\n' < "$proc/environ" 2>/dev/null || true)
                        case "$env" in
                            *CUDA_VISIBLE_DEVICES=0*|*CUDA_VISIBLE_DEVICES=3*) echo "$pid";;
                        esac
                        ;;
                esac
            done
        )
        if [[ -z "$busy" ]]; then
            echo "[ready] inference on GPU0 and GPU3 has drained"
            return
        fi
        echo "[wait] GPU0/GPU3 still used by inference pids: ${busy//$'\n'/ }"
        sleep 60
    done
}

# Use only GPUs 1, 2, and 7.  The endpoint A/C weights are reused and are not retrained.
run_one 0.25 lambda025-initial0907-seed42 1,2 29625
run_one 0.50 lambda050-initial0907-seed42 1,7 29626
wait_for_inference_gpus_03
run_one 0.75 lambda075-initial0907-seed42 0,3 29627
echo "[complete] context-noise interpolation runs finished"
