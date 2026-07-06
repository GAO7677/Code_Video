#!/usr/bin/env bash
# Train Wan2.1-T2V-1.3B on the OpenVid + MOVI-D + Genesis rigid mixed recipe.
# This produces the base LoRA that stage 0 will continue from.
# Run:
#   CUDA_VISIBLE_DEVICES=3,5,6,7 sh /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0706_wan1p3b/run_train_openvid_mixed_ctx24_384x672_lora_wan21_13b_gpu0235.sh
set -euo pipefail

GPU_SET="${GPU_SET:-3,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"

if [[ ",${GPU_SET}," == *",4,"* ]]; then
  echo "ERROR: gpu4 故障, 禁止使用。当前 GPU_SET=${GPU_SET}" >&2
  exit 1
fi

ACCELERATE_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/accelerate
PROJ=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
TRAIN_SCRIPT="${PROJ}/code_vjepa_vggt/train0706_wan1p3b/train_v_newtrain.py"
DATASET_CONFIG="${DATASET_CONFIG:-${PROJ}/code_vjepa_vggt/train0706_wan1p3b/dataset_mix_config.json}"
USE_SAMPLE_FULL_VIDEO_LENGTH="${USE_SAMPLE_FULL_VIDEO_LENGTH:-1}"
SAMPLE_FULL_VIDEO_MAX_FRAMES="${SAMPLE_FULL_VIDEO_MAX_FRAMES:-}"

WAN_ROOT=/data/gaoya/ckpt/Wan-AI-Wan2.1-T2V-1.3B
BASE_LORA="${BASE_LORA:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints_wan21_13b/openvid_mixed_ctx24_384x672_lora}"

mkdir -p "${OUTPUT_DIR}"

EXTRA_ARGS=()
if [[ -f "${OUTPUT_DIR}/training_state.pt" || -d "${OUTPUT_DIR}/checkpoints" ]]; then
  EXTRA_ARGS+=(--resume_from "${OUTPUT_DIR}")
fi

CMD=(
  env
  PYTHONPATH="${PROJ}:${DIFFSYNTH_ROOT}"
  CUDA_VISIBLE_DEVICES="${GPU_SET}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  "${ACCELERATE_BIN}" launch --multi_gpu --num_processes "${NUM_PROCESSES}" --num_machines 1 --mixed_precision bf16
  "${TRAIN_SCRIPT}"
  --diffsynth_root "${DIFFSYNTH_ROOT}"
  --wan_root "${WAN_ROOT}"
  --dataset_base_path "${DATASET_CONFIG}"
  --dataset_metadata_path ""
  --height 384
  --width 672
  --num_frames 24
  --use_sample_full_video_length
  --max_train_steps 10000
  --context_sampling_profile mixed_modes
  --min_context_frames 1
  --max_context_ratio 0.5
  --context_reference_frames 49
  --context_reference_prefixes 1,4,8,12,16
  --prefix_context_ratio 0.55
  --first_frame_context_ratio 0.20
  --sparse_context_ratio 0.15
  --random_context_ratio 0.05
  --no_context_ratio 0.05
  --dataset_repeat 1
  --dataset_num_workers 0
  --learning_rate 1e-4
  --weight_decay 0.01
  --num_epochs 10
  --gradient_accumulation_steps 4
  --save_steps 1000
  --benchmark_every_steps 1000
  --benchmark_meta_list_path /home/gaoya/Code_Video/Code_data/Code_train/train_0419/benchmark_meta_json_paths_fixed24.txt
  --benchmark_cuda_visible_devices "${GPU_SET}"
  --benchmark_context_frames 8
  --benchmark_num_frames 24
  --benchmark_height 384
  --benchmark_width 672
  --benchmark_fps 8
  --benchmark_num_inference_steps 50
  --benchmark_cfg_scale 5.0
  --benchmark_seed 42
  --benchmark_script_path /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0706_wan1p3b/batch_eval_lora.py
  --benchmark_output_subdir fixed24_generation
  --validation_every_steps 2000
  --validation_script_path /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0706_wan1p3b/run_validation_vbench.py
  --validation_meta_list_path /home/gaoya/Code_Video/Code_data/Code_train/train_0419/benchmark_meta_json_paths_validation100.txt
  --validation_context_frames_list 0,1,2,4,6,8
  --validation_output_subdir validation100_vbench
  --validation_vbench_config_path /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0706_wan1p3b/vbench_paths.yaml
  --remove_prefix_in_ckpt pipe.dit.
  --output_path "${OUTPUT_DIR}"
  --lora_base_model dit
  --lora_target_modules q,k,v,o,ffn.0,ffn.2
  --lora_rank 32
  --report_to wandb
  --wandb_project openvid-movid-genesis-wan21_13b
  --wandb_mode online
)

if [[ "${USE_SAMPLE_FULL_VIDEO_LENGTH}" != "1" ]]; then
  FILTERED_CMD=()
  for arg in "${CMD[@]}"; do
    if [[ "${arg}" == "--use_sample_full_video_length" ]]; then
      continue
    fi
    FILTERED_CMD+=("${arg}")
  done
  CMD=("${FILTERED_CMD[@]}")
fi

if [[ -n "${SAMPLE_FULL_VIDEO_MAX_FRAMES}" ]]; then
  CMD+=(--sample_full_video_max_frames "${SAMPLE_FULL_VIDEO_MAX_FRAMES}")
fi

CMD+=("${EXTRA_ARGS[@]}")
if [[ -n "${BASE_LORA}" ]]; then
  CMD+=(--lora_checkpoint "${BASE_LORA}")
fi

echo "[启动] GPU_SET=${GPU_SET} NUM_PROCESSES=${NUM_PROCESSES} 输出=${OUTPUT_DIR}"
echo "[启动] 命令: ${CMD[*]}"
exec "${CMD[@]}"
