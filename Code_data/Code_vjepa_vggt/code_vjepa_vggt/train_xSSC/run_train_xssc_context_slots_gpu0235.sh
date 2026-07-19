#!/usr/bin/env bash
# Train Wan2.2 with frozen context-only RandSFQ2 slots on GPUs 0,2,3,5.
set -euo pipefail

GPU_SET="${GPU_SET:-0,2,3,5}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
RESUME="${RESUME:-none}"
NUM_FRAMES="${NUM_FRAMES:-24}"
FIXED_NUM_CONTEXT_FRAMES="${FIXED_NUM_CONTEXT_FRAMES:-8}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-20000}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-896}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-xssc_wan_physics}"
WANDB_NAME="${WANDB_NAME:-xssc_ctx_slots_wan22_5b_gpu0235}"

if [[ ",${GPU_SET}," == *",4,"* ]]; then
  echo "ERROR: gpu4 is unavailable; GPU_SET=${GPU_SET}" >&2
  exit 1
fi

ACCELERATE_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/accelerate
PROJ=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
TRAIN_SCRIPT="${PROJ}/code_vjepa_vggt/train_xSSC/train_xssc_context_slots.py"
XSSC_ROOT=/home/gaoya/Code_Video/xSSC-main
XSSC_CONFIG="${XSSC_ROOT}/config-randsfq/rsfq2_r-ytvis.py"
XSSC_CHECKPOINT=/data/gaoya/ckpt/xSSC/rsfq2_r-ytvis/42-0130.pth

WAN_ROOT=/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B
BASE_LORA=/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors
DATASET_ROOT=/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/episodes_v1/industrial_s1_scale2_256x144_s8_f16_n6_h264_batch1500
OUTPUT_DIR="${OUTPUT_DIR:-/data/gaoya/agent-data/checkpoints/train_xssc_context_slots/wan22_5b_gpu0235}"
DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-0}"
CACHE_ROOT=/data/gaoya/agent-data/cache/xssc_wan

mkdir -p "${OUTPUT_DIR}" "${CACHE_ROOT}/huggingface" "${CACHE_ROOT}/torch" "${CACHE_ROOT}/xdg"

RESUME_ARGS=()
if [ "${RESUME}" != "none" ]; then
  RESUME_ARGS=(--stage2_resume_from "${RESUME}")
  echo "[resume] ${RESUME}"
else
  echo "[fresh] starting from the frozen physical-state LoRA"
fi

CMD=(
  env
  PYTHONPATH="${PROJ}:${DIFFSYNTH_ROOT}"
  CUDA_VISIBLE_DEVICES="${GPU_SET}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  HF_HOME="${CACHE_ROOT}/huggingface"
  TORCH_HOME="${CACHE_ROOT}/torch"
  XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
  "${ACCELERATE_BIN}" launch --multi_gpu --num_processes "${NUM_PROCESSES}" --num_machines 1 --mixed_precision bf16
  "${TRAIN_SCRIPT}"
  --diffsynth_root "${DIFFSYNTH_ROOT}"
  --wan_root "${WAN_ROOT}"
  --xssc_root "${XSSC_ROOT}"
  --xssc_config "${XSSC_CONFIG}"
  --xssc_checkpoint "${XSSC_CHECKPOINT}"
  --xssc_input_size 256
  --xssc_max_time_steps 64
  --dataset_type phys_state_episode
  --phys_state_root "${DATASET_ROOT}"
  --phys_state_split train
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --num_frames "${NUM_FRAMES}"
  --fixed_num_context_frames "${FIXED_NUM_CONTEXT_FRAMES}"
  --max_train_steps "${MAX_TRAIN_STEPS}"
  --num_epochs "${NUM_EPOCHS}"
  --dataset_num_workers "${DATASET_NUM_WORKERS}"
  --learning_rate 1e-4
  --weight_decay 0.01
  --gradient_accumulation_steps 1
  --optimizer_type paged_adamw8bit
  --max_grad_norm 1.0
  --find_unused_parameters
  --save_steps 500
  --max_checkpoints_keep 10
  --remove_prefix_in_ckpt pipe.dit.
  --output_path "${OUTPUT_DIR}"
  --lora_base_model dit
  --lora_target_modules q,k,v,o,ffn.0,ffn.2
  --lora_rank 32
  --lora_alpha 32
  --lora_checkpoint "${BASE_LORA}"
  --extra_inputs input_image
  --object_gate_init 0.1
  --lambda_main 1.0
  --lambda_object_context_reg 1e-4
  --report_to wandb
  --wandb_project "${WANDB_PROJECT}"
  --wandb_name "${WANDB_NAME}"
  --wandb_mode "${WANDB_MODE}"
)

CMD+=("${RESUME_ARGS[@]}")

echo "[launch] GPU_SET=${GPU_SET} NUM_PROCESSES=${NUM_PROCESSES}"
echo "[launch] xSSC=${XSSC_CHECKPOINT}"
echo "[launch] default object token shape=[B,$((FIXED_NUM_CONTEXT_FRAMES * 7)),3072]"
echo "[launch] output=${OUTPUT_DIR}"
echo "[launch] command: ${CMD[*]}"
exec "${CMD[@]}"
