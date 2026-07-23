#!/usr/bin/env bash
# Train Wan2.2 with frozen context-only RandSFQ2 slots on 0717 PyBullet only.
set -euo pipefail

# Run:
# tmux new-session -d -s xssc_pybullet0717_gpu0123 -- \
#   bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/run_train_xssc_context_slots_pybullet0717_gpu0123.sh

GPU_SET="${GPU_SET:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
RESUME="${RESUME:-none}"
NUM_FRAMES="${NUM_FRAMES:-49}"
FIXED_NUM_CONTEXT_FRAMES="${FIXED_NUM_CONTEXT_FRAMES:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
OBJECT_LORA_RANK="${OBJECT_LORA_RANK:-32}"
OBJECT_LORA_ALPHA="${OBJECT_LORA_ALPHA:-32}"
OBJECT_LORA_DROPOUT="${OBJECT_LORA_DROPOUT:-0.05}"
XSSC_SLOT_TRACK_DROPOUT="${XSSC_SLOT_TRACK_DROPOUT:-0.10}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-20000}"
SAVE_STEPS="${SAVE_STEPS:-500}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-896}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-xssc_wan_physics}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
WANDB_NAME="${WANDB_NAME:-xssc_pybullet0717_49f_ctx08_${RUN_TAG}}"

if [ "${FIXED_NUM_CONTEXT_FRAMES}" -ne 8 ]; then
  echo "ERROR: xSSC training requires FIXED_NUM_CONTEXT_FRAMES=8" >&2
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
PYBULLET_ROOT=/data/gaoya/AAA_test_video/Dataset_physV/0717pybullet_5000_vbenchtop5
KUBRIC_ROOT=/data/gaoya/dataset/nnsriram97-phyco_kubric
OPENVID_ROOT=/data/gaoya/dataset/mvp-lab-OpenVidHD-0.4M-720p-48fps/train
OUTPUT_DIR="${OUTPUT_DIR:-/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/offcial_xSSC/train_xssc_context_slots_pybullet0717/wan22_5b_pybullet0717_49f_ctx08_${RUN_TAG}}"
DATASET_NUM_WORKERS="${DATASET_NUM_WORKERS:-0}"
CACHE_ROOT=/data/gaoya/agent-data/cache/xssc_wan
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/train.log}"

mkdir -p "${OUTPUT_DIR}" "${CACHE_ROOT}/huggingface" "${CACHE_ROOT}/torch" "${CACHE_ROOT}/xdg"
exec > >(tee -a "${LOG_FILE}") 2>&1

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
  --object_lora_rank "${OBJECT_LORA_RANK}"
  --object_lora_alpha "${OBJECT_LORA_ALPHA}"
  --object_lora_dropout "${OBJECT_LORA_DROPOUT}"
  --xssc_slot_track_dropout "${XSSC_SLOT_TRACK_DROPOUT}"
  --dataset_type xssc_replay_mix
  --pybullet0713_root "${PYBULLET_ROOT}"
  --pybullet0713_split train
  --pybullet0713_sampling_strategy prefix
  --kubric_root "${KUBRIC_ROOT}"
  --kubric_split train
  --kubric_sampling_strategy prefix
  --kubric_cache_root /data/gaoya/agent-data/cache/kubric_no_gt_box_dataset
  --kubric_replay_index_num_frames 69
  --kubric_replay_index_num_context_frames 20
  --openvid_root "${OPENVID_ROOT}"
  --mixture_pybullet_ratio 1.0
  --mixture_kubric_ratio 0.0
  --mixture_openvid_ratio 0.0
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --num_frames "${NUM_FRAMES}"
  --fixed_num_context_frames "${FIXED_NUM_CONTEXT_FRAMES}"
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --no_context_ratio 0.0
  --max_train_steps "${MAX_TRAIN_STEPS}"
  --num_epochs "${NUM_EPOCHS}"
  --dataset_num_workers "${DATASET_NUM_WORKERS}"
  --learning_rate 1e-4
  --weight_decay 0.01
  --gradient_accumulation_steps 1
  --optimizer_type paged_adamw8bit
  --max_grad_norm 1.0
  --save_steps "${SAVE_STEPS}"
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
  --fail_on_nonfinite_train_values
  --debug_print_object_regularization
)

CMD+=("${RESUME_ARGS[@]}")

echo "[launch] GPU_SET=${GPU_SET} NUM_PROCESSES=${NUM_PROCESSES}"
echo "[launch] per-GPU batch=${TRAIN_BATCH_SIZE} global batch=$((TRAIN_BATCH_SIZE * NUM_PROCESSES))"
echo "[launch] data=100% 0717-PyBullet, frames=${NUM_FRAMES}, ctx=${FIXED_NUM_CONTEXT_FRAMES}"
echo "[launch] xSSC=${XSSC_CHECKPOINT}"
echo "[launch] object token shape=[B,$((FIXED_NUM_CONTEXT_FRAMES * 7)),3072]"
echo "[launch] object cross-attention LoRA rank=${OBJECT_LORA_RANK} alpha=${OBJECT_LORA_ALPHA}"
echo "[launch] dropout: slot-track=${XSSC_SLOT_TRACK_DROPOUT} object-LoRA=${OBJECT_LORA_DROPOUT}"
echo "[launch] output=${OUTPUT_DIR}"
echo "[launch] log=${LOG_FILE}"
echo "[launch] command: ${CMD[*]}"
exec "${CMD[@]}"
