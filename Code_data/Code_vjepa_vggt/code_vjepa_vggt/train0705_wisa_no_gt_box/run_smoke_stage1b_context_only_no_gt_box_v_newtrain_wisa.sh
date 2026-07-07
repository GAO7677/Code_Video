#!/usr/bin/env bash
# =============================================================================
# Stage1B context-only *no-GT-box* SMOKE 测试 (WISA raw dataset, DiffSynth-native)
# =============================================================================
set -euo pipefail

GPU="${GPU:-5}"
RESUME="${RESUME:-none}"

if [ "$GPU" = "4" ]; then
  echo "ERROR: gpu4 故障, 禁止使用。请指定其他 GPU。" >&2
  exit 1
fi

ACCELERATE_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/accelerate
PROJ=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
TRAIN_SCRIPT="${PROJ}/code_vjepa_vggt/train0705_wisa_no_gt_box/train_stage1b_context_only_no_gt_box_v_newtrain_wisa.py"

WAN_ROOT=/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B
BASE_LORA=/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors
STAGE1A_CKPT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt
DATASET_ROOT="${DATASET_ROOT:-/data/gaoya/dataset/qihoo360-WISA-80K}"
WISA_VIDEOS_ROOT="${WISA_VIDEOS_ROOT:-${DATASET_ROOT}/videos}"
WISA_METADATA_PATH="${WISA_METADATA_PATH:-}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
WISA_CACHE_ROOT="${WISA_CACHE_ROOT:-/data/gaoya/agent-data/cache/wisa_no_gt_box_dataset}"
WISA_SAMPLING="${WISA_SAMPLING:-prefix}"
WISA_INIT_SCAN_LIMIT="${WISA_INIT_SCAN_LIMIT:-128}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/gaoya/agent-data/checkpoints/train_stage1b_diffsynth_native0705_wisa/smoke}"

mkdir -p "${OUTPUT_DIR}"

RESUME_ARGS=()
if [ "${RESUME}" != "none" ]; then
  RESUME_ARGS=(--stage2_resume_from "${RESUME}")
  echo "[resume] 续训 smoke: ${RESUME}"
else
  echo "[fresh] 从头 smoke"
fi

CMD=(
  env
  PYTHONPATH="${PROJ}:${DIFFSYNTH_ROOT}"
  CUDA_VISIBLE_DEVICES="${GPU}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  "${ACCELERATE_BIN}" launch --num_processes 1 --num_machines 1 --mixed_precision bf16
  "${TRAIN_SCRIPT}"
  --diffsynth_root "${DIFFSYNTH_ROOT}"
  --wan_root "${WAN_ROOT}"
  --dataset_type wisa_no_gt_box
  --wisa_root "${DATASET_ROOT}"
  --wisa_videos_root "${WISA_VIDEOS_ROOT}"
  --wisa_split "${DATASET_SPLIT}"
  --wisa_cache_root "${WISA_CACHE_ROOT}"
  --wisa_sampling_strategy "${WISA_SAMPLING}"
  --height 512
  --width 896
  --num_frames 24
  --fixed_num_context_frames 8
  --max_train_steps 5
  --num_epochs 100
  --dataset_num_workers 0
  --learning_rate 1e-4
  --weight_decay 0.01
  --gradient_accumulation_steps 1
  --optimizer_type paged_adamw8bit
  --max_grad_norm 1.0
  --find_unused_parameters
  --save_steps 2
  --max_checkpoints_keep 10
  --remove_prefix_in_ckpt pipe.dit.
  --output_path "${OUTPUT_DIR}"
  --lora_base_model dit
  --lora_target_modules q,k,v,o,ffn.0,ffn.2
  --lora_rank 32
  --lora_alpha 32
  --lora_checkpoint "${BASE_LORA}"
  --extra_inputs input_image
  --enable_object_branch
  --freeze_non_object_trainables
  --train_object_adapter
  --train_object_dit_branch
  --object_num_queries 8
  --aux_max_objects 4
  --jepa_ckpt_path /data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth
  --jepa_input_size 384
  --jepa_patch_size 16
  --jepa_tubelet_size 2
  --cotracker_checkpoint /data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth
  --cotracker_input_h 384
  --cotracker_input_w 512
  --cotracker_window_len 60
  --vggt_model_path /data/gaoya/ckpt/facebook-VGGT-1B
  --vggt_input_h 420
  --vggt_input_w 728
  --object_pooler_latent_dim 16
  --cond_proj_dim 4096
  --jepa_window_radius 1
  --latent_window_radius 1
  --object_gate_init 0.1
  --lambda_main 1.0
  --lambda_track_aux 0.0
  --lambda_box_aux 0.0
  --lambda_depth_aux 0.0
  --stage1a_init_from "${STAGE1A_CKPT}"
  --grounding_proposal_source gdino_only
  --grounding_motion_score_ratio 0.15
  --grounding_text_prompt "box . cube . block . cylinder . capsule . sphere . ball ."
  --grounding_disable_caption_terms
  --grounding_gdino_box_threshold 0.20
  --grounding_gdino_text_threshold 0.15
  --grounding_prompt_frame_mode first
  --grounding_track_dedupe_iou_threshold 0.75
  --grounding_container_suppress_ratio_threshold 0.95
  --grounding_container_suppress_min_contained 2
  --grounding_container_suppress_min_area_ratio 1.5
  --grounding_container_suppress_small_iou_threshold 0.7
  --sam2_segment_len 8
  --report_to none
)

if [ -n "${WISA_METADATA_PATH}" ]; then
  CMD+=(--wisa_metadata_path "${WISA_METADATA_PATH}")
fi

if [ "${WISA_INIT_SCAN_LIMIT}" != "0" ]; then
  CMD+=(--wisa_init_scan_limit "${WISA_INIT_SCAN_LIMIT}")
fi

CMD+=("${RESUME_ARGS[@]}")

echo "[smoke] 物理 GPU=${GPU}  输出=${OUTPUT_DIR}"
echo "[smoke] 命令: ${CMD[*]}"
exec "${CMD[@]}"
