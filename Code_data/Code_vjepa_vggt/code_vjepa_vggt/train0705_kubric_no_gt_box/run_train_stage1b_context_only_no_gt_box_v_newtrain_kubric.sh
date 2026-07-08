#!/usr/bin/env bash
# =============================================================================
# Stage1B context-only *no-GT-box* 训练启动脚本 (Kubric raw dataset, DiffSynth-native)
#
# 迁移自:
#   object_token_teacher_student/run_train_teacher_student_stage1b_context_only_no_gt_box.sh
#     (-> ContextOnlyInjectionNoGTBoxTrainer)
# 到:
#   run_train_v_newtrain_gpu2367.sh 所用的 DiffSynth-native 框架
#     (-> train_v_newtrain.WanTrainingModule)
#
# 用法:
#   bash run_train_stage1b_context_only_no_gt_box_v_newtrain0705.sh                          # 默认可见 0,2,3,5,6,7；训练占前 4 个可见槽位
#   VISIBLE_GPU_IDS=0,2,3,5,6,7 bash run_train_stage1b_context_only_no_gt_box_v_newtrain0705.sh
#   RESUME=<state.pt|dir> bash ...0705.sh                                        # 断点续训
#
# 说明:
#   - 前台运行, 不使用 nohup / & / 后台方式 (遵循工作区规则)
#   - 禁用 gpu4 (故障)
#   - 默认暴露 6 张卡: 0,2,3,5,6,7
#   - 默认四进程训练卡: 总是占用前 4 个可见槽位 0,1,2,3 (对应物理 0,2,3,5)
#   - 默认 object_aux_devices: 4,4,5,5 (对应物理 6,6,7,7)
#   - 注意: 这里不能给 accelerate 传 --gpu_ids，否则它会把子进程 CUDA_VISIBLE_DEVICES 重写为 0,1,2,3，导致 aux 槽位 4/5 失效
#   - 目标 DiffSynth 框架: WAN_2p2/DiffSynth-Studio-main
#   - 每 500 step 保存一次 (--save_steps)
#   - 当前正式 train split 需要按实际 num_frames 配置确认；默认先保留 28569 steps
#   - 基础 Wan LoRA (raw-phys, 冻结) 从 --lora_checkpoint 加载
#   - Stage1A token builder (object_pooler/object_aux_heads, 冻结) 从 --stage1a_init_from 加载
#   - 可训练: DiT object 注入分支 + ObjectConditionAdapter
# =============================================================================
set -euo pipefail

# ---- 可配置项 (环境变量覆盖) ----
VISIBLE_GPU_IDS="${VISIBLE_GPU_IDS:-0,2,3,5,6,7}"   # 物理 GPU 可见列表 (禁止包含 gpu4)
RESUME="${RESUME:-none}"        # none=从头开始, 或指定 training_state.pt / checkpoint 目录

IFS=',' read -r -a VISIBLE_GPU_ARRAY <<< "${VISIBLE_GPU_IDS}"
if [ "${#VISIBLE_GPU_ARRAY[@]}" -lt 6 ]; then
  echo "ERROR: 当前布局要求至少 6 张可见卡: 前 4 张给训练进程, 后 2 张给 object aux。" >&2
  exit 1
fi
for GPU in "${VISIBLE_GPU_ARRAY[@]}"; do
  if [ "${GPU}" = "4" ]; then
    echo "ERROR: gpu4 故障, 禁止使用。请指定其他 GPU。" >&2
    exit 1
  fi
done
NUM_PROCESSES=4

ACCELERATE_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/accelerate
PROJ=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
TRAIN_SCRIPT="${PROJ}/code_vjepa_vggt/train0705_kubric_no_gt_box/train_stage1b_context_only_no_gt_box_v_newtrain_kubric.py"

WAN_ROOT=/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B
BASE_LORA=/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors
STAGE1A_CKPT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt
DATASET_ROOT="${DATASET_ROOT:-/data/gaoya/dataset/nnsriram97-phyco_kubric}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
KUBRIC_CACHE_ROOT="${KUBRIC_CACHE_ROOT:-/data/gaoya/agent-data/cache/kubric_no_gt_box_dataset}"
KUBRIC_SAMPLING="${KUBRIC_SAMPLING:-prefix}"
KUBRIC_INIT_SCAN_LIMIT="${KUBRIC_INIT_SCAN_LIMIT:-0}"
OBJECT_AUX_DEVICES="${OBJECT_AUX_DEVICES:-cuda:4,cuda:4,cuda:5,cuda:5}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708}"

mkdir -p "${OUTPUT_DIR}"

RESUME_ARGS=()
if [ "${RESUME}" != "none" ]; then
  RESUME_ARGS=(--stage2_resume_from "${RESUME}")
  echo "[resume] 断点续训: ${RESUME}"
else
  echo "[fresh] 从头开始训练"
fi

CMD=(
  env
  PYTHONNOUSERSITE=1
  PYTHONPATH="${PROJ}:${DIFFSYNTH_ROOT}"
  CUDA_VISIBLE_DEVICES="${VISIBLE_GPU_IDS}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  "${ACCELERATE_BIN}" launch --num_processes "${NUM_PROCESSES}" --num_machines 1 --mixed_precision bf16
  "${TRAIN_SCRIPT}"
  --diffsynth_root "${DIFFSYNTH_ROOT}"
  --wan_root "${WAN_ROOT}"
  --dataset_type kubric_no_gt_box
  --kubric_root "${DATASET_ROOT}"
  --kubric_split "${DATASET_SPLIT}"
  --kubric_cache_root "${KUBRIC_CACHE_ROOT}"
  --kubric_sampling_strategy "${KUBRIC_SAMPLING}"
  --height 512
  --width 896
  --num_frames 69
  --fixed_num_context_frames 20
  --ctx_max_length 20
  --min_context_frames 0
  --max_context_ratio 1.0
  --context_length_sampling short_biased
  --no_context_ratio 0.0
  --max_train_steps 28569
  --num_epochs 100
  --dataset_num_workers 4
  --learning_rate 1e-4
  --weight_decay 0.01
  --gradient_accumulation_steps 1
  --optimizer_type paged_adamw8bit
  --max_grad_norm 1.0
  --find_unused_parameters
  --save_steps 500
  --max_checkpoints_keep 20
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
  --object_aux_devices "${OBJECT_AUX_DEVICES}"
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
  --report_to wandb
  --wandb_project vjepa_vggt_wan
  --wandb_name stage1b_kubric0708_vis023567_train0123_aux4455_f69_ctx020_1epoch
  --wandb_mode online
)

if [ "${KUBRIC_INIT_SCAN_LIMIT}" != "0" ]; then
  CMD+=(--kubric_init_scan_limit "${KUBRIC_INIT_SCAN_LIMIT}")
fi

CMD+=("${RESUME_ARGS[@]}")

echo "[启动] 可见物理 GPU=${VISIBLE_GPU_IDS}  训练槽位=0,1,2,3  aux=${OBJECT_AUX_DEVICES}  进程数=${NUM_PROCESSES}  输出=${OUTPUT_DIR}"
echo "[启动] 命令: ${CMD[*]}"
exec "${CMD[@]}"
