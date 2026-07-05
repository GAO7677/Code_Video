#!/usr/bin/env bash
set -euo pipefail

# Current plan:
# - baseline: official Wan2.2 TI2V
# - base lora 1: openvid_lora_10000
# - base lora 2: 0613pybullet_lora_000500
# - train0705: step-002500 and step-007000
# - datasets:
#   - morpheus_real_world (123 cases)
#   - physicIQ (67 cases)
#
# Notes:
# - Keep seed=42, steps=40, cfg=5.0
# - Do not use gpu4
# - This script runs each job sequentially by default for predictability.
#   If you want parallel runs, split the commands by section manually.

PY=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
REPO=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
WAN22=/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main
TRAIN0419=/home/gaoya/Code_Video/Code_data/Code_train/train_0419

LIST_MORPHEUS=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_morpheus_real_world.txt
LIST_PHYSICIQ=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt

TRAIN0705_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints
OPENVID_LORA_ROOT=/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora/checkpoints/step-010000
PYBULLET_LORA_ROOT=/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500

run_dataset() {
  local dataset_tag="$1"
  local list_path="$2"

  echo "============================================================"
  echo "[start] dataset=${dataset_tag}"
  echo "[list]  ${list_path}"
  echo "============================================================"

  echo "[1/5] baseline wan2p2_ti2v5B"
  PYTHONPATH="${REPO}:${WAN22}" \
  CUDA_VISIBLE_DEVICES=0 \
  "${PY}" \
  "${REPO}/code_vjepa_vggt/AAAinfer/wanti2v.py" \
    --input-list "${list_path}" \
    --model-name "wan2p2_ti2v5B_${dataset_tag}" \
    --output-root "/data/gaoya/AAA_test_video/0623/test/v2v/basemodel/wan2p2_ti2v5B_${dataset_tag}" \
    --frame-num 25 \
    --sampling-steps 40 \
    --cfg-scale 5.0 \
    --fps 30 \
    --seed 42 \
    --offload-model

  echo "[2/5] openvid_lora_10000"
  PYTHONPATH="${REPO}:${DIFFSYNTH}:${TRAIN0419}" \
  CUDA_VISIBLE_DEVICES=1 \
  "${PY}" \
  "${REPO}/code_vjepa_vggt/AAAinfer/wan_openvid_lorav2v.py" \
    --weights-root "${OPENVID_LORA_ROOT}" \
    --input-json-list-path "${list_path}" \
    --model-name "wan_openvid_lorav2v_step10000_${dataset_tag}" \
    --output-root "/data/gaoya/AAA_test_video/0623/test/v2v/loramodel/wan_openvid_lorav2v_step10000_${dataset_tag}" \
    --runtime-root "/data/gaoya/AAA_test_video/0623/test/v2v/loramodel/wan_openvid_lorav2v_step10000_${dataset_tag}_runtime" \
    --num-frames 25 \
    --num-inference-steps 40 \
    --cfg-scale 5.0 \
    --seed 42

  echo "[3/5] 0613pybullet_lora_000500"
  PYTHONPATH="${REPO}:${DIFFSYNTH}:${TRAIN0419}" \
  CUDA_VISIBLE_DEVICES=2 \
  "${PY}" \
  "${REPO}/code_vjepa_vggt/AAAinfer/wan_openvid_0613pybullet_lorav2v.py" \
    --weights-root "${PYBULLET_LORA_ROOT}" \
    --input-json-list-path "${list_path}" \
    --model-name "wan_openvid_0613pybullet_lorav2v_step000500_${dataset_tag}" \
    --output-root "/data/gaoya/AAA_test_video/0623/test/v2v/loramodel/wan_openvid_0613pybullet_lorav2v_step000500_${dataset_tag}" \
    --runtime-root "/data/gaoya/AAA_test_video/0623/test/v2v/loramodel/wan_openvid_0613pybullet_lorav2v_step000500_${dataset_tag}_runtime" \
    --num-frames 25 \
    --num-inference-steps 40 \
    --cfg-scale 5.0 \
    --seed 42

  echo "[4/5] train0705 step-002500"
  PYTHONPATH="${REPO}:${DIFFSYNTH}" \
  CUDA_VISIBLE_DEVICES=3 \
  "${PY}" \
  "${REPO}/code_vjepa_vggt/train0705/wan_stage1b_context_only_no_gt_box_vnewtrain0705_v2v.py" \
    --weights-root "${TRAIN0705_ROOT}/step-002500" \
    --input-json-list-path "${list_path}" \
    --model-name "train_stage1b_diffsynth_native0705_0705_${dataset_tag}" \
    --output-root "/data/gaoya/AAA_test_video/0623/test/v2v/train_stage1b_diffsynth_native0705_0705_${dataset_tag}" \
    --num-inference-steps 40 \
    --seed 42

  echo "[5/5] train0705 step-007000"
  PYTHONPATH="${REPO}:${DIFFSYNTH}" \
  CUDA_VISIBLE_DEVICES=5 \
  "${PY}" \
  "${REPO}/code_vjepa_vggt/train0705/wan_stage1b_context_only_no_gt_box_vnewtrain0705_v2v.py" \
    --weights-root "${TRAIN0705_ROOT}/step-007000" \
    --input-json-list-path "${list_path}" \
    --model-name "train_stage1b_diffsynth_native0705_0705_${dataset_tag}" \
    --output-root "/data/gaoya/AAA_test_video/0623/test/v2v/train_stage1b_diffsynth_native0705_0705_${dataset_tag}" \
    --num-inference-steps 40 \
    --seed 42

  echo "[done] dataset=${dataset_tag}"
}

run_dataset "morpheus_real_world" "${LIST_MORPHEUS}"
run_dataset "physicIQ" "${LIST_PHYSICIQ}"

echo "[all_done]"
