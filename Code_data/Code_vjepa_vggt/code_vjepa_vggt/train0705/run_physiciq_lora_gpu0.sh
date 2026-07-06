#!/usr/bin/env bash
set -euo pipefail

# Run physicIQ LoRA methods on gpu0 in sequence.
# This script is intended to be launched inside a tmux window.

PY=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
REPO=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
TRAIN0419=/home/gaoya/Code_Video/Code_data/Code_train/train_0419

LIST_PHYSICIQ=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt
OPENVID_LORA_ROOT=/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora/checkpoints/step-010000
PYBULLET_LORA_ROOT=/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500

RESULT_BASE=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/physicIQ
RUNTIME_BASE=/data/gaoya/agent-data/outputs/train0705_formal_compare_runtime_20260705/physicIQ

mkdir -p "${RESULT_BASE}" "${RUNTIME_BASE}"

echo "[start] physicIQ openvid_lora_10000 on gpu0"
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="${REPO}:${DIFFSYNTH}:${TRAIN0419}" \
"${PY}" \
  "${REPO}/code_vjepa_vggt/AAAinfer/wan_openvid_lorav2v.py" \
  --weights-root "${OPENVID_LORA_ROOT}" \
  --input-json-list-path "${LIST_PHYSICIQ}" \
  --model-name "wan_openvid_lorav2v_step10000_physicIQ" \
  --output-root "${RESULT_BASE}/loramodel/wan_openvid_lorav2v_step10000" \
  --runtime-root "${RUNTIME_BASE}/loramodel/wan_openvid_lorav2v_step10000_runtime" \
  --num-frames 25 \
  --num-inference-steps 40 \
  --cfg-scale 5.0 \
  --seed 42

echo "[start] physicIQ 0613pybullet_lora_000500 on gpu0"
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="${REPO}:${DIFFSYNTH}:${TRAIN0419}" \
"${PY}" \
  "${REPO}/code_vjepa_vggt/AAAinfer/wan_openvid_0613pybullet_lorav2v.py" \
  --weights-root "${PYBULLET_LORA_ROOT}" \
  --input-json-list-path "${LIST_PHYSICIQ}" \
  --model-name "wan_openvid_0613pybullet_lorav2v_step000500_physicIQ" \
  --output-root "${RESULT_BASE}/loramodel/wan_openvid_0613pybullet_lorav2v_step000500" \
  --runtime-root "${RUNTIME_BASE}/loramodel/wan_openvid_0613pybullet_lorav2v_step000500_runtime" \
  --num-frames 25 \
  --num-inference-steps 40 \
  --cfg-scale 5.0 \
  --seed 42

echo "[done] physicIQ LoRA sequence finished"
