#!/usr/bin/env bash
set -euo pipefail

PY=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
REPO=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
WAN22=/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main
TRAIN0419=/home/gaoya/Code_Video/Code_data/Code_train/train_0419

LIST=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt
OPENVID_LORA_ROOT=/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora/checkpoints/step-010000
PYBULLET_LORA_ROOT=/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500

NEGATIVE_PROMPT="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

LOGROOT=/data/gaoya/agent-data/outputs/train0705_formal_compare_aligned49_logs
mkdir -p "${LOGROOT}"

echo "[start] physicIQ aligned49 gpu1/2/3"

CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH="${REPO}:${WAN22}" \
"${PY}" \
  "${REPO}/code_vjepa_vggt/AAAinfer/wanti2v.py" \
  --input-list "${LIST}" \
  --model-name wan2p2_ti2v5B_physicIQ_aligned49 \
  --output-root /data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/physicIQ/basemodel/wan2p2_ti2v5B_aligned49 \
  --size 512*896 \
  --frame-num 49 \
  --sampling-steps 40 \
  --cfg-scale 5.0 \
  --fps 30 \
  --seed 42 \
  --negative-prompt "${NEGATIVE_PROMPT}" \
  --offload-model \
  > "${LOGROOT}/physicIQ_wan2p2_ti2v5B_aligned49.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 \
PYTHONPATH="${REPO}:${DIFFSYNTH}:${TRAIN0419}" \
"${PY}" \
  "${REPO}/code_vjepa_vggt/AAAinfer/wan_openvid_lorav2v.py" \
  --weights-root "${OPENVID_LORA_ROOT}" \
  --input-json-list-path "${LIST}" \
  --model-name wan_openvid_lorav2v_step10000_physicIQ_aligned49 \
  --output-root /data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/physicIQ/loramodel/wan_openvid_lorav2v_step10000_aligned49 \
  --runtime-root /data/gaoya/agent-data/outputs/train0705_formal_compare_aligned49_runtime/physicIQ/loramodel/wan_openvid_lorav2v_step10000_aligned49_runtime \
  --height 512 \
  --width 896 \
  --num-frames 49 \
  --num-inference-steps 40 \
  --cfg-scale 5.0 \
  --fps 30 \
  --seed 42 \
  --negative-prompt "${NEGATIVE_PROMPT}" \
  > "${LOGROOT}/physicIQ_wan_openvid_lorav2v_step10000_aligned49.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 \
PYTHONPATH="${REPO}:${DIFFSYNTH}:${TRAIN0419}" \
"${PY}" \
  "${REPO}/code_vjepa_vggt/AAAinfer/wan_openvid_0613pybullet_lorav2v.py" \
  --weights-root "${PYBULLET_LORA_ROOT}" \
  --input-json-list-path "${LIST}" \
  --model-name wan_openvid_0613pybullet_lorav2v_step000500_physicIQ_aligned49 \
  --output-root /data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/physicIQ/loramodel/wan_openvid_0613pybullet_lorav2v_step000500_aligned49 \
  --runtime-root /data/gaoya/agent-data/outputs/train0705_formal_compare_aligned49_runtime/physicIQ/loramodel/wan_openvid_0613pybullet_lorav2v_step000500_aligned49_runtime \
  --height 512 \
  --width 896 \
  --num-frames 49 \
  --num-inference-steps 40 \
  --cfg-scale 5.0 \
  --fps 30 \
  --seed 42 \
  --negative-prompt "${NEGATIVE_PROMPT}" \
  > "${LOGROOT}/physicIQ_wan_openvid_0613pybullet_lorav2v_step000500_aligned49.log" 2>&1 &

wait

echo "[done] physicIQ aligned49 gpu1/2/3"
