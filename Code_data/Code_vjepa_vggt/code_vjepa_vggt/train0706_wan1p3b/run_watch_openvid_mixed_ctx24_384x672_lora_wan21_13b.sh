#!/usr/bin/env bash
set -euo pipefail

# OpenVid watcher for the Wan2.1-1.3B mixed_ctx24 training run.
# It watches new step-* checkpoints under the openvid training output root,
# runs inference on /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt,
# and then triggers bench.sh once gpu0/1/2 are idle.
#
# Run:
#   sh /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0706_wan1p3b/run_watch_openvid_mixed_ctx24_384x672_lora_wan21_13b.sh

PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
PROJECT_ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
WATCHER_SCRIPT="${PROJECT_ROOT}/code_vjepa_vggt/train0706_wan1p3b/watch_stage1b_context_only_no_gt_box_vnewtrain0705.py"

CHECKPOINT_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints_wan21_13b/openvid_mixed_ctx24_384x672_lora/checkpoints
INPUT_JSON_LIST_PATH=/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt
RESULT_ROOT=/data/gaoya/AAA_test_video/0623/test/v2v_1p3b
MODEL_NAME=openvid_mixed_ctx24_384x672_lora

exec env \
  PYTHONPATH="${PROJECT_ROOT}:${DIFFSYNTH_ROOT}" \
  "${PYTHON_BIN}" \
  "${WATCHER_SCRIPT}" \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --input-json-list-path "${INPUT_JSON_LIST_PATH}" \
  --result-root "${RESULT_ROOT}" \
  --model-name "${MODEL_NAME}" \
  --infer-script "${PROJECT_ROOT}/code_vjepa_vggt/train0706_wan1p3b/wan_openvid_lorav2v_1p3b.py" \
  --bench-script "${PROJECT_ROOT}/code_vjepa_vggt/AAAinfer/bench.sh" \
  --project-root "${PROJECT_ROOT}" \
  --diffsynth-root "${DIFFSYNTH_ROOT}" \
  --watch-state-dirname ".watch_openvid_mixed_ctx24_384x672_lora_wan21_13b" \
  --infer-gpu 2 \
  --bench-gpu 0 \
  --bench-idle-gpus 0,1,2 \
  --num-inference-steps 40
