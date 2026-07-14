#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box
RUNNER="${ROOT}/run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh"
INFER_SCRIPT="${ROOT}/visualize_text_noun_attention_x0_v2v.py"
CASE_LIST="${CASE_LIST:-/data/gaoya/agent-data/cache/text_noun_attention_physiq025_static_case.txt}"
GPU_PAIR="${GPU_PAIR:-2,2}"
SUPPRESS_SCALE="${SUPPRESS_SCALE:-0.25}"
SUPPRESS_TAG="${SUPPRESS_TAG:-scale025}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/agent-data/outputs/AAA_physv/block_wrong_region_attention_suppression_step1000_20260714/${SUPPRESS_TAG}}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_raw49f_kubric_openvid_replay_sourceaware_fp32gate_fixedctx8_init3500_save500_keepall_20260713T090024Z/checkpoints/step-001000}"

GPU_PAIR="${GPU_PAIR}" \
AUTO_SPLIT_INPUT=0 \
TEST_JSON_TXT="${CASE_LIST}" \
WEIGHTS_ROOT="${WEIGHTS_ROOT}" \
METHOD_NAME="block_wrong_region_suppress_${SUPPRESS_TAG}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
OUTPUT_FRAMES=49 \
CTX=8 \
NUM_INFERENCE_STEPS=40 \
CFG_SCALE=5.0 \
SEED=42 \
COMPACT_OBJECT_CONTEXT_SLOTS=1 \
OBJECT_ADAPTER_MLP_RESIDUAL_MAX_RATIO=3.0 \
OBJECT_BRANCH_RATIO_GUARD_MAX_RATIO=0.30 \
OBJECT_BRANCH_RATIO_GUARD_MAX_BLOCK_ID=-1 \
INFER_SCRIPT_OVERRIDE="${INFER_SCRIPT}" \
EXTRA_INFER_ARGS="--attention-capture-progress-indices auto5 --attention-query-chunk 256 --attention-save-all-maps 1 --attention-shared-context-future-scale 1 --attention-noun-suppress-name block --attention-noun-suppress-layers 10,11,13 --attention-noun-suppress-scale ${SUPPRESS_SCALE} --attention-noun-suppress-rect 0.05,0.28,0.0,0.35" \
bash "${RUNNER}"
