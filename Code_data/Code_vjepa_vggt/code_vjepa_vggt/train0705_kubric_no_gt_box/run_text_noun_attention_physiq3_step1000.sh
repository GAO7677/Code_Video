#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box
RUNNER="${ROOT}/run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh"
INFER_SCRIPT="${ROOT}/visualize_text_noun_attention_x0_v2v.py"
CASE_LIST="${CASE_LIST:-${ROOT}/context_guidance_physiq_cases.txt}"
GPU_PAIR="${GPU_PAIR:-0,0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/agent-data/outputs/AAA_physv/text_noun_attention_step1000_physiq3_20260714}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_raw49f_kubric_openvid_replay_sourceaware_fp32gate_fixedctx8_init3500_save500_keepall_20260713T090024Z/checkpoints/step-001000}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
ATTENTION_CAPTURE_PROGRESS_INDICES="${ATTENTION_CAPTURE_PROGRESS_INDICES:-auto5}"
ATTENTION_QUERY_CHUNK="${ATTENTION_QUERY_CHUNK:-256}"
METHOD_NAME="${METHOD_NAME:-text_noun_attention_step1000}"

GPU_PAIR="${GPU_PAIR}" \
TEST_JSON_TXT="${CASE_LIST}" \
WEIGHTS_ROOT="${WEIGHTS_ROOT}" \
METHOD_NAME="${METHOD_NAME}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
OUTPUT_FRAMES=49 \
CTX=8 \
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}" \
CFG_SCALE=5.0 \
SEED=42 \
COMPACT_OBJECT_CONTEXT_SLOTS=1 \
OBJECT_ADAPTER_MLP_RESIDUAL_MAX_RATIO=3.0 \
OBJECT_BRANCH_RATIO_GUARD_MAX_RATIO=0.30 \
OBJECT_BRANCH_RATIO_GUARD_MAX_BLOCK_ID=-1 \
INFER_SCRIPT_OVERRIDE="${INFER_SCRIPT}" \
EXTRA_INFER_ARGS="--attention-capture-progress-indices ${ATTENTION_CAPTURE_PROGRESS_INDICES} --attention-query-chunk ${ATTENTION_QUERY_CHUNK}" \
bash "${RUNNER}"
