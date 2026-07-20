#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/data/gaoya/agent-data/checkpoints/train_xssc_context_slots/formal_mix49_b2_dropout_metrics_20260719T204359Z/checkpoints/step-001500}"
GPU_ID="${1:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/agent-data/outputs/AAA_physv/AAA_xSSC/xssc_preprocess_ablation_step1500}"
TEST_LIST="${TEST_LIST:-/data/gaoya/agent-data/outputs/AAA_physv/AAA_xSSC/xssc_preprocess_ablation_step1500/cases_4.txt}"
MODES="${MODES:-center_crop left_crop right_crop resize_pad_square}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"

PROJ=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
XSSC_ROOT="${XSSC_ROOT:-/home/gaoya/Code_Video/xSSC-main}"
XSSC_CONFIG="${XSSC_CONFIG:-${XSSC_ROOT}/config-randsfq/rsfq2_r-ytvis.py}"
XSSC_CHECKPOINT="${XSSC_CHECKPOINT:-/data/gaoya/ckpt/xSSC/rsfq2_r-ytvis/42-0130.pth}"

mkdir -p "${OUTPUT_ROOT}" "$(dirname "${TEST_LIST}")"
if [ ! -s "${TEST_LIST}" ]; then
  cat > "${TEST_LIST}" <<'EOF'
/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed-ball-and-block-fall_motion_to_end.json
/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed.json
/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/physicIQ_026_Solid_Mechanics_0005_perspective-center_trimmed-ball-behind-rotating-paper.json
/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed_crop_top60px.json
EOF
fi

for MODE in ${MODES}; do
  echo "[xssc-preprocess-ablation] mode=${MODE} checkpoint=${CHECKPOINT_DIR}"
  TRACE_ROOT="${OUTPUT_ROOT}/numeric_traces/step-001500_${MODE}"
  mkdir -p "${TRACE_ROOT}"
  env \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="${PROJ}:${DIFFSYNTH_ROOT}" \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    XSSC_ROOT="${XSSC_ROOT}" \
    XSSC_CONFIG="${XSSC_CONFIG}" \
    XSSC_CHECKPOINT="${XSSC_CHECKPOINT}" \
    XSSC_PREPROCESS_MODE="${MODE}" \
    "${PYTHON}" -m code_vjepa_vggt.train_xSSC.infer_xssc_context_slots \
    --weights-root "${CHECKPOINT_DIR}" \
    --input-json-list-path "${TEST_LIST}" \
    --model-name "xssc_ctx_slots_wan22_5b_${MODE}" \
    --output-root "${OUTPUT_ROOT}" \
    --step-output-dir-name "step001500_xssc_preprocess_${MODE}" \
    --device cuda:0 \
    --aux-device cuda:0 \
    --inference-devices cuda:0,cuda:0 \
    --height 512 \
    --width 896 \
    --num-frames 49 \
    --context-frames 8 \
    --sampling-mode prefix \
    --num-inference-steps "${NUM_INFERENCE_STEPS}" \
    --dump-numeric-trace-root "${TRACE_ROOT}" \
    --force
done
