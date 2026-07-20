#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 CHECKPOINT_DIR [GPU_ID] [OUTPUT_ROOT]" >&2
  exit 2
fi

CHECKPOINT_DIR="$1"
GPU_ID="${2:-1}"
OUTPUT_ROOT="${3:-/data/gaoya/agent-data/outputs/xssc_slot_perturb_ablation}"
MODES="${MODES:-none zero shuffle_slot shuffle_time noise drop_slot}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
TEST_LIST="${TEST_LIST:-/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for MODE in ${MODES}; do
  MODE_OUTPUT_ROOT="${OUTPUT_ROOT}/${MODE}"
  echo "[slot-ablation] mode=${MODE} output=${MODE_OUTPUT_ROOT}"
  XSSC_SLOT_PERTURB="${MODE}" \
  XSSC_SLOT_PERTURB_SEED="${XSSC_SLOT_PERTURB_SEED:-1234}" \
  XSSC_SLOT_NOISE_STD="${XSSC_SLOT_NOISE_STD:-1.0}" \
  XSSC_SLOT_DROP_PROB="${XSSC_SLOT_DROP_PROB:-0.5}" \
  TEST_LIST="${TEST_LIST}" \
  NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}" \
  STEP_OUTPUT_DIR_NAME="$(basename "${CHECKPOINT_DIR}")_${MODE}" \
  "${SCRIPT_DIR}/run_infer_xssc_context_slots.sh" \
    "${CHECKPOINT_DIR}" \
    "${GPU_ID}" \
    "${MODE_OUTPUT_ROOT}"
done
