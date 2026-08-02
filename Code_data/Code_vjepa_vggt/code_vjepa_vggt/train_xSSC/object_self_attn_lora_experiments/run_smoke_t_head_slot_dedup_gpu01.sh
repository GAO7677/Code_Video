#!/usr/bin/env bash
# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/run_smoke_t_head_slot_dedup_gpu01.sh
set -euo pipefail

# Keep PEFT/Accelerate on the mutually compatible versions installed inside
# wan-cu128 instead of mixing them with packages from ~/.local.
export PYTHONNOUSERSITE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
CONFIG="${CONFIG:-${SCRIPT_DIR}/configs/smoke_t_head_slot_dedup_merge_gpu01.json}"
RUN_TAG="${RUN_TAG:-smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
EXPERIMENT_NAME="smoke_common_t_head_full70_slot_dedup_merge_gpu01"
CHECKPOINT_ROOT="/data/gaoya/agent-data/checkpoints/xssc_object_self_attn_lora_smoke"
RUN_ROOT="${CHECKPOINT_ROOT}/${EXPERIMENT_NAME}/${RUN_TAG}"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints/step-000001"
CONTROL_ROOT="${CONTROL_ROOT:-/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_smoke/t_head_slot_dedup/${RUN_TAG}}"
TEST_SOURCE="/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt"
TEST_LIST="${CONTROL_ROOT}/test_5_first_case.txt"
INFER_ROOT="${CONTROL_ROOT}/inference"
INFER_NAME="${EXPERIMENT_NAME}_step-000001_steps2_512x896_ctx08_49f"
INFER_GPU="${INFER_GPU:-1}"

mkdir -p "${CONTROL_ROOT}" "${INFER_ROOT}"
sed -n '1p' "${TEST_SOURCE}" > "${TEST_LIST}"

echo "[smoke:train] config=${CONFIG} run_tag=${RUN_TAG} gpu=0,1"
bash "${SCRIPT_DIR}/run_train_slot_dedup_from_config.sh" \
  "${CONFIG}" \
  --run-tag "${RUN_TAG}"

if [[ ! -s "${CHECKPOINT_DIR}/checkpoint.safetensors" ]]; then
  echo "Missing smoke checkpoint: ${CHECKPOINT_DIR}/checkpoint.safetensors" >&2
  exit 1
fi
if [[ ! -s "${CHECKPOINT_DIR}/training_state.pt" ]]; then
  echo "Missing smoke training state: ${CHECKPOINT_DIR}/training_state.pt" >&2
  exit 1
fi

echo "[smoke:infer] checkpoint=${CHECKPOINT_DIR} gpu=${INFER_GPU}"
TEST_LIST="${TEST_LIST}" \
NUM_INFERENCE_STEPS=2 \
STEP_OUTPUT_DIR_NAME="${INFER_NAME}" \
bash "${SCRIPT_DIR}/run_infer_slot_dedup_checkpoint.sh" \
  "${CHECKPOINT_DIR}" \
  "${INFER_GPU}" \
  "${INFER_ROOT}"

"${PYTHON}" "${SCRIPT_DIR}/validate_t_head_slot_dedup_smoke.py" \
  --run-root "${RUN_ROOT}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --inference-dir "${INFER_ROOT}/${INFER_NAME}" \
  --expected-width 896 \
  --expected-height 512 \
  --expected-frames 49 \
  --expected-heads 70 \
  --report "${CONTROL_ROOT}/inference_validation.json"

echo "[smoke:complete]"
echo "run_root=${RUN_ROOT}"
echo "checkpoint=${CHECKPOINT_DIR}"
echo "inference=${INFER_ROOT}/${INFER_NAME}"
