#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0715_scheme_d_object_tube_resampler
RUN_ROOT="${RUN_ROOT:-/data/gaoya/agent-data/checkpoints/train_stage1b_scheme_d_v3_object_tube_fresh_20260715T174459Z}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-000500}"
CHECKPOINT_DIR="${RUN_ROOT}/checkpoints/step-${CHECKPOINT_STEP}"
INPUT_JSON_LIST="${INPUT_JSON_LIST:-/data/gaoya/agent-data/outputs/AAA_physv/entity_id_binding_physiq3_current_20260714/input_jsons.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/agent-data/outputs/AAA_physv/scheme_d_v3_object_tube_checkpoint_val_20260715/step-${CHECKPOINT_STEP}}"
GPU_PAIR="${GPU_PAIR:-4}"
INFERENCE_DEVICES="${INFERENCE_DEVICES:-cuda:0,cuda:0}"
POLL_SECONDS="${POLL_SECONDS:-60}"

while [[ ! -s "${CHECKPOINT_DIR}/checkpoint.safetensors" || ! -s "${CHECKPOINT_DIR}/training_state.pt" ]]; do
  printf '%s waiting for %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${CHECKPOINT_DIR}"
  sleep "${POLL_SECONDS}"
done

mkdir -p "${OUTPUT_ROOT}"

run_variant() {
  local name="$1"
  local ablation="$2"
  local residual_scale="$3"
  local variant_root="${OUTPUT_ROOT}/${name}"
  mkdir -p "${variant_root}"
  printf '%s starting %s ablation=%s residual_scale=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${name}" "${ablation}" "${residual_scale}" \
    | tee -a "${OUTPUT_ROOT}/validation_status.log"
  env \
    GPU_PAIR="${GPU_PAIR}" \
    INFERENCE_DEVICES="${INFERENCE_DEVICES}" \
    WEIGHTS_ROOT="${CHECKPOINT_DIR}" \
    INPUT_JSON_LIST="${INPUT_JSON_LIST}" \
    MODEL_NAME="scheme_d_v3_step${CHECKPOINT_STEP}_${name}" \
    OUTPUT_ROOT="${variant_root}" \
    OBJECT_CONTEXT_ABLATION="${ablation}" \
    OBJECT_BRANCH_RESIDUAL_SCALE="${residual_scale}" \
    NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}" \
    bash "${PROJECT}/run_infer.sh" 2>&1 | tee "${variant_root}/inference.log"
  /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
    "${PROJECT}/validate_inference_summary.py" "${variant_root}/summary.json" \
    | tee -a "${variant_root}/inference.log"
  printf '%s completed %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${name}" \
    | tee -a "${OUTPUT_ROOT}/validation_status.log"
}

run_variant baseline none 1.0
run_variant no_object_context zero 1.0
run_variant object_residual_1p5x none 1.5

/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  "${PROJECT}/compare_validation_variants.py" \
  --baseline-dir "${OUTPUT_ROOT}/baseline/results" \
  --variant no_object_context "${OUTPUT_ROOT}/no_object_context/results" \
  --variant object_residual_1p5x "${OUTPUT_ROOT}/object_residual_1p5x/results" \
  --context-frames 8 \
  --output "${OUTPUT_ROOT}/variant_pixel_metrics.json"

printf '%s all checkpoint validation variants completed\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  | tee -a "${OUTPUT_ROOT}/validation_status.log"
