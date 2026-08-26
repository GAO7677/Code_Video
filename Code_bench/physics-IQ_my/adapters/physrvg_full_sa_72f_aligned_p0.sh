#!/usr/bin/env bash
# Generic Full-SA PhysRVG adapter. It uses the same direct in-memory 189->120
# encoder for every PEFT LoRA. With no argument it selects the latent-mask
# step-001000 checkpoint; an explicit argument selects another Full-SA LoRA.
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: physrvg_full_sa_72f_aligned_p0.sh [LORA_CHECKPOINT]" >&2
  exit 2
fi

: "${PHYSIQ_RESULT_FILE:?run_physicsiq_p0.py must set PHYSIQ_RESULT_FILE}"
: "${PHYSIQ_INPUT_LIST:?run_physicsiq_p0.py must set PHYSIQ_INPUT_LIST}"
: "${PHYSIQ_RAW_ROOT:?run_physicsiq_p0.py must set PHYSIQ_RAW_ROOT}"
: "${PHYSIQ_SUBMISSION_ROOT:?run_physicsiq_p0.py must set PHYSIQ_SUBMISSION_ROOT}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/../PhysRVG/run_full_sa_latent_mask_72f_aligned.py"
PYTHON="${PHYSIQ_PHYSRVG_PYTHON:-/data/gaoya/agent-data/envs/physrvg-full-sa/bin/python}"
MODEL_ID="${PHYSIQ_MODEL_ID:-/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B-Diffusers}"
DIT_CHECKPOINT="${PHYSIQ_DIT_CHECKPOINT:-/data/gaoya/agent-data/weights/physrvg-diffusers-d8caf2/dit/diffusion_pytorch_model.safetensors}"
LORA_CHECKPOINT="${1:-${PHYSIQ_LORA_CHECKPOINT:-/data/gaoya/agent-data/checkpoints/physrvg_full_sa_latent_mask/full-sa-pybullet-physrvg-latent-mask-b2-gacc2-20260818T052732Z/checkpoints/step-001000}}"
GPU_ID="${PHYSIQ_GPU_ID:?GPU id was not supplied}"

[[ "$GPU_ID" != 4 ]] || {
  echo "GPU 4 is prohibited by the workspace rules" >&2
  exit 2
}
[[ -x "$PYTHON" ]] || { echo "PhysRVG Python not found: $PYTHON" >&2; exit 2; }
[[ -f "$RUNNER" ]] || { echo "runner not found: $RUNNER" >&2; exit 2; }

export PYTHONNOUSERSITE=1
export PYTHONPATH="${PHYSIQ_PHYSRVG_ROOT:-/home/gaoya/code_V2V_baselines/PhysRVG-main}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EXTRA_ARGS=()
if [[ "${PHYSIQ_FORCE:-0}" == 1 ]]; then
  EXTRA_ARGS+=(--force)
fi

"$PYTHON" "$RUNNER" \
  --input-json-list "$PHYSIQ_INPUT_LIST" \
  --raw-output-root "$PHYSIQ_RAW_ROOT" \
  --submission-output-root "$PHYSIQ_SUBMISSION_ROOT" \
  --manifest-path "$PHYSIQ_ENCODING_MANIFEST" \
  --model-id "$MODEL_ID" \
  --physrvg-dit-checkpoint "$DIT_CHECKPOINT" \
  --lora-checkpoint "$LORA_CHECKPOINT" \
  --device cuda:0 \
  --height "$PHYSIQ_HEIGHT" \
  --width "$PHYSIQ_WIDTH" \
  --condition-frames "$PHYSIQ_CONDITION_FRAMES" \
  --condition-fps "$PHYSIQ_CONDITION_FPS" \
  --num-frames "$PHYSIQ_RAW_FRAMES" \
  --fps "$PHYSIQ_FPS" \
  --num-inference-steps "$PHYSIQ_NUM_INFERENCE_STEPS" \
  --guidance-scale "$PHYSIQ_GUIDANCE_SCALE" \
  --seed "$PHYSIQ_SEED" \
  --context-mask-mode "$PHYSIQ_CONTEXT_MASK_MODE" \
  --shard-index 0 \
  --shard-count 1 \
  "${EXTRA_ARGS[@]}"

printf '%s\n' "$PHYSIQ_SUBMISSION_ROOT" >"$PHYSIQ_RESULT_FILE"
