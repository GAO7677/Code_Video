#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  cat >&2 <<'EOF'
Usage:
  run_physrvg_full_sa_vjepa_verified.sh CHECKPOINT_DIR GPU_ID OUTPUT_ROOT [SHARD_INDEX] [SHARD_COUNT]

Environment:
  TEST_LIST       Exact P0 input list; defaults to the shared 198-case BPP list.
  CASE_LIMIT      Optional positive prefix limit for smoke tests.
  FORCE_INFERENCE Set to 1 to overwrite existing per-case outputs; default 1.
EOF
  exit 2
fi

CHECKPOINT_DIR="$(realpath "$1")"
GPU_ID="$2"
OUTPUT_ROOT="$(realpath -m "$3")"
SHARD_INDEX="${4:-0}"
SHARD_COUNT="${5:-1}"

TEST_LIST="${TEST_LIST:-/data/gaoya/AAA_test_video/0623/test/physicsiq/physicsiq_verified/inputs/bpp/verified_v2v_bpp_198.txt}"
CASE_LIMIT="${CASE_LIMIT:-}"
FORCE_INFERENCE="${FORCE_INFERENCE:-1}"
MODEL_ID="${MODEL_ID:-/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B-Diffusers}"
PHYSRVG_DIT_CHECKPOINT="${PHYSRVG_DIT_CHECKPOINT:-/data/gaoya/agent-data/weights/physrvg-diffusers-d8caf2/dit/diffusion_pytorch_model.safetensors}"
REPO_ROOT=/home/gaoya/code_V2V_baselines/PhysRVG-main
PYTHON=/data/gaoya/agent-data/envs/physrvg-full-sa/bin/python
INFER_SCRIPT="${REPO_ROOT}/scripts_mytrain/infer_full_sa_lora_json_list.py"

[[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo "GPU_ID must be an integer" >&2; exit 2; }
[[ "$GPU_ID" != "4" ]] || { echo "GPU 4 is prohibited by workspace rules." >&2; exit 2; }
[[ "$SHARD_INDEX" =~ ^[0-9]+$ && "$SHARD_COUNT" =~ ^[1-9][0-9]*$ ]] || {
  echo "SHARD_INDEX and SHARD_COUNT must be non-negative/positive integers" >&2
  exit 2
}
(( SHARD_INDEX < SHARD_COUNT )) || { echo "SHARD_INDEX must be less than SHARD_COUNT" >&2; exit 2; }
[[ -s "${CHECKPOINT_DIR}/adapter_model.safetensors" && -s "${CHECKPOINT_DIR}/adapter_config.json" ]] || {
  echo "Incomplete PEFT adapter checkpoint: ${CHECKPOINT_DIR}" >&2
  exit 2
}
[[ -s "$TEST_LIST" ]] || { echo "TEST_LIST not found or empty: $TEST_LIST" >&2; exit 2; }
[[ -s "$PHYSRVG_DIT_CHECKPOINT" ]] || {
  echo "PhysRVG DiT checkpoint not found: $PHYSRVG_DIT_CHECKPOINT" >&2
  exit 2
}

mkdir -p "$OUTPUT_ROOT"
FORCE_ARGS=()
if [[ "$FORCE_INFERENCE" == "1" ]]; then
  FORCE_ARGS=(--force)
fi
LIMIT_ARGS=()
if [[ -n "$CASE_LIMIT" ]]; then
  [[ "$CASE_LIMIT" =~ ^[1-9][0-9]*$ ]] || { echo "CASE_LIMIT must be positive" >&2; exit 2; }
  LIMIT_ARGS=(--limit "$CASE_LIMIT")
fi

echo "checkpoint=${CHECKPOINT_DIR}"
echo "gpu=${GPU_ID}"
echo "test_list=${TEST_LIST}"
echo "output_root=${OUTPUT_ROOT}"
echo "shard=${SHARD_INDEX}/${SHARD_COUNT}"
echo "protocol=Physics-IQ-Verified P0"
echo "condition=72 frames @ 24 FPS"
echo "output=189 frames @ 24 FPS; dynamic-effective 72-frame condition mask"
echo "steps=40 guidance=5 seed=42"

exec env \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="${REPO_ROOT}" \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PYTHON" "$INFER_SCRIPT" \
  --input-json-list "$TEST_LIST" \
  --output-root "$OUTPUT_ROOT" \
  --model-id "$MODEL_ID" \
  --physrvg-dit-checkpoint "$PHYSRVG_DIT_CHECKPOINT" \
  --lora-checkpoint "$CHECKPOINT_DIR" \
  --device cuda:0 \
  --height 512 \
  --width 896 \
  --num-frames 189 \
  --fps 24 \
  --num-inference-steps 40 \
  --guidance-scale 5 \
  --seed 42 \
  --context-frames 72 \
  --context-mask-mode dynamic_effective \
  --shard-index "$SHARD_INDEX" \
  --shard-count "$SHARD_COUNT" \
  --flat-output \
  "${LIMIT_ARGS[@]}" \
  "${FORCE_ARGS[@]}"
