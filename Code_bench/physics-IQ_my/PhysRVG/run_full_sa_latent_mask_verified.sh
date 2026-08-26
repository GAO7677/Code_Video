#!/usr/bin/env bash
# Run the PhysRVG Full-SA latent-mask checkpoint under the effective Full-SA
# Physics-IQ Verified P0 configuration.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/physrvg_full_sa_latent_mask_verified.env"

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 2
}

usage() {
  cat <<'EOF'
Usage:
  run_full_sa_latent_mask_verified.sh preflight
  run_full_sa_latent_mask_verified.sh prepare
  run_full_sa_latent_mask_verified.sh smoke
  run_full_sa_latent_mask_verified.sh generate
  run_full_sa_latent_mask_verified.sh postprocess
  run_full_sa_latent_mask_verified.sh validate
  run_full_sa_latent_mask_verified.sh score
  run_full_sa_latent_mask_verified.sh all

The script uses the adjacent fixed .env configuration.  It does not accept
per-run inference overrides: changing 49/30, context 8, official_fixed2,
CFG, or global seed reset would create a different protocol.
EOF
}

[[ -r "$CONFIG_FILE" ]] || die "configuration file not found: $CONFIG_FILE"
# shellcheck disable=SC1090
source "$CONFIG_FILE"

sha256_of() {
  sha256sum "$1" | awk '{print $1}'
}

require_file() {
  [[ -f "$1" ]] || die "required file not found: $1"
}

require_nonempty_file() {
  [[ -s "$1" ]] || die "required file is missing or empty: $1"
}

assert_value() {
  local name="$1"
  local actual="$2"
  local expected="$3"
  [[ "$actual" == "$expected" ]] || die "fixed config mismatch: ${name}=${actual}, expected ${expected}"
}

validate_config() {
  assert_value P0_CASE_COUNT "$P0_CASE_COUNT" 198
  assert_value CONDITION_FRAMES "$CONDITION_FRAMES" 72
  assert_value CONDITION_FPS "$CONDITION_FPS" 24
  assert_value CONDITION_SECONDS "$CONDITION_SECONDS" 3.0
  assert_value HEIGHT "$HEIGHT" 512
  assert_value WIDTH "$WIDTH" 896
  assert_value RAW_FRAMES "$RAW_FRAMES" 189
  assert_value RAW_FPS "$RAW_FPS" 24
  assert_value NUM_INFERENCE_STEPS "$NUM_INFERENCE_STEPS" 40
  assert_value GUIDANCE_SCALE "$GUIDANCE_SCALE" 5.0
  assert_value SEED "$SEED" 42
  assert_value CONTEXT_FRAMES "$CONTEXT_FRAMES" 72
  assert_value CONTEXT_MASK_MODE "$CONTEXT_MASK_MODE" dynamic_effective
  assert_value DO_CFG "$DO_CFG" 0
  assert_value RESET_GLOBAL_SEED_PER_CASE "$RESET_GLOBAL_SEED_PER_CASE" 0
  [[ "$GPU_ID" =~ ^[0-9]+$ ]] || die "GPU_ID must be an integer"
  [[ "$GPU_ID" != 4 ]] || die "GPU 4 is prohibited by workspace rules"

  require_nonempty_file "$INPUT_LIST"
  require_file "$CHECKPOINT/adapter_model.safetensors"
  require_file "$CHECKPOINT/adapter_config.json"
  require_file "$PHYSRVG_DIT_CHECKPOINT"
  require_file "$MODEL_ID/model_index.json"
  require_file "$PYTHON"
  require_file "$INFER_SCRIPT"
  require_file "$PREPARE_INPUTS"
  require_file "$PREPARE_OUTPUTS"
  require_file "$VALIDATOR"
  require_file "$OFFICIAL_RUNNER"
  require_file "$AGGREGATOR"
  require_file "$DESCRIPTIONS_FILE"

  local input_lines
  input_lines="$(awk 'NF && $1 !~ /^#/ {count++} END {print count + 0}' "$INPUT_LIST")"
  [[ "$input_lines" == "$P0_CASE_COUNT" ]] || \
    die "P0 input list has ${input_lines} entries, expected ${P0_CASE_COUNT}"

  local input_hash
  input_hash="$(sha256_of "$INPUT_LIST")"
  [[ "$input_hash" == "$EXPECTED_INPUT_LIST_SHA256" ]] || \
    die "P0 input-list SHA256 mismatch: ${input_hash}"

  local lora_hash
  lora_hash="$(sha256_of "$CHECKPOINT/adapter_model.safetensors")"
  [[ "$lora_hash" == "$EXPECTED_LORA_SHA256" ]] || \
    die "LoRA SHA256 mismatch: ${lora_hash}"

  grep -q '"peft_type": "LORA"' "$CHECKPOINT/adapter_config.json" || \
    die "checkpoint is not a PEFT LoRA adapter"
  grep -q '"lora_alpha": 32.0' "$CHECKPOINT/adapter_config.json" || \
    die "unexpected LoRA alpha in adapter_config.json"
  grep -q '"r": 32' "$CHECKPOINT/adapter_config.json" || \
    die "unexpected LoRA rank in adapter_config.json"
}

print_protocol() {
  printf '%s\n' "run=${RUN_NAME}"
  printf '%s\n' "checkpoint=${CHECKPOINT}"
  printf '%s\n' "gpu=${GPU_ID}"
  printf '%s\n' "input_list=${INPUT_LIST}"
  printf '%s\n' "protocol=Physics-IQ-Verified P0"
  printf '%s\n' "condition=${CONDITION_FRAMES}@${CONDITION_FPS}fps/${CONDITION_SECONDS}s"
  printf '%s\n' "raw=${RAW_FRAMES}@${RAW_FPS}fps"
  printf '%s\n' "resolution=${HEIGHT}x${WIDTH}"
  printf '%s\n' "steps=${NUM_INFERENCE_STEPS} guidance=${GUIDANCE_SCALE} seed=${SEED}"
  printf '%s\n' "context=${CONTEXT_FRAMES} mask=${CONTEXT_MASK_MODE} cfg=${DO_CFG} global_seed_reset=${RESET_GLOBAL_SEED_PER_CASE}"
}

full_normalized_list() {
  printf '%s\n' "${CACHE_ROOT}/case198/verified_v2v_bpp_198.txt"
}

smoke_normalized_list() {
  printf '%s\n' "${CACHE_ROOT}/case1/verified_v2v_bpp_1.txt"
}

prepare() {
  mkdir -p "$CACHE_ROOT"
  "$PYTHON" "$PREPARE_INPUTS" \
    --input-list "$INPUT_LIST" \
    --output-root "$CACHE_ROOT/case198"
  "$PYTHON" "$PREPARE_INPUTS" \
    --input-list "$INPUT_LIST" \
    --output-root "$CACHE_ROOT/case1" \
    --limit 1
  require_nonempty_file "$(full_normalized_list)"
  require_nonempty_file "$(smoke_normalized_list)"
}

run_inference() {
  local input_json_list="$1"
  local output_root="$2"
  local force="$3"
  local force_args=()
  if [[ "$force" == 1 ]]; then
    force_args=(--force)
  fi

  mkdir -p "$output_root"
  env \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH="$REPO_ROOT" \
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PYTHON" "$INFER_SCRIPT" \
    --input-json-list "$input_json_list" \
    --output-root "$output_root" \
    --model-id "$MODEL_ID" \
    --physrvg-dit-checkpoint "$PHYSRVG_DIT_CHECKPOINT" \
    --lora-checkpoint "$CHECKPOINT" \
    --device cuda:0 \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --num-frames "$RAW_FRAMES" \
    --fps "$RAW_FPS" \
    --num-inference-steps "$NUM_INFERENCE_STEPS" \
    --guidance-scale "$GUIDANCE_SCALE" \
    --negative-prompt "$NEGATIVE_PROMPT" \
    --context-frames "$CONTEXT_FRAMES" \
    --context-mask-mode "$CONTEXT_MASK_MODE" \
    --seed "$SEED" \
    --shard-index 0 \
    --shard-count 1 \
    --flat-output \
    "${force_args[@]}"
}

count_mp4() {
  find "$1" -maxdepth 1 -type f -name '*.mp4' -print | wc -l
}

smoke() {
  require_nonempty_file "$(smoke_normalized_list)"
  run_inference "$(smoke_normalized_list)" "$SMOKE_ROOT" 0
  local count
  count="$(count_mp4 "$SMOKE_ROOT")"
  [[ "$count" == 1 ]] || die "smoke output has ${count} MP4 files, expected 1"
  local metadata
  metadata="$(find "$SMOKE_ROOT" -maxdepth 1 -type f -name '*.json' -print | head -n 1)"
  require_nonempty_file "$metadata"
  rg -q '"num_frames": 189' "$metadata" || die "smoke metadata frame count mismatch"
  rg -q '"fps": 24' "$metadata" || die "smoke metadata FPS mismatch"
  rg -q '"context_mask_mode": "dynamic_effective"' "$metadata" || die "smoke mask mismatch"
  rg -q '"classifier_free_guidance_enabled": false' "$metadata" || die "smoke CFG mismatch"
  rg -q '"global_seed_reset_per_case": false' "$metadata" || die "smoke seed-reset mismatch"
  printf '%s\n' "smoke=PASS output=${SMOKE_ROOT}"
}

generate() {
  require_nonempty_file "$(full_normalized_list)"
  run_inference "$(full_normalized_list)" "$RAW_ROOT" "$FORCE_INFERENCE"
  local count
  count="$(count_mp4 "$RAW_ROOT")"
  [[ "$count" == "$P0_CASE_COUNT" ]] || \
    die "raw output has ${count} MP4 files, expected ${P0_CASE_COUNT}"
  printf '%s\n' "raw=PASS output=${RAW_ROOT} videos=${count}"
}

postprocess() {
  [[ -d "$RAW_ROOT" ]] || die "raw output directory not found: $RAW_ROOT"
  [[ "$(count_mp4 "$RAW_ROOT")" == "$P0_CASE_COUNT" ]] || \
    die "raw output is incomplete"
  "$PYTHON" "$PREPARE_OUTPUTS" \
    --raw-folder "$RAW_ROOT" \
    --input-list "$INPUT_LIST" \
    --output-folder "$SUBMISSION_ROOT" \
    --force
  [[ "$(count_mp4 "$SUBMISSION_ROOT")" == "$P0_CASE_COUNT" ]] || \
    die "submission output is incomplete"
  printf '%s\n' "submission=PASS output=${SUBMISSION_ROOT}"
}

validate() {
  "$PYTHON" "$VALIDATOR" \
    --descriptions-file "$DESCRIPTIONS_FILE" \
    "$SUBMISSION_ROOT"
}

score() {
  mkdir -p "$EVALUATION_ROOT"
  export PATH="${OFFICIAL_ENV_BIN}:${PATH}"
  export UV_CACHE_DIR
  export UV_PROJECT_ENVIRONMENT
  bash "$OFFICIAL_RUNNER" \
    --n-process "$OFFICIAL_N_PROCESS" \
    --output-folder "$EVALUATION_ROOT" \
    --descriptions-file "$DESCRIPTIONS_FILE" \
    "$SUBMISSION_ROOT"

  local result_csv="${EVALUATION_ROOT}/physics-IQ-benchmark-verified/results/${RUN_NAME}.csv"
  require_nonempty_file "$result_csv"
  bash "$AGGREGATOR" \
    "$result_csv" \
    --save-csv "${EVALUATION_ROOT}/${RUN_NAME}_verified_summary.csv" \
    --model-name "$RUN_NAME"
}

COMMAND="${1:-help}"
if [[ "$COMMAND" == help || "$COMMAND" == --help || "$COMMAND" == -h ]]; then
  usage
  exit 0
fi

validate_config
print_protocol

case "$COMMAND" in
  preflight)
    printf '%s\n' "preflight=PASS"
    ;;
  prepare)
    prepare
    ;;
  smoke)
    smoke
    ;;
  generate)
    generate
    ;;
  postprocess)
    postprocess
    ;;
  validate)
    validate
    ;;
  score)
    score
    ;;
  all)
    prepare
    generate
    postprocess
    validate
    score
    ;;
  *)
    usage >&2
    die "unknown command: ${COMMAND}"
    ;;
esac
