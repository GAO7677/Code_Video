#!/usr/bin/env bash
# Run the Full-SA VJEPA checkpoint under the shared Physics-IQ P0 protocol.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIGINAL_SCRIPT="/home/gaoya/code_V2V_baselines/PhysRVG-main/scripts_mytrain/run_infer_full_sa_physrvg_vjepa.sh"
PREPARE_INPUTS="${SCRIPT_DIR}/prepare_full_sa_vjepa_verified_inputs.py"
PREPARE_OUTPUTS="${SCRIPT_DIR}/../xSSC/prepare_verified_outputs.py"
VALIDATOR="${SCRIPT_DIR}/../common/validate_verified_run.py"
OFFICIAL_RUNNER="${SCRIPT_DIR}/../run_verified_official.sh"
AGGREGATOR="${SCRIPT_DIR}/../aggregate_verified_official.sh"
PYTHON="/data/gaoya/agent-data/envs/physrvg-full-sa/bin/python"
OFFICIAL_ENV_BIN="${OFFICIAL_ENV_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin}"
FFPROBE="/home/gaoya/miniconda3/envs/wan-cu128/bin/ffprobe"
FFMPEG="/home/gaoya/miniconda3/envs/wan-cu128/bin/ffmpeg"

# The PhysRVG inference venv deliberately stays separate from the official
# Physics-IQ uv environment used for the shared P0 score.
export PATH="${OFFICIAL_ENV_BIN}:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/data/gaoya/agent-data/cache/uv}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/data/gaoya/agent-data/cache/envs/physics-iq-verified}"

WORKSPACE="${PHYSIQ_WORKSPACE:-/data/gaoya/AAA_test_video/0623/test/physicsiq/physicsiq_verified}"
INPUT_LIST="${PHYSIQ_P0_INPUT_LIST:-${WORKSPACE}/inputs/bpp/verified_v2v_bpp_198.txt}"
MODEL_ID="${MODEL_ID:-/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B-Diffusers}"
PHYSRVG_DIT_CHECKPOINT="${PHYSRVG_DIT_CHECKPOINT:-/data/gaoya/agent-data/weights/physrvg-diffusers-d8caf2/dit/diffusion_pytorch_model.safetensors}"
RUN_NAME="${RUN_NAME:-physrvg-full-sa-vjepa-step000500-bpp-run_01}"
CHECKPOINT_DEFAULT="/data/gaoya/agent-data/checkpoints/physrvg_full_sa_vjepa/full-sa-pybullet-physrvg-vjepa-b2-gacc2-ddp-sync-20260817T190000Z/checkpoints/step-000500"
CACHE_ROOT="${PHYSIQ_CACHE_ROOT:-/data/gaoya/agent-data/cache/physics-iq-verified/physrvg_full_sa_vjepa/${RUN_NAME}}"
OFFICIAL_N_PROCESS="${OFFICIAL_N_PROCESS:-0}"

usage() {
  cat <<'EOF'
Usage:
  run_full_sa_vjepa_verified.sh prepare [CASE_LIMIT]
  run_full_sa_vjepa_verified.sh generate CHECKPOINT GPU_ID SHARD_INDEX SHARD_COUNT [CASE_LIMIT]
  run_full_sa_vjepa_verified.sh merge SHARD_COUNT
  run_full_sa_vjepa_verified.sh postprocess
  run_full_sa_vjepa_verified.sh validate
  run_full_sa_vjepa_verified.sh score

P0 generation is fixed to 72-frame/24-FPS conditioning, 189 raw frames,
69-frame prefix removal, 120-frame/24-FPS submission, 40 steps,
guidance 5, seed 42, and dynamic-effective condition masking.
EOF
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 2
}

case_limit_or_default() {
  local value="${1:-198}"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "CASE_LIMIT must be a positive integer"
  ((value >= 1 && value <= 198)) || die "CASE_LIMIT must be between 1 and 198"
  printf '%s\n' "$value"
}

prepared_list() {
  local case_limit="$1"
  printf '%s\n' "${CACHE_ROOT}/case${case_limit}/verified_v2v_bpp_${case_limit}.txt"
}

prepare_inputs() {
  local case_limit="$1"
  local cache_dir="${CACHE_ROOT}/case${case_limit}"
  mkdir -p "$cache_dir"
  "$PYTHON" "$PREPARE_INPUTS" \
    --input-list "$INPUT_LIST" \
    --output-root "$cache_dir" \
    --limit "$case_limit"
  [[ -s "$(prepared_list "$case_limit")" ]] || die "prepared input list was not created"
}

generate() {
  local checkpoint="$(realpath "$1")"
  local gpu_id="$2"
  local shard_index="$3"
  local shard_count="$4"
  local case_limit="$(case_limit_or_default "${5:-198}")"
  [[ "$gpu_id" != "4" ]] || die "GPU 4 is prohibited by workspace rules"
  [[ -d "$checkpoint" ]] || die "checkpoint directory not found: $checkpoint"
  [[ -s "$checkpoint/adapter_model.safetensors" ]] || die "missing adapter_model.safetensors"
  [[ -s "$checkpoint/adapter_config.json" ]] || die "missing adapter_config.json"
  [[ "$shard_index" =~ ^[0-9]+$ && "$shard_count" =~ ^[1-9][0-9]*$ ]] || die "invalid shard"
  ((shard_index < shard_count)) || die "shard index must be less than shard count"
  ((case_limit == 198 || shard_count == 1)) || die "smoke generation must use SHARD_COUNT=1"

  prepare_inputs "$case_limit" >/dev/null
  local list_path="$(prepared_list "$case_limit")"
  local raw_parent="$WORKSPACE/raw_shards"
  local shard_tag="$(printf '%s_shard%02d-of-%02d' "$RUN_NAME" "$shard_index" "$shard_count")"
  mkdir -p "$raw_parent"

  printf 'P0 inference command: GPU=%s shard=%s/%s cases=%s\n' \
    "$gpu_id" "$shard_index" "$shard_count" "$case_limit"
  env \
    TEST_LIST="$list_path" \
    NUM_INFERENCE_STEPS=40 \
    NUM_FRAMES=189 \
    FPS=24 \
    GUIDANCE_SCALE=5 \
    CONTEXT_FRAMES=72 \
    CONTEXT_MASK_MODE=dynamic_effective \
    RESET_GLOBAL_SEED_PER_CASE=1 \
    SHARD_INDEX="$shard_index" \
    SHARD_COUNT="$shard_count" \
    FORCE_INFERENCE="${FORCE_INFERENCE:-0}" \
    MODEL_ID="$MODEL_ID" \
    PHYSRVG_DIT_CHECKPOINT="$PHYSRVG_DIT_CHECKPOINT" \
    STEP_OUTPUT_DIR_NAME="$shard_tag" \
    bash "$ORIGINAL_SCRIPT" "$checkpoint" "$gpu_id" "$raw_parent"
}

merge_shards() {
  local shard_count="$1"
  [[ "$shard_count" =~ ^[1-9][0-9]*$ ]] || die "SHARD_COUNT must be positive"
  local final_raw="$WORKSPACE/raw/$RUN_NAME"
  mkdir -p "$final_raw"
  for ((index = 0; index < shard_count; index++)); do
    local tag
    tag="$(printf '%s_shard%02d-of-%02d' "$RUN_NAME" "$index" "$shard_count")"
    local source_dir="$WORKSPACE/raw_shards/$tag"
    [[ -d "$source_dir" ]] || die "missing shard directory: $source_dir"
    local count=0
    while IFS= read -r -d '' video; do
      local target="$final_raw/$(basename "$video")"
      if [[ -e "$target" ]]; then
        [[ "$(stat -c '%d:%i' "$video")" == "$(stat -c '%d:%i' "$target")" ]] || \
          die "conflicting existing raw file: $target"
      else
        ln "$video" "$target"
      fi
      count=$((count + 1))
    done < <(find "$source_dir" -maxdepth 1 -type f -name '*.mp4' -print0)
    printf 'merged shard=%s videos=%s\n' "$index" "$count"
  done
  local merged_count
  merged_count="$(find "$final_raw" -maxdepth 1 -type f -name '*.mp4' | wc -l)"
  [[ "$merged_count" -eq 198 ]] || die "merged raw folder has $merged_count MP4 files, expected 198"
  printf 'merged_raw=%s\n' "$final_raw"
}

postprocess() {
  local raw_folder="$WORKSPACE/raw/$RUN_NAME"
  local submission="$WORKSPACE/generated_videos_5s/$RUN_NAME"
  [[ -d "$raw_folder" ]] || die "raw folder not found: $raw_folder"
  mkdir -p "$(dirname "$submission")"
  "$PYTHON" "$PREPARE_OUTPUTS" \
    --raw-folder "$raw_folder" \
    --input-list "$INPUT_LIST" \
    --output-folder "$submission"
  printf 'submission=%s\n' "$submission"
}

validate() {
  local submission="$WORKSPACE/generated_videos_5s/$RUN_NAME"
  [[ -d "$submission" ]] || die "submission folder not found: $submission"
  "$PYTHON" "$VALIDATOR" \
    --descriptions-file "/home/gaoya/Code_Video/Code_bench/physics-IQ-benchmark-main/descriptions/best_practice/descriptions_base.csv" \
    "$submission"
}

score() {
  local submission="$WORKSPACE/generated_videos_5s/$RUN_NAME"
  local evaluation="$WORKSPACE/evaluation/$RUN_NAME"
  local descriptions="/home/gaoya/Code_Video/Code_bench/physics-IQ-benchmark-main/descriptions/best_practice/descriptions_base.csv"
  [[ -d "$submission" ]] || die "submission folder not found: $submission"
  mkdir -p "$evaluation"
  bash "$OFFICIAL_RUNNER" \
    --n-process "$OFFICIAL_N_PROCESS" \
    --output-folder "$evaluation" \
    --descriptions-file "$descriptions" \
    "$submission"
  local result_csv="$evaluation/physics-IQ-benchmark-verified/results/${RUN_NAME}.csv"
  [[ -s "$result_csv" ]] || die "official CSV not found: $result_csv"
  bash "$AGGREGATOR" \
    "$result_csv" \
    --save-csv "$evaluation/${RUN_NAME}_verified_summary.csv" \
    --model-name "$RUN_NAME"
}

[[ $# -ge 1 ]] || { usage; exit 2; }
COMMAND="$1"
shift
case "$COMMAND" in
  prepare)
    prepare_inputs "$(case_limit_or_default "${1:-198}")"
    ;;
  generate)
    [[ $# -ge 4 ]] || { usage; exit 2; }
    generate "$@"
    ;;
  merge)
    [[ $# -eq 1 ]] || { usage; exit 2; }
    merge_shards "$1"
    ;;
  postprocess)
    [[ $# -eq 0 ]] || { usage; exit 2; }
    postprocess
    ;;
  validate)
    [[ $# -eq 0 ]] || { usage; exit 2; }
    validate
    ;;
  score)
    [[ $# -eq 0 ]] || { usage; exit 2; }
    score
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    die "unknown command: $COMMAND"
    ;;
esac
