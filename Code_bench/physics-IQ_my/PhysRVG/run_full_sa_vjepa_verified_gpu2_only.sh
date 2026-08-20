#!/usr/bin/env bash
# Complete the Full-SA VJEPA Physics-IQ P0 run using GPU 2 only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${PHYSIQ_WORKSPACE:-/data/gaoya/AAA_test_video/0623/test/physicsiq/physicsiq_verified}"
RUN_NAME="${RUN_NAME:-physrvg-full-sa-vjepa-step000500-bpp-run_01}"
CHECKPOINT="${CHECKPOINT:-/data/gaoya/agent-data/checkpoints/physrvg_full_sa_vjepa/full-sa-pybullet-physrvg-vjepa-b2-gacc2-ddp-sync-20260817T190000Z/checkpoints/step-000500}"
INPUT_LIST="${INPUT_LIST:-${WORKSPACE}/inputs/bpp/verified_v2v_bpp_198.txt}"
CACHE_ROOT="${CACHE_ROOT:-/data/gaoya/agent-data/cache/physics-iq-verified/physrvg_full_sa_vjepa/${RUN_NAME}}"
PYTHON="/data/gaoya/agent-data/envs/physrvg-full-sa/bin/python"
OFFICIAL_ENV_BIN="${OFFICIAL_ENV_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin}"
WORKER="${SCRIPT_DIR}/run_physrvg_full_sa_vjepa_verified.sh"
PREPARE="${SCRIPT_DIR}/prepare_full_sa_vjepa_verified_inputs.py"
PREPARE_OUTPUTS="${SCRIPT_DIR}/../xSSC/prepare_verified_outputs.py"
VALIDATOR="${SCRIPT_DIR}/../common/validate_verified_run.py"
OFFICIAL_RUNNER="${SCRIPT_DIR}/../run_verified_official.sh"
AGGREGATOR="${SCRIPT_DIR}/../aggregate_verified_official.sh"
DESCRIPTIONS="/home/gaoya/Code_Video/Code_bench/physics-IQ-benchmark-main/descriptions/best_practice/descriptions_base.csv"
GPU_ID=2
OFFICIAL_N_PROCESS="${OFFICIAL_N_PROCESS:-24}"

# Keep official scoring in the same reusable uv environment as the other P0
# runs.  The inference environment intentionally does not provide uv on PATH.
export PATH="${OFFICIAL_ENV_BIN}:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/data/gaoya/agent-data/cache/uv}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/data/gaoya/agent-data/cache/envs/physics-iq-verified}"

RAW_SHARD_ODD="${WORKSPACE}/raw_shards/${RUN_NAME}_shard00-of-02"
RAW_SHARD_EVEN="${WORKSPACE}/raw_shards/${RUN_NAME}_shard01-of-02_gpu2"
RAW_FINAL="${WORKSPACE}/raw/${RUN_NAME}"
SUBMISSION="${WORKSPACE}/generated_videos_5s/${RUN_NAME}"
EVALUATION="${WORKSPACE}/evaluation/${RUN_NAME}"
ODD_LIST="${CACHE_ROOT}/case_odd_99/verified_v2v_bpp_99.txt"
EVEN_LIST="${CACHE_ROOT}/case_even_99/verified_v2v_bpp_99.txt"

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

[[ "$GPU_ID" != "4" ]] || die "GPU 4 is prohibited"
[[ -s "$CHECKPOINT/adapter_model.safetensors" ]] || die "missing adapter checkpoint"
[[ -s "$CHECKPOINT/adapter_config.json" ]] || die "missing adapter config"
[[ -s "$INPUT_LIST" ]] || die "missing P0 input list"

prepare_parity() {
  local parity="$1"
  local output_root="${CACHE_ROOT}/case_${parity}_99"
  local list="${output_root}/verified_v2v_bpp_99.txt"
  if [[ ! -s "$list" ]]; then
    "$PYTHON" "$PREPARE" \
      --input-list "$INPUT_LIST" \
      --output-root "$output_root" \
      --p0-index-parity "$parity"
  fi
  [[ "$(wc -l < "$list")" -eq 99 ]] || die "invalid $parity list: $list"
}

run_stage() {
  local list="$1"
  local output_root="$2"
  local force="$3"
  mkdir -p "$output_root"
  TOKENIZERS_PARALLELISM=false \
    TEST_LIST="$list" \
    FORCE_INFERENCE="$force" \
    bash "$WORKER" "$CHECKPOINT" "$GPU_ID" "$output_root" 0 1
}

count_mp4() {
  find "$1" -maxdepth 1 -type f -name '*.mp4' -print 2>/dev/null | wc -l
}

merge_gpu2_shards() {
  mkdir -p "$(dirname "$RAW_FINAL")" "$RAW_FINAL"
  local source video target
  for source in "$RAW_SHARD_ODD" "$RAW_SHARD_EVEN"; do
    [[ "$(count_mp4 "$source")" -eq 99 ]] || die "incomplete raw shard: $source"
    while IFS= read -r -d '' video; do
      target="$RAW_FINAL/$(basename "$video")"
      if [[ -e "$target" ]]; then
        [[ "$(stat -c '%d:%i' "$video")" == "$(stat -c '%d:%i' "$target")" ]] || \
          die "conflicting final raw file: $target"
      else
        ln "$video" "$target"
      fi
    done < <(find "$source" -maxdepth 1 -type f -name '*.mp4' -print0)
  done
  [[ "$(count_mp4 "$RAW_FINAL")" -eq 198 ]] || die "final raw count is not 198"
}

prepare_parity odd
prepare_parity even

printf 'stage=odd gpu=%s list=%s output=%s\n' "$GPU_ID" "$ODD_LIST" "$RAW_SHARD_ODD"
run_stage "$ODD_LIST" "$RAW_SHARD_ODD" 0
printf 'odd_raw_count=%s\n' "$(count_mp4 "$RAW_SHARD_ODD")"

printf 'stage=even gpu=%s list=%s output=%s\n' "$GPU_ID" "$EVEN_LIST" "$RAW_SHARD_EVEN"
run_stage "$EVEN_LIST" "$RAW_SHARD_EVEN" 0
printf 'even_raw_count=%s\n' "$(count_mp4 "$RAW_SHARD_EVEN")"

merge_gpu2_shards
mkdir -p "$(dirname "$SUBMISSION")"
"$PYTHON" "$PREPARE_OUTPUTS" \
  --raw-folder "$RAW_FINAL" \
  --input-list "$INPUT_LIST" \
  --output-folder "$SUBMISSION" \
  --force

"$PYTHON" "$VALIDATOR" --descriptions-file "$DESCRIPTIONS" "$SUBMISSION"
mkdir -p "$EVALUATION"
bash "$OFFICIAL_RUNNER" \
  --n-process "$OFFICIAL_N_PROCESS" \
  --output-folder "$EVALUATION" \
  --descriptions-file "$DESCRIPTIONS" \
  "$SUBMISSION"

RESULT_CSV="$EVALUATION/physics-IQ-benchmark-verified/results/${RUN_NAME}.csv"
[[ -s "$RESULT_CSV" ]] || die "official CSV not found: $RESULT_CSV"
bash "$AGGREGATOR" \
  "$RESULT_CSV" \
  --save-csv "$EVALUATION/${RUN_NAME}_verified_summary.csv" \
  --model-name "$RUN_NAME"

printf 'GPU2-only Physics-IQ Verified pipeline complete.\nsubmission=%s\nevaluation=%s\n' \
  "$SUBMISSION" "$EVALUATION"
