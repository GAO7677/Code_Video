#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VALIDATOR="$ROOT/common/validate_verified_run.py"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
export PATH="/home/gaoya/miniconda3/envs/wan-cu128/bin:$PATH"
export PHYSIQ_WORKSPACE="${PHYSIQ_WORKSPACE:-/data/gaoya/AAA_test_video/0623/test/physicsiq/physicsiq_verified}"
export PHYSIQ_DATASET="${PHYSIQ_DATASET:-/data/gaoya/dataset/Anates-Labs-Research-Physics-IQ-Verified}"
export PHYSIQ_PROMPT_SETTING="${PHYSIQ_PROMPT_SETTING:-bpp}"
export PHYSIQ_INPUT_MODE="${PHYSIQ_INPUT_MODE:-v2v}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/data/gaoya/agent-data/cache/uv}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/data/gaoya/agent-data/cache/envs/physics-iq-verified}"

P0_PROMPT_CONFIG="$ROOT/common/physicsiq_p0_prompt.env"
[[ -r "$P0_PROMPT_CONFIG" ]] || {
  echo "Missing shared P0 negative-prompt config: $P0_PROMPT_CONFIG" >&2
  exit 2
}
# shellcheck source=/dev/null
source "$P0_PROMPT_CONFIG"
if [[ "$PHYSIQ_PROMPT_SETTING" == "bpp" && "$PHYSIQ_INPUT_MODE" == "v2v" ]]; then
  export PHYSIQ_NEGATIVE_PROMPT="$PHYSIQ_P0_NEGATIVE_PROMPT"
  export PHYSIQ_NEGATIVE_PROMPT_VERSION="$PHYSIQ_P0_NEGATIVE_PROMPT_VERSION"
  export PHYSIQ_NEGATIVE_PROMPT_SHA256="$PHYSIQ_P0_NEGATIVE_PROMPT_SHA256"
fi

usage() {
  cat <<'EOF'
Usage:
  launch_verified_benchmark.sh generate ADAPTER RUN_INDEX [ADAPTER_ARGS ...]
  launch_verified_benchmark.sh validate RUN_FOLDER [RUN_FOLDER ...]
  launch_verified_benchmark.sh score RUN_FOLDER [RUN_FOLDER ... up to 4]

Environment overrides:
  PHYSIQ_WORKSPACE       Shared data/output root.
  PHYSIQ_MODEL_NAME      Stable model name exposed to adapters.
  PHYSIQ_PROMPT_SETTING  bpp (default) or op.
  PHYSIQ_INPUT_MODE      v2v (default) or i2v.
  PHYSIQ_NEGATIVE_PROMPT New P0 adapters receive the canonical long prompt.
  PHYSIQ_NEGATIVE_PROMPT_VERSION  Version of the canonical P0 prompt.

An adapter must write its completed run-folder path to PHYSIQ_RESULT_FILE.
EOF
}

descriptions_file() {
  case "$PHYSIQ_PROMPT_SETTING" in
    bpp)
      printf '%s\n' "$ROOT/../physics-IQ-benchmark-main/descriptions/best_practice/descriptions_base.csv"
      ;;
    op)
      printf '%s\n' "$ROOT/../physics-IQ-benchmark-main/descriptions/descriptions_original.csv"
      ;;
    *)
      echo "PHYSIQ_PROMPT_SETTING must be bpp or op" >&2
      return 2
      ;;
  esac
}

validate_runs() {
  "$PYTHON" "$VALIDATOR" \
    --descriptions-file "$(descriptions_file)" \
    "$@"
}

(($# >= 1)) || { usage; exit 2; }
COMMAND="$1"
shift

case "$COMMAND" in
  generate)
    (($# >= 2)) || { usage; exit 2; }
    ADAPTER="$(realpath "$1")"
    RUN_INDEX="$2"
    shift 2
    [[ -f "$ADAPTER" ]] || { echo "Adapter not found: $ADAPTER" >&2; exit 2; }
    [[ "$RUN_INDEX" =~ ^[1-4]$ ]] || { echo "RUN_INDEX must be 1..4" >&2; exit 2; }
    export PHYSIQ_RUN_INDEX="$RUN_INDEX"
    export PHYSIQ_RUN_TAG="$(printf 'run_%02d' "$RUN_INDEX")"
    export PHYSIQ_SEED=$((41 + RUN_INDEX))
    mkdir -p /data/gaoya/agent-data/cache/physics-iq-launcher
    PHYSIQ_RESULT_FILE="$(mktemp /data/gaoya/agent-data/cache/physics-iq-launcher/result.XXXXXXXX)"
    export PHYSIQ_RESULT_FILE
    cleanup() { rm -f -- "$PHYSIQ_RESULT_FILE"; }
    trap cleanup EXIT
    bash "$ADAPTER" "$@"
    [[ -s "$PHYSIQ_RESULT_FILE" ]] || {
      echo "Adapter did not write PHYSIQ_RESULT_FILE" >&2
      exit 1
    }
    RUN_FOLDER="$(cat "$PHYSIQ_RESULT_FILE")"
    validate_runs "$RUN_FOLDER"
    printf 'Validated generated run: %s\n' "$RUN_FOLDER"
    ;;
  validate)
    (($# >= 1)) || { usage; exit 2; }
    validate_runs "$@"
    ;;
  score)
    (($# >= 1 && $# <= 4)) || { usage; exit 2; }
    RUN_FOLDERS=("$@")
    validate_runs "${RUN_FOLDERS[@]}"
    EVALUATION_ROOT="$PHYSIQ_WORKSPACE/evaluation"
    bash "$ROOT/run_verified_official.sh" \
      --output-folder "$EVALUATION_ROOT" \
      --descriptions-file "$(descriptions_file)" \
      "${RUN_FOLDERS[@]}"
    RESULTS="$EVALUATION_ROOT/physics-IQ-benchmark-verified/results"
    CSVS=()
    for folder in "${RUN_FOLDERS[@]}"; do
      name="$(basename "${folder%/}")"
      csv="$RESULTS/$name.csv"
      [[ -s "$csv" ]] || { echo "Missing official result CSV: $csv" >&2; exit 1; }
      CSVS+=("$csv")
    done
    first_name="$(basename "${RUN_FOLDERS[0]%/}")"
    group_name="$(printf '%s' "$first_name" | sed -E 's/-run_[0-9]{2}$//')"
    summary="$EVALUATION_ROOT/${group_name}_verified_summary.csv"
    bash "$ROOT/aggregate_verified_official.sh" \
      "${CSVS[@]}" \
      --save-csv "$summary" \
      --model-name "${PHYSIQ_MODEL_NAME:-$group_name}"
    printf 'Verified summary: %s\n' "$summary"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown command: $COMMAND" >&2
    usage >&2
    exit 2
    ;;
esac
