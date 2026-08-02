#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: xssc_full_sa_no_object.sh CHECKPOINT_DIR GPU_ID" >&2
  exit 2
fi

: "${PHYSIQ_RESULT_FILE:?launcher must set PHYSIQ_RESULT_FILE}"
: "${PHYSIQ_RUN_INDEX:?launcher must set PHYSIQ_RUN_INDEX}"
: "${PHYSIQ_WORKSPACE:?launcher must set PHYSIQ_WORKSPACE}"
[[ "${PHYSIQ_PROMPT_SETTING:-bpp}" == "bpp" ]] || {
  echo "The current xSSC adapter supports bpp only" >&2
  exit 2
}
[[ "${PHYSIQ_INPUT_MODE:-v2v}" == "v2v" ]] || {
  echo "The current xSSC adapter supports v2v only" >&2
  exit 2
}

EXPECTED_WORKSPACE=/data/gaoya/AAA_test_video/0623/test/physicsiq/physicsiq_verified
[[ "$PHYSIQ_WORKSPACE" == "$EXPECTED_WORKSPACE" ]] || {
  echo "Current xSSC implementation writes to $EXPECTED_WORKSPACE" >&2
  exit 2
}

CHECKPOINT_DIR="$(realpath "$1")"
GPU_ID="$2"
CHECKPOINT_SHA="$(sha256sum "$CHECKPOINT_DIR/checkpoint.safetensors" | cut -c1-12)"
RUN_TAG="$(printf 'run_%02d' "$PHYSIQ_RUN_INDEX")"

CASE_LIMIT=198 bash \
  /home/gaoya/Code_Video/Code_bench/physics-IQ_my/xSSC/run_verified_inference.sh \
  "$CHECKPOINT_DIR" \
  "$GPU_ID" \
  "$PHYSIQ_RUN_INDEX"

shopt -s nullglob
matches=("$PHYSIQ_WORKSPACE/generated_videos_5s/"*"-$CHECKPOINT_SHA-bpp-$RUN_TAG")
shopt -u nullglob
if ((${#matches[@]} != 1)); then
  echo "Expected one xSSC output matching checkpoint $CHECKPOINT_SHA, found ${#matches[@]}" >&2
  exit 1
fi
printf '%s\n' "${matches[0]}" > "$PHYSIQ_RESULT_FILE"
