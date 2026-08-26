#!/usr/bin/env bash
# Compatibility adapter that stages the existing xSSC output into the shared
# P0 raw/submission roots without changing the generated video bytes.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: xssc_full_sa_no_object_p0.sh CHECKPOINT_DIR [GPU_ID]" >&2
  exit 2
fi

: "${PHYSIQ_RESULT_FILE:?run_physicsiq_p0.py must set PHYSIQ_RESULT_FILE}"
: "${PHYSIQ_WORKSPACE:?run_physicsiq_p0.py must set PHYSIQ_WORKSPACE}"
: "${PHYSIQ_SUBMISSION_ROOT:?run_physicsiq_p0.py must set PHYSIQ_SUBMISSION_ROOT}"
: "${PHYSIQ_RAW_ROOT:?run_physicsiq_p0.py must set PHYSIQ_RAW_ROOT}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEGACY_ADAPTER="${SCRIPT_DIR}/xssc_full_sa_no_object.sh"
CHECKPOINT_DIR="$(realpath "$1")"
GPU_ID="${2:-${PHYSIQ_GPU_ID:?GPU id was not supplied}}"
TEMP_RESULT="$(mktemp /data/gaoya/agent-data/cache/xssc-p0-result.XXXXXX)"
trap 'rm -f -- "$TEMP_RESULT"' EXIT

[[ "$GPU_ID" != 4 ]] || { echo "GPU 4 is prohibited" >&2; exit 2; }

env \
  PHYSIQ_RESULT_FILE="$TEMP_RESULT" \
  PHYSIQ_RUN_INDEX="${PHYSIQ_RUN_INDEX:-1}" \
  PHYSIQ_NEGATIVE_PROMPT="${PHYSIQ_NEGATIVE_PROMPT}" \
  PHYSIQ_NEGATIVE_PROMPT_VERSION="${PHYSIQ_NEGATIVE_PROMPT_VERSION}" \
  PHYSIQ_NEGATIVE_PROMPT_SHA256="${PHYSIQ_NEGATIVE_PROMPT_SHA256}" \
  CASE_LIMIT="${PHYSIQ_P0_CASE_COUNT:-198}" \
  bash "$LEGACY_ADAPTER" "$CHECKPOINT_DIR" "$GPU_ID"

SOURCE_SUBMISSION="$(head -n 1 "$TEMP_RESULT")"
[[ -d "$SOURCE_SUBMISSION" ]] || { echo "xSSC submission not found: $SOURCE_SUBMISSION" >&2; exit 1; }

source_raw="${SOURCE_SUBMISSION/\/generated_videos_5s\//\/raw\/}"
if [[ ! -d "$source_raw" ]]; then
  source_raw="$(dirname "$(dirname "$SOURCE_SUBMISSION")")/raw/$(basename "$SOURCE_SUBMISSION")"
fi

stage_mp4s() {
  local source="$1"
  local target="$2"
  [[ -d "$source" ]] || { echo "xSSC source folder not found: $source" >&2; exit 1; }
  mkdir -p "$target"
  while IFS= read -r -d '' video; do
    ln "$video" "$target/$(basename "$video")"
  done < <(find "$source" -maxdepth 1 -type f -name '*.mp4' -print0)
}

stage_mp4s "$SOURCE_SUBMISSION" "$PHYSIQ_SUBMISSION_ROOT"
if [[ -d "$source_raw" ]]; then
  stage_mp4s "$source_raw" "$PHYSIQ_RAW_ROOT"
fi

printf '%s\n' "$PHYSIQ_SUBMISSION_ROOT" >"$PHYSIQ_RESULT_FILE"
