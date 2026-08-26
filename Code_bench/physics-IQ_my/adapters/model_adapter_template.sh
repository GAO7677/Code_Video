#!/usr/bin/env bash
set -euo pipefail

# Copy this file for a new model. The generic launcher exports:
#   PHYSIQ_WORKSPACE, PHYSIQ_DATASET, PHYSIQ_PROMPT_SETTING,
#   PHYSIQ_INPUT_MODE, PHYSIQ_RUN_INDEX, PHYSIQ_RUN_TAG,
#   PHYSIQ_SEED, PHYSIQ_MODEL_NAME, PHYSIQ_RESULT_FILE, and (for new P0
#   bpp/v2v runs) PHYSIQ_NEGATIVE_PROMPT plus its version/digest.

: "${PHYSIQ_RESULT_FILE:?launcher must set PHYSIQ_RESULT_FILE}"
: "${PHYSIQ_WORKSPACE:?launcher must set PHYSIQ_WORKSPACE}"
: "${PHYSIQ_RUN_TAG:?launcher must set PHYSIQ_RUN_TAG}"
: "${PHYSIQ_MODEL_NAME:?set PHYSIQ_MODEL_NAME for this adapter}"

RUN_FOLDER="$PHYSIQ_WORKSPACE/generated_videos_5s/${PHYSIQ_MODEL_NAME}-${PHYSIQ_PROMPT_SETTING}-${PHYSIQ_RUN_TAG}"
mkdir -p "$RUN_FOLDER"

cat >&2 <<EOF
Implement this adapter's model inference here.
It must write exactly 198 canonical Physics-IQ Verified MP4 files to:
$RUN_FOLDER
Each video must be exactly 5.000 seconds and all videos must share one integer FPS.
EOF
exit 2

# After successful inference, leave only the canonical MP4 files in RUN_FOLDER,
# then publish the result to the generic launcher:
# printf '%s\n' "$RUN_FOLDER" > "$PHYSIQ_RESULT_FILE"
