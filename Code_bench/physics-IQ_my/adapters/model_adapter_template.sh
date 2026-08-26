#!/usr/bin/env bash
set -euo pipefail

# Copy this file for a new model. The canonical P0 runner exports:
#   PHYSIQ_WORKSPACE, PHYSIQ_DATASET, PHYSIQ_PROMPT_SETTING,
#   PHYSIQ_INPUT_MODE, PHYSIQ_RUN_INDEX, PHYSIQ_RUN_TAG,
#   PHYSIQ_SEED, PHYSIQ_MODEL_NAME, PHYSIQ_INPUT_LIST,
#   PHYSIQ_RAW_ROOT, PHYSIQ_SUBMISSION_ROOT, PHYSIQ_RESULT_FILE,
#   PHYSIQ_NEGATIVE_PROMPT plus its version/digest, and all fixed P0 frame/
#   resolution/sampling values.

: "${PHYSIQ_RESULT_FILE:?launcher must set PHYSIQ_RESULT_FILE}"
: "${PHYSIQ_SUBMISSION_ROOT:?runner must set PHYSIQ_SUBMISSION_ROOT}"
: "${PHYSIQ_RAW_ROOT:?runner must set PHYSIQ_RAW_ROOT}"

RUN_FOLDER="$PHYSIQ_SUBMISSION_ROOT"
mkdir -p "$PHYSIQ_RAW_ROOT" "$RUN_FOLDER"

cat >&2 <<EOF
Implement this adapter's model inference here.
It must write exactly 198 canonical Physics-IQ Verified MP4 files to:
$RUN_FOLDER
Each video must be exactly 5.000 seconds and all videos must share one integer FPS.
EOF
exit 2

# After successful inference, leave only the canonical MP4 files in RUN_FOLDER,
# keep any 189-frame raw MP4s under PHYSIQ_RAW_ROOT, then publish the result:
# printf '%s\n' "$RUN_FOLDER" > "$PHYSIQ_RESULT_FILE"
