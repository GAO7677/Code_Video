#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 CONFIG.env" >&2
  exit 2
fi

CONFIG="$(realpath "$1")"
# shellcheck source=/dev/null
source "$CONFIG"
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKER_LOCAL="$SCRIPT_ROOT/run_xssc_experiment_verified_gpu67_remote_worker.sh"
INPUT_PREPARER_LOCAL="$SCRIPT_ROOT/prepare_verified_remote_inputs.py"
OUTPUT_PREPARER_LOCAL="$SCRIPT_ROOT/prepare_verified_outputs.py"

for path in \
  "$WORKER_LOCAL" \
  "$INPUT_PREPARER_LOCAL" \
  "$OUTPUT_PREPARER_LOCAL" \
  "$LOCAL_CHECKPOINT_DIR/checkpoint.safetensors" \
  "$LOCAL_EXPERIMENT_CONFIG" \
  "$LOCAL_PRETRAINED_LORA"; do
  [[ -e "$path" ]] || { echo "Missing local path: $path" >&2; exit 1; }
done

ssh -o BatchMode=yes "$REMOTE_HOST" \
  "mkdir -p \
    '$REMOTE_XSSC_ROOT' \
    '$REMOTE_RUNTIME_TRAIN_ROOT' \
    '$REMOTE_RUNTIME_EXPERIMENT_ROOT' \
    '$REMOTE_CHECKPOINT_DIR' \
    '$(dirname "$REMOTE_PRETRAINED_LORA")' \
    '$REMOTE_LOG_DIR'"

rsync -a "$CONFIG" "$REMOTE_HOST:$REMOTE_XSSC_ROOT/"
rsync -a "$WORKER_LOCAL" "$INPUT_PREPARER_LOCAL" \
  "$REMOTE_HOST:$REMOTE_XSSC_ROOT/"
rsync -a "$OUTPUT_PREPARER_LOCAL" \
  "$REMOTE_HOST:$REMOTE_OUTPUT_PREPARER"

# Isolated runtime: do not overwrite the model repository on SSH 118.
rsync -a \
  --exclude wandb \
  --exclude __pycache__ \
  --exclude '*.pyc' \
  "$LOCAL_EXPERIMENT_ROOT/" \
  "$REMOTE_HOST:$REMOTE_RUNTIME_EXPERIMENT_ROOT/"
rsync -a --exclude __pycache__ --exclude '*.pyc' \
  "$LOCAL_TRAIN_XSSC_ROOT/"*.py \
  "$REMOTE_HOST:$REMOTE_RUNTIME_TRAIN_ROOT/"

rsync -a "$LOCAL_CHECKPOINT_DIR/" "$REMOTE_HOST:$REMOTE_CHECKPOINT_DIR/"
rsync -a "$LOCAL_EXPERIMENT_CONFIG" \
  "$REMOTE_HOST:$REMOTE_ORIGINAL_CONFIG"
rsync -a "$LOCAL_PRETRAINED_LORA" \
  "$REMOTE_HOST:$REMOTE_PRETRAINED_LORA"

REMOTE_CONFIG_PATH="$REMOTE_XSSC_ROOT/$(basename "$CONFIG")"
ssh -o BatchMode=yes "$REMOTE_HOST" \
  "REMOTE_ORIGINAL_CONFIG='$REMOTE_ORIGINAL_CONFIG' \
   REMOTE_EXPERIMENT_CONFIG='$REMOTE_EXPERIMENT_CONFIG' \
   REMOTE_OUTPUT_PREPARER='$REMOTE_OUTPUT_PREPARER' \
   REMOTE_CHECKPOINT_DIR='$REMOTE_CHECKPOINT_DIR' \
   REMOTE_RUNTIME_EXPERIMENT_ROOT='$REMOTE_RUNTIME_EXPERIMENT_ROOT' \
   REMOTE_DESCRIPTIONS_FILE='$REMOTE_DESCRIPTIONS_FILE' \
   REMOTE_LOG_DIR='$REMOTE_LOG_DIR' \
   /bin/bash -s" <<'REMOTE_SETUP'
set -euo pipefail
/home/gaoya/data/agent-data/envs/wan-cu128/bin/python - <<'PY'
import json
import os
from pathlib import Path

source = Path(os.environ["REMOTE_ORIGINAL_CONFIG"])
target = Path(os.environ["REMOTE_EXPERIMENT_CONFIG"])

def remap(value):
    if isinstance(value, dict):
        return {key: remap(item) for key, item in value.items()}
    if isinstance(value, list):
        return [remap(item) for item in value]
    if isinstance(value, str):
        return value.replace("/data/gaoya", "/home/gaoya/data")
    return value

target.write_text(
    json.dumps(remap(json.loads(source.read_text())), indent=2, ensure_ascii=False)
    + "\n"
)
PY
sed -i \
  's#/home/gaoya/miniconda3/envs/wan-cu128#/home/gaoya/data/agent-data/envs/wan-cu128#g' \
  "$REMOTE_OUTPUT_PREPARER"
chmod +x "$REMOTE_OUTPUT_PREPARER"
{
  echo "deployed_at=$(date -Is)"
  sha256sum \
    "$REMOTE_CHECKPOINT_DIR/checkpoint.safetensors" \
    "$REMOTE_RUNTIME_EXPERIMENT_ROOT/run_infer_from_experiment.sh" \
    "$REMOTE_RUNTIME_EXPERIMENT_ROOT/infer_xssc_object_self_attn_lora.py" \
    "$REMOTE_DESCRIPTIONS_FILE"
} >"$REMOTE_LOG_DIR/deployment_manifest.txt"
REMOTE_SETUP

ssh -o BatchMode=yes "$REMOTE_HOST" \
  "chmod +x '$REMOTE_WORKER' '$REMOTE_INPUT_PREPARER'; \
   if tmux has-session -t '$TMUX_SESSION' 2>/dev/null; then \
     echo 'tmux session already exists: $TMUX_SESSION' >&2; exit 1; \
   fi; \
   tmux new-session -d -s '$TMUX_SESSION' \
     \"bash '$REMOTE_WORKER' '$REMOTE_CONFIG_PATH' 2>&1 | tee '$REMOTE_MASTER_LOG'\"; \
   echo 'tmux_session=$TMUX_SESSION'; \
   echo 'master_log=$REMOTE_MASTER_LOG'; \
   echo 'raw_output=$REMOTE_RAW_RUN_DIR'; \
   echo 'submission=$REMOTE_SUBMISSION_DIR'; \
   echo 'official_csv=$REMOTE_EVAL_ROOT/physics-IQ-benchmark-verified/results/$RUN_NAME.csv'"
