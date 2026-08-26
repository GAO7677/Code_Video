#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  cat >&2 <<'EOF'
Usage:
  run_verified_inference.sh CHECKPOINT_DIR GPU_ID RUN_INDEX

RUN_INDEX must be 1, 2, 3, or 4 and selects seeds 42, 43, 44, or 45.
EOF
  exit 2
fi

CHECKPOINT_DIR="$(realpath "$1")"
GPU_ID="$2"
RUN_INDEX="$3"
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo "GPU_ID must be an integer" >&2; exit 2; }
[[ "$GPU_ID" != "4" ]] || { echo "GPU 4 is prohibited" >&2; exit 2; }
[[ "$RUN_INDEX" =~ ^[1-4]$ ]] || { echo "RUN_INDEX must be 1, 2, 3, or 4" >&2; exit 2; }
[[ -s "$CHECKPOINT_DIR/checkpoint.safetensors" ]] || {
  echo "Missing checkpoint: $CHECKPOINT_DIR/checkpoint.safetensors" >&2
  exit 2
}

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P0_PROMPT_CONFIG="${SCRIPT_ROOT}/../common/physicsiq_p0_prompt.env"
[[ -r "$P0_PROMPT_CONFIG" ]] || {
  echo "Missing shared P0 negative-prompt config: $P0_PROMPT_CONFIG" >&2
  exit 2
}
# shellcheck source=/dev/null
source "$P0_PROMPT_CONFIG"
EXPERIMENT_ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments
PROJECT_ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
INFER_SCRIPT="$EXPERIMENT_ROOT/infer_full_sa_no_object_lora.py"
RESULT_BASE=/data/gaoya/AAA_test_video/0623/test/physicsiq/physicsiq_verified
CASE_LIMIT="${CASE_LIMIT:-198}"
[[ "$CASE_LIMIT" =~ ^[0-9]+$ ]] && ((CASE_LIMIT >= 1 && CASE_LIMIT <= 198)) || {
  echo "CASE_LIMIT must be an integer from 1 to 198" >&2
  exit 2
}

"$PYTHON" "$SCRIPT_ROOT/build_verified_v2v_inputs.py" \
  --result-base "$RESULT_BASE" \
  --limit "$CASE_LIMIT"

ACTIVE_INPUT_LIST="$RESULT_BASE/inputs/bpp/verified_v2v_bpp_${CASE_LIMIT}.txt"
SMOKE_SUFFIX=""
if ((CASE_LIMIT < 198)); then
  SMOKE_SUFFIX="-case${CASE_LIMIT}-smoke"
fi

EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-}"
if [[ -z "$EXPERIMENT_CONFIG" ]]; then
  search_dir="$CHECKPOINT_DIR"
  while [[ "$search_dir" != "/" ]]; do
    candidate="$search_dir/resolved_experiment_config.json"
    if [[ -s "$candidate" ]]; then
      EXPERIMENT_CONFIG="$candidate"
      break
    fi
    search_dir="$(dirname "$search_dir")"
  done
fi
[[ -s "$EXPERIMENT_CONFIG" ]] || {
  echo "Could not resolve resolved_experiment_config.json" >&2
  exit 2
}

readarray -t CONFIG_VALUES < <(
  "$PYTHON" - "$EXPERIMENT_CONFIG" <<'PY'
import json
import re
import sys
config = json.load(open(sys.argv[1], encoding="utf-8"))["resolved_config"]
if config["adaptation"]["mode"] != "full_sa":
    raise SystemExit("Expected adaptation.mode=full_sa")
if config["adaptation"].get("enable_object_branch", True):
    raise SystemExit("Expected adaptation.enable_object_branch=false")
name = re.sub(r"[^0-9A-Za-z_.-]+", "-", config["experiment"]["name"]).strip(".-")
print(name)
print(config["paths"]["wan_root"])
print(config["paths"]["pretrained_lora_checkpoint"])
PY
)

EXPERIMENT_NAME="${CONFIG_VALUES[0]}"
WAN_ROOT="${CONFIG_VALUES[1]}"
PRETRAINED_LORA="${CONFIG_VALUES[2]}"
STEP_TAG="$(basename "$CHECKPOINT_DIR")"
CHECKPOINT_SHA="$(sha256sum "$CHECKPOINT_DIR/checkpoint.safetensors" | cut -c1-12)"
RUN_TAG="$(printf 'run_%02d' "$RUN_INDEX")"
RUN_NAME="${EXPERIMENT_NAME}-${STEP_TAG}-${CHECKPOINT_SHA}-bpp-${RUN_TAG}${SMOKE_SUFFIX}"
SEED=$((41 + RUN_INDEX))
if ((CASE_LIMIT < 198)); then
  RAW_ROOT="$RESULT_BASE/smoke/raw"
  FINAL_ROOT="$RESULT_BASE/smoke/generated_videos_5s"
else
  RAW_ROOT="$RESULT_BASE/raw"
  FINAL_ROOT="$RESULT_BASE/generated_videos_5s"
fi
RAW_FOLDER="$RAW_ROOT/$RUN_NAME"
FINAL_FOLDER="$FINAL_ROOT/$RUN_NAME"
# New P0 runs use the same long prompt as PhysRVG-72f-adapted. An explicit
# PHYSIQ_NEGATIVE_PROMPT override remains available for reproducing a frozen
# historical run whose metadata records a different prompt.
NEGATIVE_PROMPT="${PHYSIQ_NEGATIVE_PROMPT:-${PHYSIQ_P0_NEGATIVE_PROMPT}}"
NEGATIVE_PROMPT_VERSION="${PHYSIQ_NEGATIVE_PROMPT_VERSION:-${PHYSIQ_P0_NEGATIVE_PROMPT_VERSION}}"
mkdir -p "$RAW_ROOT" "$FINAL_ROOT"

CMD=(
  "$PYTHON" "$INFER_SCRIPT"
  --weights-root "$CHECKPOINT_DIR"
  --input-json-list-path "$ACTIVE_INPUT_LIST"
  --model-name "$EXPERIMENT_NAME"
  --output-root "$RAW_ROOT"
  --step-output-dir-name "$RUN_NAME"
  --shard-tag "$RUN_NAME"
  --wan-root "$WAN_ROOT"
  --lora-checkpoint "$PRETRAINED_LORA"
  --device cuda:0
  --aux-device cuda:0
  --inference-devices cuda:0,cuda:0
  --height 512
  --width 896
  --num-frames 189
  --context-frames 72
  --fps 24
  --sampling-mode prefix
  --num-inference-steps 40
  --seed "$SEED"
  --negative-prompt "$NEGATIVE_PROMPT"
)

printf 'negative_prompt_version=%s\n' "$NEGATIVE_PROMPT_VERSION"
printf 'Inference command:'
printf ' %q' "${CMD[@]}"
printf '\nrun_name=%s\ncheckpoint_sha=%s\ncases=%s\nseed=%s\nraw=%s\nfinal=%s\n' \
  "$RUN_NAME" "$CHECKPOINT_SHA" "$CASE_LIMIT" "$SEED" "$RAW_FOLDER" "$FINAL_FOLDER"

env \
  PYTHONPATH="$PROJECT_ROOT:$EXPERIMENT_ROOT:$DIFFSYNTH_ROOT" \
  PYTHONNOUSERSITE=1 \
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  EXPERIMENT_CONFIG="$EXPERIMENT_CONFIG" \
  "${CMD[@]}"

"$PYTHON" "$SCRIPT_ROOT/prepare_verified_outputs.py" \
  --raw-folder "$RAW_FOLDER" \
  --input-list "$ACTIVE_INPUT_LIST" \
  --output-folder "$FINAL_FOLDER"

printf 'Physics-IQ Verified run ready: %s\n' "$FINAL_FOLDER"
