#!/usr/bin/env bash
# Run:
# TEST_LIST=/path/to/input_list.txt NUM_INFERENCE_STEPS=2 \
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/run_infer_full_sa_no_object.sh \
#   /path/to/checkpoints/step-000010 3 /data/gaoya/agent-data/outputs/full_sa_no_object
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 CHECKPOINT_DIR GPU_ID [OUTPUT_ROOT]" >&2
  exit 2
fi

CHECKPOINT_DIR="$(realpath "$1")"
GPU_ID="$2"
OUTPUT_ROOT="${3:-/data/gaoya/agent-data/outputs/full_sa_no_object_inference}"
TEST_LIST="${TEST_LIST:-/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-8}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-模糊，低质量，变形，伪影，文字，水印，过曝，欠曝，颜色异常，几何扭曲，物体融化，物理不合理}"

EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
INFER_SCRIPT="${EXPERIMENT_ROOT}/infer_full_sa_no_object_lora.py"

if [[ "${GPU_ID}" == "4" ]]; then
  echo "GPU 4 is prohibited by workspace rules." >&2
  exit 2
fi
if [[ ! -s "${CHECKPOINT_DIR}/checkpoint.safetensors" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT_DIR}/checkpoint.safetensors" >&2
  exit 2
fi
if [[ ! -s "${TEST_LIST}" ]]; then
  echo "TEST_LIST not found or empty: ${TEST_LIST}" >&2
  exit 2
fi

EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-}"
if [[ -z "${EXPERIMENT_CONFIG}" ]]; then
  search_dir="${CHECKPOINT_DIR}"
  while [[ "${search_dir}" != "/" ]]; do
    candidate="${search_dir}/resolved_experiment_config.json"
    if [[ -s "${candidate}" ]]; then
      EXPERIMENT_CONFIG="${candidate}"
      break
    fi
    search_dir="$(dirname "${search_dir}")"
  done
fi
if [[ ! -s "${EXPERIMENT_CONFIG}" ]]; then
  echo "Could not resolve resolved_experiment_config.json for ${CHECKPOINT_DIR}" >&2
  exit 2
fi

readarray -t CONFIG_VALUES < <(
  "${PYTHON}" - "${EXPERIMENT_CONFIG}" <<'PY'
import json
import sys
config = json.load(open(sys.argv[1], "r", encoding="utf-8"))["resolved_config"]
if config["adaptation"]["mode"] != "full_sa":
    raise SystemExit("Expected adaptation.mode=full_sa")
if config["adaptation"].get("enable_object_branch", True):
    raise SystemExit("Expected adaptation.enable_object_branch=false")
print(config["experiment"]["name"])
print(config["paths"]["wan_root"])
print(config["paths"]["pretrained_lora_checkpoint"])
PY
)
EXPERIMENT_NAME="${CONFIG_VALUES[0]}"
WAN_ROOT="${CONFIG_VALUES[1]}"
PRETRAINED_LORA="${CONFIG_VALUES[2]}"
STEP_TAG="$(basename "${CHECKPOINT_DIR}")"
STEP_OUTPUT_DIR_NAME="${STEP_OUTPUT_DIR_NAME:-${EXPERIMENT_NAME}_${STEP_TAG}_steps${NUM_INFERENCE_STEPS}_512x896_ctx08_49f}"
TRACE_ROOT="${TRACE_ROOT:-${OUTPUT_ROOT}/_numeric_traces/${STEP_OUTPUT_DIR_NAME}}"
mkdir -p "${OUTPUT_ROOT}" "${TRACE_ROOT}"

echo "experiment=${EXPERIMENT_NAME}"
echo "checkpoint=${CHECKPOINT_DIR}"
echo "config=${EXPERIMENT_CONFIG}"
echo "gpu=${GPU_ID}"
echo "object_branch=false"
echo "output=${OUTPUT_ROOT}/${STEP_OUTPUT_DIR_NAME}"

exec env \
  PYTHONPATH="${PROJECT_ROOT}:${EXPERIMENT_ROOT}:${DIFFSYNTH_ROOT}" \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG}" \
  "${PYTHON}" "${INFER_SCRIPT}" \
  --weights-root "${CHECKPOINT_DIR}" \
  --input-json-list-path "${TEST_LIST}" \
  --model-name "${EXPERIMENT_NAME}" \
  --output-root "${OUTPUT_ROOT}" \
  --step-output-dir-name "${STEP_OUTPUT_DIR_NAME}" \
  --shard-tag "${EXPERIMENT_NAME}" \
  --wan-root "${WAN_ROOT}" \
  --lora-checkpoint "${PRETRAINED_LORA}" \
  --device cuda:0 \
  --aux-device cuda:0 \
  --inference-devices cuda:0,cuda:0 \
  --height 512 \
  --width 896 \
  --num-frames 49 \
  --context-frames 8 \
  --sampling-mode prefix \
  --num-inference-steps "${NUM_INFERENCE_STEPS}" \
  --negative-prompt "${NEGATIVE_PROMPT}" \
  --dump-numeric-trace-root "${TRACE_ROOT}" \
  --force
