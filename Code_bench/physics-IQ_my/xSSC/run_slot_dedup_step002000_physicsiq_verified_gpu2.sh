#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-${SCRIPT_DIR}/slot_dedup_step002000_physicsiq_verified_gpu2.json}"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python

if [[ ! -s "${CONFIG}" ]]; then
  echo "Missing benchmark config: ${CONFIG}" >&2
  exit 2
fi

eval "$("${PYTHON}" - "${CONFIG}" <<'PY'
import json
import shlex
import sys

config = json.load(open(sys.argv[1], "r", encoding="utf-8"))
inference = config["inference"]
values = {
    "BENCHMARK": config["benchmark"],
    "PROMPT_SETTING": config["prompt_setting"],
    "INPUT_MODE": config["input_mode"],
    "WORKSPACE": config["workspace"],
    "INPUT_LIST": config["input_list"],
    "INFER_SCRIPT": config["model_inference_entry"],
    "CHECKPOINT_DIR": config["checkpoint_dir"],
    "CHECKPOINT_SHA256": config["checkpoint_sha256"],
    "EXPERIMENT_CONFIG": config["resolved_experiment_config"],
    "MODEL_NAME": config["model_name"],
    "RUN_NAME": config["run_name"],
    "GPU_ID": config["gpu_id"],
    "HEIGHT": inference["height"],
    "WIDTH": inference["width"],
    "FPS": inference["fps"],
    "CONTEXT_FRAMES": inference["conditioning_frames"],
    "NUM_FRAMES": inference["num_frames"],
    "SAMPLING_MODE": inference["sampling_mode"],
    "NUM_INFERENCE_STEPS": inference["num_inference_steps"],
    "SEED": inference["seed"],
    "NEGATIVE_PROMPT": inference["negative_prompt"],
    "XSSC_PREPROCESS_MODE": inference["xssc_preprocess_mode"],
    "XSSC_SLOT_TEMPORAL_MODE": inference["xssc_slot_temporal_mode"],
    "CASE_COUNT": config["case_count"],
    "RUN_OFFICIAL_SCORING": int(bool(config["run_official_scoring"])),
}
for name, value in values.items():
    print(f"{name}={shlex.quote(str(value))}")
PY
)"

if [[ "${BENCHMARK}" != "Physics-IQ-Verified" || "${PROMPT_SETTING}" != "bpp" || "${INPUT_MODE}" != "v2v" ]]; then
  echo "This launcher requires Physics-IQ-Verified bpp/v2v configuration." >&2
  exit 2
fi
if [[ "${GPU_ID}" == "4" ]]; then
  echo "GPU 4 is prohibited by workspace rules." >&2
  exit 2
fi
if [[ "${XSSC_SLOT_TEMPORAL_MODE}" != "last_time7" ]]; then
  echo "Expected the user-selected xSSC temporal mode last_time7." >&2
  exit 2
fi
if [[ ! -s "${CHECKPOINT_DIR}/checkpoint.safetensors" ]]; then
  echo "Missing checkpoint: ${CHECKPOINT_DIR}/checkpoint.safetensors" >&2
  exit 2
fi
if [[ ! -s "${EXPERIMENT_CONFIG}" || ! -s "${INFER_SCRIPT}" || ! -s "${INPUT_LIST}" ]]; then
  echo "Missing experiment config, inference entry, or Verified input list." >&2
  exit 2
fi
if [[ "$(wc -l < "${INPUT_LIST}")" -ne "${CASE_COUNT}" ]]; then
  echo "Verified input list must contain exactly ${CASE_COUNT} cases." >&2
  exit 2
fi
actual_sha="$(sha256sum "${CHECKPOINT_DIR}/checkpoint.safetensors" | awk '{print $1}')"
if [[ "${actual_sha}" != "${CHECKPOINT_SHA256}" ]]; then
  echo "Checkpoint SHA-256 mismatch: expected ${CHECKPOINT_SHA256}, got ${actual_sha}" >&2
  exit 2
fi

EXPERIMENT_ROOT="$(dirname "${INFER_SCRIPT}")"
TRAIN_XSSC_ROOT="$(dirname "${EXPERIMENT_ROOT}")"
PROJECT_ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
WAN_ROOT=/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B
PRETRAINED_LORA=/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora/checkpoints/step-010000/checkpoint.safetensors
RAW_FOLDER="${WORKSPACE}/raw/${RUN_NAME}"
OUTPUT_FOLDER="${WORKSPACE}/generated_videos_5s/${RUN_NAME}"
TRACE_ROOT="${WORKSPACE}/numeric_traces/${RUN_NAME}"
XSSC_BOX_CACHE_DIR=/data/gaoya/agent-data/cache/xssc_object_self_attn_lora_infer/physicsiq_verified_slot_dedup_step002000
LOG_DIR="${WORKSPACE}/logs"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"

mkdir -p "${WORKSPACE}/raw" "${WORKSPACE}/generated_videos_5s" "${TRACE_ROOT}" "${XSSC_BOX_CACHE_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1
trap 'rc=$?; printf "[%s] pipeline_exit=%s\n" "$(date -Is)" "$rc"' EXIT

echo "[$(date -Is)] Physics-IQ Verified slot-dedup pipeline start"
echo "config=${CONFIG}"
echo "checkpoint=${CHECKPOINT_DIR}"
echo "checkpoint_sha256=${CHECKPOINT_SHA256}"
echo "physical_gpu=${GPU_ID}"
echo "wan_context=${CONTEXT_FRAMES} frames @ ${FPS} fps, mode=${SAMPLING_MODE}"
echo "xssc_temporal_mode=${XSSC_SLOT_TEMPORAL_MODE}"
echo "raw_folder=${RAW_FOLDER}"
echo "output_folder=${OUTPUT_FOLDER}"

env \
  PYTHONNOUSERSITE=1 \
  PYTHONPATH="${PROJECT_ROOT}:${TRAIN_XSSC_ROOT}:${EXPERIMENT_ROOT}:${DIFFSYNTH_ROOT}" \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  TOKENIZERS_PARALLELISM=false \
  EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG}" \
  XSSC_BOX_CACHE_DIR="${XSSC_BOX_CACHE_DIR}" \
  XSSC_PREPROCESS_MODE="${XSSC_PREPROCESS_MODE}" \
  XSSC_SLOT_TEMPORAL_MODE="${XSSC_SLOT_TEMPORAL_MODE}" \
  "${PYTHON}" "${INFER_SCRIPT}" \
    --weights-root "${CHECKPOINT_DIR}" \
    --input-json-list-path "${INPUT_LIST}" \
    --model-name "${MODEL_NAME}" \
    --output-root "${WORKSPACE}/raw" \
    --step-output-dir-name "${RUN_NAME}" \
    --shard-tag "${RUN_NAME}" \
    --wan-root "${WAN_ROOT}" \
    --lora-checkpoint "${PRETRAINED_LORA}" \
    --device cuda:0 \
    --aux-device cuda:0 \
    --inference-devices cuda:0,cuda:0 \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --num-frames "${NUM_FRAMES}" \
    --context-frames "${CONTEXT_FRAMES}" \
    --fps "${FPS}" \
    --sampling-mode "${SAMPLING_MODE}" \
    --num-inference-steps "${NUM_INFERENCE_STEPS}" \
    --seed "${SEED}" \
    --negative-prompt "${NEGATIVE_PROMPT}" \
    --dump-numeric-trace-root "${TRACE_ROOT}" \
    --force

"${PYTHON}" "${SCRIPT_DIR}/prepare_verified_outputs.py" \
  --raw-folder "${RAW_FOLDER}" \
  --input-list "${INPUT_LIST}" \
  --output-folder "${OUTPUT_FOLDER}" \
  --force

if [[ "${RUN_OFFICIAL_SCORING}" == "1" ]]; then
  bash "${SCRIPT_DIR}/score_verified_runs.sh" "${OUTPUT_FOLDER}"
fi

echo "[$(date -Is)] Physics-IQ Verified slot-dedup pipeline complete"
