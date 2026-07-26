#!/usr/bin/env bash
set -euo pipefail

# Run in foreground:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_spatiotemporal_generalization_gpu34.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
TRAIN0419_ROOT=/home/gaoya/Code_Video/Code_data/Code_train/train_0419
PHYSRVG_ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_phys_papers_compare/PhysRVG-main
WAN_PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
ANALYSIS_PYTHON=/data/gaoya/miniconda3/envs/vjepa2/bin/python

OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/agent-data/outputs/wan_dit_block17_spatiotemporal_generalization}"
SOURCE_LIST="${SOURCE_LIST:-/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt}"
SEED_CASE="${SEED_CASE:-/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/0613pybullet_sample_001460_w002.json}"
GPU_WAN="${GPU_WAN:-3}"
GPU_PHYRVG="${GPU_PHYRVG:-4}"
ATTENTION_STEPS="${ATTENTION_STEPS:-5,15,25,35}"
ATTENTION_QUERY_CHUNK="${ATTENTION_QUERY_CHUNK:-128}"
RUN_MODE="${RUN_MODE:-both}"
SKIP_ANALYSIS="${SKIP_ANALYSIS:-0}"

WAN_ROOT=/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B
WAN_LORA_ROOT=/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500
MODEL_ID=/data/gaoya/ckpt/HappyP4nda-PhysRVG/Wan2.2-TI2V-5B-Diffusers
DIT_CHECKPOINT=/data/gaoya/ckpt/HappyP4nda-PhysRVG/dit/diffusion_pytorch_model.safetensors
LORA_CHECKPOINT=/data/gaoya/ckpt/HappyP4nda-PhysRVG/lora/checkpoint
NEGATIVE_PROMPT="模糊，低质量，变形，伪影，文字，水印，过曝，欠曝，颜色异常，几何扭曲，物体融化，物理不合理"

INPUT_ROOT="${OUTPUT_ROOT}/inputs"
STATISTICS_ROOT="${OUTPUT_ROOT}/statistics"
GENERATED_ROOT="${OUTPUT_ROOT}/generated"
LOG_ROOT="${OUTPUT_ROOT}/logs"
mkdir -p "${INPUT_ROOT}" "${STATISTICS_ROOT}" "${GENERATED_ROOT}" "${LOG_ROOT}"

"${ANALYSIS_PYTHON}" "${SCRIPT_DIR}/prepare_spatiotemporal_generalization_inputs.py" \
  --source-list "${SOURCE_LIST}" \
  --seed-case "${SEED_CASE}" \
  --output-dir "${INPUT_ROOT}"

INPUT_LIST="${INPUT_LIST_OVERRIDE:-${INPUT_ROOT}/combined_69_runs.txt}"
SEED_MAP="${INPUT_ROOT}/seed_map.json"

run_wan_lora() {
  env \
    PYTHONPATH="${PROJECT_ROOT}:${DIFFSYNTH_ROOT}:${TRAIN0419_ROOT}:${SCRIPT_DIR}" \
    CUDA_VISIBLE_DEVICES="${GPU_WAN}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${WAN_PYTHON}" "${SCRIPT_DIR}/capture_wan_lora_spatiotemporal_queries.py" \
    --attention-output-root "${STATISTICS_ROOT}" \
    --attention-block 17 \
    --attention-steps "${ATTENTION_STEPS}" \
    --attention-query-chunk "${ATTENTION_QUERY_CHUNK}" \
    --seed-map-json "${SEED_MAP}" \
    --weights-root "${WAN_LORA_ROOT}" \
    --input-json-list-path "${INPUT_LIST}" \
    --model-name wan_lora_block17_spatiotemporal_generalization \
    --wan-root "${WAN_ROOT}" \
    --output-root "${GENERATED_ROOT}/wan_lora" \
    --runtime-root "${GENERATED_ROOT}/wan_lora_runtime" \
    --device cuda --height 512 --width 896 --num-frames 49 \
    --context-frames 8 --conditioning-mode context_aware \
    --context-resize-mode crop --num-inference-steps 40 --cfg-scale 5.0 \
    --fps 30 --seed 42 --negative-prompt "${NEGATIVE_PROMPT}" --overwrite
}

run_physrvg() {
  env \
    CUDA_VISIBLE_DEVICES="${GPU_PHYRVG}" \
    PYTHONPATH="${PHYSRVG_ROOT}:${SCRIPT_DIR}" \
    PYTHONNOUSERSITE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${ANALYSIS_PYTHON}" "${SCRIPT_DIR}/capture_physrvg_spatiotemporal_queries.py" \
    --attention-output-root "${STATISTICS_ROOT}" \
    --attention-block 17 \
    --attention-steps "${ATTENTION_STEPS}" \
    --attention-query-chunk "${ATTENTION_QUERY_CHUNK}" \
    --seed-map-json "${SEED_MAP}" \
    --physrvg-root "${PHYSRVG_ROOT}" \
    --input-json-list-paths "${INPUT_LIST}" \
    --output-root "${GENERATED_ROOT}/physrvg" \
    --model-id "${MODEL_ID}" --dit-checkpoint "${DIT_CHECKPOINT}" \
    --lora-checkpoint "${LORA_CHECKPOINT}" --device cuda:0 \
    --height 512 --width 896 --num-frames 49 --fps 30 \
    --num-inference-steps 40 --guidance-scale 5.0 --seed 42 --force
}

status=0
declare -a pids=()
if [[ "${RUN_MODE}" == "both" || "${RUN_MODE}" == "wan_lora" ]]; then
  run_wan_lora >"${LOG_ROOT}/wan_lora.log" 2>&1 &
  pids+=("$!")
fi
if [[ "${RUN_MODE}" == "both" || "${RUN_MODE}" == "physrvg" ]]; then
  run_physrvg >"${LOG_ROOT}/physrvg.log" 2>&1 &
  pids+=("$!")
fi
if ((${#pids[@]} == 0)); then
  echo "RUN_MODE must be both, wan_lora, or physrvg; got ${RUN_MODE}" >&2
  exit 2
fi
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if ((status != 0)); then
  echo "At least one model failed; inspect ${LOG_ROOT}" >&2
  exit "${status}"
fi

if [[ "${SKIP_ANALYSIS}" == "1" ]]; then
  echo "statistics=${STATISTICS_ROOT}"
  exit 0
fi

PYTHONPATH="${SCRIPT_DIR}" \
"${ANALYSIS_PYTHON}" "${SCRIPT_DIR}/analyze_spatiotemporal_head_generalization.py" \
  --statistics-root "${STATISTICS_ROOT}" \
  --output-dir "${OUTPUT_ROOT}/analysis"

echo "analysis=${OUTPUT_ROOT}/analysis/generalization_summary.json"
