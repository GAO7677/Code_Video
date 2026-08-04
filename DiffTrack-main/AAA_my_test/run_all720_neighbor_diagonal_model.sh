#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 GPU_ID {gt|lora|baseline}" >&2
  exit 2
fi

GPU="$1"
MODEL="$2"
PROJECT="/home/gaoya/Code_Video/DiffTrack-main"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
WORKER="${PROJECT}/AAA_my_test/capture_stable_heads_alltoken_qk_worker.py"
METRIC_ROOT="/data/gaoya/agent-data/outputs/three_model_all720_neighbor_diagonal_5case"
HEAT_ROOT="/data/gaoya/agent-data/outputs/three_model_all720_neighbor_diagonal_heatmaps_case001"
STATUS_ROOT="${METRIC_ROOT}/status"
LOG_ROOT="${METRIC_ROOT}/logs"

if [[ "${GPU}" == "4" || "${GPU}" == "6" || "${GPU}" == "7" ]]; then
  echo "GPU ${GPU} is reserved and will not be used." >&2
  exit 2
fi
if [[ "${MODEL}" != "gt" && "${MODEL}" != "lora" && "${MODEL}" != "baseline" ]]; then
  echo "MODEL must be gt, lora, or baseline" >&2
  exit 2
fi

mkdir -p "${STATUS_ROOT}" "${LOG_ROOT}" "${HEAT_ROOT}/logs"
rm -f "${STATUS_ROOT}/${MODEL}.complete" "${STATUS_ROOT}/${MODEL}.failed"
trap 'status=$?; if (( status != 0 )); then printf "%s\n" "$status" > "${STATUS_ROOT}/${MODEL}.failed"; fi' EXIT

COMBINATIONS=""
for block in $(seq 0 29); do
  for head in $(seq 0 23); do
    if [[ -n "${COMBINATIONS}" ]]; then COMBINATIONS+=","; fi
    COMBINATIONS+="${block}:${head}"
  done
done
mapfile -t LAYERS < <(seq 0 29)
CASES=(
  case_001_ball_roll
  case_002_puck_slide
  case_003_capsule_slide
  case_004_cylinder_topple
  case_005_box_slide
)
EXTRA=()
if [[ "${MODEL}" != "gt" ]]; then EXTRA+=(--analysis-no-cotracker); fi

run_capture() {
  local output="$1"
  local metrics_only="$2"
  shift 2
  CUDA_VISIBLE_DEVICES="${GPU}" \
  PYTHONNOUSERSITE=1 \
  PYTHONUNBUFFERED=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  ALLTOKEN_COMBINATIONS="${COMBINATIONS}" \
  ALLTOKEN_METRICS_ONLY="${metrics_only}" \
  PYTHONPATH="${PROJECT}:/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main:/home/gaoya/Code_Video/Code_data/Code_train/train_0419" \
  "${PYTHON}" "${WORKER}" \
    --model-kind "${MODEL}" \
    --worker-id 0 \
    --num-workers 1 \
    --output-dir "${output}" \
    --sampling-steps 40 \
    --analysis-matching-mode q_to_k \
    --analysis-layers "${LAYERS[@]}" \
    --analysis-step-indices 39 \
    --analysis-no-hidden \
    --analysis-no-video \
    --case-keys "$@" \
    --overwrite \
    "${EXTRA[@]}"
}

echo "[$(date -Is)] metrics start model=${MODEL} gpu=${GPU}"
run_capture "${METRIC_ROOT}/${MODEL}" 1 "${CASES[@]}" \
  2>&1 | tee "${LOG_ROOT}/${MODEL}_metrics_gpu${GPU}.log"
echo "[$(date -Is)] heatmap capture start model=${MODEL} gpu=${GPU}"
run_capture "${HEAT_ROOT}/${MODEL}" 0 case_001_ball_roll \
  2>&1 | tee "${HEAT_ROOT}/logs/${MODEL}_capture_gpu${GPU}.log"
printf 'model=%s\ngpu=%s\ncompleted=%s\n' \
  "${MODEL}" "${GPU}" "$(date -u +%FT%TZ)" > "${STATUS_ROOT}/${MODEL}.complete"
echo "ALL720_NEIGHBOR_MODEL_COMPLETE model=${MODEL} gpu=${GPU}"
