#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box
LAUNCHER="${ROOT}/run_context_guidance_physiq3_step1000.sh"
GPU_PAIR="${GPU_PAIR:-6,6}"
PRIMARY_GPU="${GPU_PAIR%%,*}"
MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB:-44000}"
POLL_SECONDS="${POLL_SECONDS:-30}"
STABLE_FREE_POLLS="${STABLE_FREE_POLLS:-4}"
BLOCKING_TMUX_SESSION="${BLOCKING_TMUX_SESSION-mixdataset_physiq_objres_sweep_gpu6_20260713}"
SMOKE_OUTPUT_ROOT="${SMOKE_OUTPUT_ROOT:-/data/gaoya/agent-data/outputs/AAA_physv/context_guidance_physiq3_step1000_20260713_smoke2}"
FULL_OUTPUT_ROOT="${FULL_OUTPUT_ROOT:-/data/gaoya/agent-data/outputs/AAA_physv/context_guidance_physiq3_step1000_20260713}"
QUEUE_LOG="${QUEUE_LOG:-/data/gaoya/agent-data/outputs/AAA_physv/context_guidance_physiq3_step1000_20260713_queue.log}"

mkdir -p "$(dirname "${QUEUE_LOG}")"
exec > >(tee -a "${QUEUE_LOG}") 2>&1

echo "[context-guidance-queue] waiting for GPU ${PRIMARY_GPU}"
stable_free_count=0
while true; do
  if [ -n "${BLOCKING_TMUX_SESSION}" ] && tmux has-session -t "${BLOCKING_TMUX_SESSION}" 2>/dev/null; then
    echo "[context-guidance-queue] blocking_tmux_active=${BLOCKING_TMUX_SESSION}"
    stable_free_count=0
    sleep "${POLL_SECONDS}"
    continue
  fi
  free_gpu_mib="$(
    nvidia-smi --id="${PRIMARY_GPU}" --query-gpu=memory.free \
      --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:]'
  )"
  echo "[context-guidance-queue] free_gpu_mib=${free_gpu_mib} stable=${stable_free_count}/${STABLE_FREE_POLLS}"
  if [ -n "${free_gpu_mib}" ] && [ "${free_gpu_mib}" -ge "${MIN_FREE_GPU_MIB}" ]; then
    stable_free_count=$((stable_free_count + 1))
    if [ "${stable_free_count}" -ge "${STABLE_FREE_POLLS}" ]; then
      break
    fi
  else
    stable_free_count=0
  fi
  sleep "${POLL_SECONDS}"
done

echo "[context-guidance-queue] running 2-step adaptive smoke"
GPU_PAIR="${GPU_PAIR}" \
INPUT_LIST="${ROOT}/context_guidance_smoke_case.txt" \
OUTPUT_ROOT="${SMOKE_OUTPUT_ROOT}" \
MODES=adaptive_context_guard \
NUM_INFERENCE_STEPS=2 \
MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB}" \
bash "${LAUNCHER}"

smoke_json="$(
  find "${SMOKE_OUTPUT_ROOT}" -type f \
    -name 'physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed.json' \
    | head -n 1
)"
if [ -z "${smoke_json}" ]; then
  echo "ERROR: smoke result JSON not found under ${SMOKE_OUTPUT_ROOT}" >&2
  exit 1
fi

/home/gaoya/miniconda3/envs/wan-cu128/bin/python - "${smoke_json}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open("r", encoding="utf-8") as handle:
    payload = json.load(handle)
guidance = payload["context_guidance"]
coverage = guidance["coverage"]
assert guidance["mode"] == "adaptive_context_guard", guidance
assert guidance["guard_applied"] is True, guidance
assert coverage["coverage_complete"] is True, coverage
assert guidance["effective_cfg_scale"] == 2.5, guidance
assert payload["guidance"] == 2.5, payload["guidance"]
assert payload["model_args"]["cfg_scale"] == 2.5, payload["model_args"]
print(f"[context-guidance-queue] smoke assertions passed: {path}")
PY

echo "[context-guidance-queue] running full seven-mode experiment"
GPU_PAIR="${GPU_PAIR}" \
OUTPUT_ROOT="${FULL_OUTPUT_ROOT}" \
MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB}" \
bash "${LAUNCHER}"

echo "[context-guidance-queue] building aligned H.264 comparisons"
PYTHONNOUSERSITE=1 /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  "${ROOT}/build_context_guidance_comparisons.py" \
  --case-list "${ROOT}/context_guidance_physiq_cases.txt" \
  --method "text_video_baseline=${FULL_OUTPUT_ROOT}/text_video_baseline" \
  --method "positive_text_off=${FULL_OUTPUT_ROOT}/positive_text_off" \
  --method "negative_text_off=${FULL_OUTPUT_ROOT}/negative_text_off" \
  --method "video_only=${FULL_OUTPUT_ROOT}/video_only" \
  --method "low_text_cfg=${FULL_OUTPUT_ROOT}/low_text_cfg" \
  --method "anti_duplicate_prompt=${FULL_OUTPUT_ROOT}/anti_duplicate_prompt" \
  --method "adaptive_context_guard=${FULL_OUTPUT_ROOT}/adaptive_context_guard" \
  --output-dir "${FULL_OUTPUT_ROOT}/comparisons"

echo "[context-guidance-queue] measuring future object-count increases"
PYTHONNOUSERSITE=1 \
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
CUDA_VISIBLE_DEVICES="${PRIMARY_GPU}" \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  "${ROOT}/analyze_context_guidance_object_counts.py" \
  --case-list "${ROOT}/context_guidance_physiq_cases.txt" \
  --method "text_video_baseline=${FULL_OUTPUT_ROOT}/text_video_baseline" \
  --method "positive_text_off=${FULL_OUTPUT_ROOT}/positive_text_off" \
  --method "negative_text_off=${FULL_OUTPUT_ROOT}/negative_text_off" \
  --method "video_only=${FULL_OUTPUT_ROOT}/video_only" \
  --method "low_text_cfg=${FULL_OUTPUT_ROOT}/low_text_cfg" \
  --method "anti_duplicate_prompt=${FULL_OUTPUT_ROOT}/anti_duplicate_prompt" \
  --method "adaptive_context_guard=${FULL_OUTPUT_ROOT}/adaptive_context_guard" \
  --output-dir "${FULL_OUTPUT_ROOT}/object_count_analysis" \
  --device cuda:0

echo "[context-guidance-queue] complete: ${FULL_OUTPUT_ROOT}"
