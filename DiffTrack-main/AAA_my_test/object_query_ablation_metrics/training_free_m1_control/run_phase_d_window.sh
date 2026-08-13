#!/usr/bin/env bash
set -euo pipefail

if (( $# != 1 )); then
  echo "Usage: PHASE_D_ALPHA=<winner> $0 GPU_ID" >&2
  exit 2
fi

GPU_ID="$1"
ALPHA="${PHASE_D_ALPHA:-}"
SHARD_COUNT="${PHASE_D_SHARD_COUNT:-1}"
SHARD_INDEX="${PHASE_D_SHARD_INDEX:-0}"
DRY_RUN="${PHASE_DRY_RUN:-0}"

if ! [[ "${GPU_ID}" =~ ^[0-9]+$ ]] || [[ "${GPU_ID}" == "4" ]]; then
  echo "GPU_ID must be one physical GPU other than forbidden GPU 4." >&2
  exit 2
fi
if [[ -z "${ALPHA}" ]]; then
  echo "PHASE_D_ALPHA is required; use the frozen Phase-B/C fixed-gain winner." >&2
  exit 2
fi
if ! [[ "${ALPHA}" =~ ^(0([.][0-9]+)?|1([.]0+)?)$ ]] ||
   ! awk -v value="${ALPHA}" 'BEGIN { exit !(value > 0 && value <= 1) }'; then
  echo "PHASE_D_ALPHA must be numeric and in (0,1]." >&2
  exit 2
fi
if ! [[ "${SHARD_COUNT}" =~ ^[1-9][0-9]*$ ]] ||
   ! [[ "${SHARD_INDEX}" =~ ^[0-9]+$ ]] ||
   (( SHARD_INDEX >= SHARD_COUNT )); then
  echo "Invalid shard: index=${SHARD_INDEX}, count=${SHARD_COUNT}." >&2
  exit 2
fi
if [[ "${DRY_RUN}" != "0" && "${DRY_RUN}" != "1" ]]; then
  echo "PHASE_DRY_RUN must be 0 or 1." >&2
  exit 2
fi

PYTHON_BIN="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
REPO_ROOT="/home/gaoya/Code_Video/DiffTrack-main"
CONTROL_DIR="${REPO_ROOT}/AAA_my_test/object_query_ablation_metrics/training_free_m1_control"
RUNNER="${CONTROL_DIR}/run_m1_direct_scaling_phase_bd.py"
EXPERIMENT_ROOT="/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1"
OUTPUT_ROOT="${EXPERIMENT_ROOT}/training_free_m1_direct_enhancement_v2"
MANIFEST="${EXPERIMENT_ROOT}/training_free_top100_m23_guidance_v1/guidance_grid_manifest.json"
RANKING="${EXPERIMENT_ROOT}/head_scopes_latest3350_with_random100.json"
TRACKS="${EXPERIMENT_ROOT}/stage4_temporal_v1"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="/data/gaoya/agent-data/cache/huggingface"
export TORCH_HOME="/data/gaoya/agent-data/cache/torch"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

cases=(
  "0613pybullet_sample_001460_w002"
  "0613pybullet_sample_000331_w001"
  "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed-ball-and-block-fall_motion_to_end"
)
seeds=(47326 42)
windows=(
  "0:9"
  "0:19"
  "0:39"
)
dry_args=()
if [[ "${DRY_RUN}" == "1" ]]; then
  dry_args+=(--dry-run)
fi

unit_index=0
for case_name in "${cases[@]}"; do
  for seed_value in "${seeds[@]}"; do
    current_unit=${unit_index}
    unit_index=$((unit_index + 1))
    if (( current_unit % SHARD_COUNT != SHARD_INDEX )); then
      continue
    fi
    for window in "${windows[@]}"; do
      IFS=: read -r denoise_start denoise_end <<< "${window}"
      echo "[$(date -u +%FT%TZ)] Phase D/window unit=${current_unit}/6 GPU=${GPU_ID} case=${case_name} seed=${seed_value} alpha=${ALPHA} scope=all_time denoise=${denoise_start}..${denoise_end}"
      "${PYTHON_BIN}" "${RUNNER}" \
        --phase-label phase_d \
        --case "${case_name}" \
        --seed "${seed_value}" \
        --region object_A \
        --alpha "${ALPHA}" \
        --time-scope all_time \
        --denoise-start "${denoise_start}" \
        --denoise-end "${denoise_end}" \
        --cfg-scale 5 \
        --sampling-steps 40 \
        --manifest-path "${MANIFEST}" \
        --head-ranking-path "${RANKING}" \
        --tracks-root "${TRACKS}" \
        --output-root "${OUTPUT_ROOT}" \
        --device cuda \
        --record-dose \
        "${dry_args[@]}"
    done
  done
done

