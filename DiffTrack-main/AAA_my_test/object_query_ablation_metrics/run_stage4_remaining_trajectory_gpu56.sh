#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/gaoya/Code_Video/DiffTrack-main"
SCRIPT_DIR="${REPO_ROOT}/AAA_my_test/object_query_ablation_metrics"
EXPERIMENT_ROOT="/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1"
CASE="physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed-ball-and-block-fall_motion_to_end"
SEED_DIR="${EXPERIMENT_ROOT}/stage4_temporal_v1/${CASE}/seed_90094"
CANONICAL_BASE="${EXPERIMENT_ROOT}/stage4_metrics/head_scope_trajectory"
SCRATCH_ROOT="${EXPERIMENT_ROOT}/stage4_metrics_parallel_scratch/gpu56_remaining_20260813"
LOG_ROOT="${EXPERIMENT_ROOT}/logs/stage4_remaining_trajectory_gpu56_20260813"
PYTHON="/data/gaoya/miniconda3/envs/wan/bin/python"
SCRIPT="${SCRIPT_DIR}/compute_head_scope_trajectory_metrics.py"

mkdir -p "${SCRATCH_ROOT}" "${LOG_ROOT}"
cd "${REPO_ROOT}"

run_gpu5() {
  env CUDA_VISIBLE_DEVICES=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PYTHON}" -u "${SCRIPT}" "${SEED_DIR}" \
      --output-base "${SCRATCH_ROOT}/gpu5" --device cuda:0 \
      --head-scopes top100 \
      > "${LOG_ROOT}/gpu5_top100.log" 2>&1
  env CUDA_VISIBLE_DEVICES=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PYTHON}" -u "${SCRIPT}" "${SEED_DIR}" \
      --output-base "${SCRATCH_ROOT}/gpu5" --device cuda:0 \
      --head-scopes random100_layer_matched_draw0 \
      > "${LOG_ROOT}/gpu5_random100.log" 2>&1
}

run_gpu6() {
  env CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PYTHON}" -u "${SCRIPT}" "${SEED_DIR}" \
      --output-base "${SCRATCH_ROOT}/gpu6" --device cuda:0 \
      --head-scopes bottom100 \
      > "${LOG_ROOT}/gpu6_bottom100.log" 2>&1
}

run_gpu5 &
gpu5_pid=$!
run_gpu6 &
gpu6_pid=$!
wait "${gpu5_pid}"
wait "${gpu6_pid}"

canonical_seed="${CANONICAL_BASE}/${CASE}/seed_90094"
mkdir -p "${canonical_seed}/tracks" "${canonical_seed}/overlays"
for worker in gpu5 gpu6; do
  worker_seed="${SCRATCH_ROOT}/${worker}/${CASE}/seed_90094"
  cp -aln "${worker_seed}/tracks/." "${canonical_seed}/tracks/"
  cp -aln "${worker_seed}/overlays/." "${canonical_seed}/overlays/"
done

# Rebuild one canonical report after all isolated track caches have been merged.
env CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "${PYTHON}" -u "${SCRIPT}" "${SEED_DIR}" \
    --output-base "${CANONICAL_BASE}" --device cuda:0 \
    > "${LOG_ROOT}/consolidate.log" 2>&1

touch "${LOG_ROOT}/COMPLETE"
echo "[done] Stage-4 remaining PhysicIQ trajectory metrics merged"
