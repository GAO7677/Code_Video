#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 1 ]] || { echo "usage: $0 GPU_ID" >&2; exit 2; }
GPU_ID="$1"
[[ "${GPU_ID}" == "5" || "${GPU_ID}" == "6" ]] || {
  echo "GPU_ID must be 5 or 6" >&2
  exit 2
}

REPO_ROOT="/home/gaoya/Code_Video/DiffTrack-main"
SCRIPT_DIR="${REPO_ROOT}/AAA_my_test/object_query_ablation_metrics"
EXPERIMENT_ROOT="/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1"
RESULT_ROOT="${EXPERIMENT_ROOT}/stage4_temporal_v1"
OUTPUT_BASE="${EXPERIMENT_ROOT}/stage4_metrics"
SURVIVAL_OUTPUT="${OUTPUT_BASE}/head_scope_trajectory"
SAM_PYTHON="/data/gaoya/miniconda3/envs/physxnet_mpm_env/bin/python"
LOG_ROOT="${EXPERIMENT_ROOT}/logs/stage4_remaining_metrics_gpu56_20260813"

wait_for_free_memory() {
  local required_mib="$1"
  local free_mib
  while true; do
    free_mib="$(nvidia-smi -i "${GPU_ID}" --query-gpu=memory.free \
      --format=csv,noheader,nounits | tr -d ' ')"
    if (( free_mib >= required_mib )); then
      echo "[memory-ready] gpu=${GPU_ID} free=${free_mib}MiB required=${required_mib}MiB"
      return 0
    fi
    echo "[memory-wait] gpu=${GPU_ID} free=${free_mib}MiB required=${required_mib}MiB"
    sleep 30
  done
}

run_survival() {
  local seed_dir="$1"
  local log_name="$2"
  wait_for_free_memory 10000
  env CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    PYTHONPATH="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/Code_data/Code_try0526${PYTHONPATH:+:${PYTHONPATH}}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${SAM_PYTHON}" -u "${SCRIPT_DIR}/compute_head_scope_object_survival_metrics.py" \
      "${seed_dir}" --output-base "${SURVIVAL_OUTPUT}" \
      --device cuda:0 --batch-size 1 \
      2>&1 | tee "${LOG_ROOT}/${log_name}"
}

mkdir -p "${LOG_ROOT}"
cd "${REPO_ROOT}"

if [[ "${GPU_ID}" == "5" ]]; then
  seed_dir="${RESULT_ROOT}/0613pybullet_sample_001460_w002/seed_90094"
  run_survival "${seed_dir}" "gpu5_seed90094_survival.log"

  # Complete25 has a higher observed peak than survival. Keep it queued until
  # the co-located training process leaves a conservative 16 GiB margin.
  wait_for_free_memory 16000
  bash "${SCRIPT_DIR}/bench.sh" \
    "${seed_dir}/metrics_inventory_all_generated.json" \
    --gpu 5 --output-base "${OUTPUT_BASE}/head_scope_complete25" --no-aggregate \
    2>&1 | tee "${LOG_ROOT}/gpu5_seed90094_complete25.log"
else
  seed_dir="${RESULT_ROOT}/physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed-ball-and-block-fall_motion_to_end/seed_90094"
  run_survival "${seed_dir}" "gpu6_physiq_seed90094_survival.log"
fi

echo "[done] gpu=${GPU_ID} remaining Stage-4 metric tasks complete"
