#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 RUN_ID" >&2
  exit 2
fi

RUN_ID="$1"
REPO="/home/gaoya/Code_Video/DiffTrack-main"
SCRIPT_DIR="${REPO}/AAA_my_test/object_query_ablation_metrics"
PYTHON="/data/gaoya/miniconda3/envs/wan/bin/python"
EXPERIMENT_ROOT="/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1"
RUNTIME_ROOT="${EXPERIMENT_ROOT}/stage4_runtime"
STATE_DIR="${RUNTIME_ROOT}/runs/${RUN_ID}"
OUTPUT_ROOT="${EXPERIMENT_ROOT}/stage4_temporal_v1"
METRICS_ROOT="${EXPERIMENT_ROOT}/stage4_metrics"
MANIFEST="${RUNTIME_ROOT}/stage4_manifest.json"
RANKING="${EXPERIMENT_ROOT}/head_scopes_latest3350_with_random100.json"
TAG="s039r3350_stage4v1"
SMOKE_GPU="${SMOKE_GPU:-3}"
CASE="0613pybullet_sample_001460_w002"
VARIANT="single_object__object_A__self_future__top100_${TAG}"
VARIANT_DIR="${OUTPUT_ROOT}/${CASE}/seed_47326/${VARIANT}"

mkdir -p "${STATE_DIR}"
cd "${REPO}"
trap 'touch "${STATE_DIR}/smoke.failed"' ERR

wait_for_gpu_free() {
  local used
  while true; do
    used="$(nvidia-smi --id="${SMOKE_GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    if [[ "${used}" =~ ^[0-9]+$ ]] && (( used < 2000 )); then
      sleep 5
      used="$(nvidia-smi --id="${SMOKE_GPU}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
      if [[ "${used}" =~ ^[0-9]+$ ]] && (( used < 2000 )); then
        return
      fi
    fi
    echo "[stage4-smoke-wait-gpu] gpu=${SMOKE_GPU} used_mib=${used}"
    sleep 30
  done
}

"${PYTHON}" "${SCRIPT_DIR}/prepare_stage4_runtime.py" --output-dir "${RUNTIME_ROOT}"
"${PYTHON}" -m unittest \
  AAA_my_test.test_temporal_directional_ablations \
  AAA_my_test.object_query_ablation_metrics.test_object_query_information_flow_stage2 \
  AAA_my_test.test_legacy_attention_matrix_ablations

wait_for_gpu_free
export CUDA_VISIBLE_DEVICES="${SMOKE_GPU}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"${PYTHON}" -u AAA_my_test/run_legacy_ti2v_temporal_object_tube_ablations.py \
  --case "${CASE}" \
  --seed 47326 \
  --task-index 0 \
  --manifest-path "${MANIFEST}" \
  --head-ranking-path "${RANKING}" \
  --ranking-tag "${TAG}" \
  --output-root "${OUTPUT_ROOT}" \
  --device cuda \
  --head-scopes top100 \
  --mask-modes self_future \
  --record-dose

bash "${SCRIPT_DIR}/bench_missing.sh" \
  "${OUTPUT_ROOT}/${CASE}/seed_47326" \
  --gpu "${SMOKE_GPU}" \
  --output-base "${METRICS_ROOT}" \
  --stages fast,trajectory,survival,complete25 \
  --skip-vbench

"${PYTHON}" "${SCRIPT_DIR}/validate_stage4_smoke.py" \
  "${VARIANT_DIR}" \
  --metrics-root "${METRICS_ROOT}" \
  > "${STATE_DIR}/smoke_validation.json"
touch "${STATE_DIR}/smoke.passed"
echo "[stage4-smoke-pass] ${RUN_ID}"
