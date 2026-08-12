#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 GPU_ID WORKER_ID RUN_ID" >&2
  exit 2
fi

GPU_ID="$1"
WORKER_ID="$2"
RUN_ID="$3"
[[ "${GPU_ID}" =~ ^(2|3)$ ]] || { echo "Stage 4 worker GPU must be 2 or 3" >&2; exit 2; }
[[ "${WORKER_ID}" =~ ^(0|1)$ ]] || { echo "worker must be 0 or 1" >&2; exit 2; }

REPO="/home/gaoya/Code_Video/DiffTrack-main"
SCRIPT_DIR="${REPO}/AAA_my_test/object_query_ablation_metrics"
PYTHON="/data/gaoya/miniconda3/envs/wan/bin/python"
EXPERIMENT_ROOT="/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1"
RUNTIME_ROOT="${EXPERIMENT_ROOT}/stage4_runtime"
STATE_DIR="${RUNTIME_ROOT}/runs/${RUN_ID}"
OUTPUT_ROOT="${EXPERIMENT_ROOT}/stage4_temporal_v1"
METRICS_ROOT="${EXPERIMENT_ROOT}/stage4_metrics"
MANIFEST="${RUNTIME_ROOT}/stage4_manifest.json"
MANIFEST_001460="${RUNTIME_ROOT}/stage4_manifest_001460.json"
RANKING="${EXPERIMENT_ROOT}/head_scopes_latest3350_with_random100.json"
TAG="s039r3350_stage4v1"
NUM_WORKERS=2

mkdir -p "${STATE_DIR}"
cd "${REPO}"
trap 'touch "${STATE_DIR}/gpu${GPU_ID}.failed"' ERR

wait_for_marker() {
  local marker="$1"
  while [[ ! -f "${STATE_DIR}/${marker}" ]]; do
    if [[ -f "${STATE_DIR}/smoke.failed" ]] \
      || [[ -f "${STATE_DIR}/gpu2.failed" ]] \
      || [[ -f "${STATE_DIR}/gpu3.failed" ]]; then
      echo "[stage4-abort] failure marker detected while waiting for ${marker}" >&2
      exit 1
    fi
    sleep 10
  done
}

wait_for_gpu_free() {
  local used
  while true; do
    used="$(nvidia-smi --id="${GPU_ID}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    if [[ "${used}" =~ ^[0-9]+$ ]] && (( used < 2000 )); then
      sleep 5
      used="$(nvidia-smi --id="${GPU_ID}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
      if [[ "${used}" =~ ^[0-9]+$ ]] && (( used < 2000 )); then
        return
      fi
    fi
    echo "[stage4-wait-gpu] gpu=${GPU_ID} used_mib=${used}"
    sleep 30
  done
}

wait_for_marker "smoke.passed"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
wait_for_gpu_free

"${PYTHON}" -u AAA_my_test/run_legacy_ti2v_temporal_object_tube_ablations.py \
  --all-samples \
  --manifest-path "${MANIFEST}" \
  --head-ranking-path "${RANKING}" \
  --ranking-tag "${TAG}" \
  --output-root "${OUTPUT_ROOT}" \
  --device cuda \
  --worker-id "${WORKER_ID}" \
  --num-workers "${NUM_WORKERS}" \
  --head-scopes top100 bottom100 random100_layer_matched_draw0 \
  --mask-modes \
    self_same self_future self_past \
    incoming_same incoming_future incoming_past \
    outgoing_same outgoing_future outgoing_past \
  --record-dose
touch "${STATE_DIR}/directional_gpu${GPU_ID}.done"
wait_for_marker "directional_gpu2.done"
wait_for_marker "directional_gpu3.done"
wait_for_gpu_free

"${PYTHON}" -u AAA_my_test/run_legacy_ti2v_temporal_object_tube_ablations.py \
  --all-samples \
  --manifest-path "${MANIFEST_001460}" \
  --head-ranking-path "${RANKING}" \
  --ranking-tag "${TAG}" \
  --output-root "${OUTPUT_ROOT}" \
  --device cuda \
  --worker-id "${WORKER_ID}" \
  --num-workers "${NUM_WORKERS}" \
  --head-scopes top100 bottom100 random100_layer_matched_draw0 \
  --mask-modes self_only incoming_only outgoing_only \
  --record-dose
touch "${STATE_DIR}/alltime_gpu${GPU_ID}.done"
wait_for_marker "alltime_gpu2.done"
wait_for_marker "alltime_gpu3.done"
wait_for_gpu_free

"${PYTHON}" -u AAA_my_test/run_legacy_ti2v_temporal_object_tube_ablations.py \
  --case 0613pybullet_sample_001460_w002 \
  --seed 47326 \
  --manifest-path "${MANIFEST}" \
  --head-ranking-path "${RANKING}" \
  --ranking-tag "${TAG}" \
  --output-root "${OUTPUT_ROOT}" \
  --device cuda \
  --worker-id "${WORKER_ID}" \
  --num-workers "${NUM_WORKERS}" \
  --head-scopes all720 \
  --mask-modes \
    self_same self_future self_past \
    incoming_same incoming_future incoming_past \
    outgoing_same outgoing_future outgoing_past \
  --record-dose
touch "${STATE_DIR}/sentinel_gpu${GPU_ID}.done"
wait_for_marker "sentinel_gpu2.done"
wait_for_marker "sentinel_gpu3.done"
wait_for_gpu_free

bash "${SCRIPT_DIR}/bench_missing.sh" "${OUTPUT_ROOT}" \
  --gpu "${GPU_ID}" \
  --output-base "${METRICS_ROOT}" \
  --stages fast,trajectory,survival,complete25 \
  --skip-vbench \
  --num-shards 2 \
  --shard-index "${WORKER_ID}"
touch "${STATE_DIR}/metrics_gpu${GPU_ID}.done"
echo "[stage4-worker-complete] GPU${GPU_ID} worker=${WORKER_ID} run=${RUN_ID}"
