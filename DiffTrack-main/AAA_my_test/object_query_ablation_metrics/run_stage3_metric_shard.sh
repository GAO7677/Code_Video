#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 GPU_ID SHARD_INDEX SHARD_COUNT CURRENT_SEED_LIST" >&2
  exit 2
fi

GPU_ID="$1"
SHARD_INDEX="$2"
SHARD_COUNT="$3"
CURRENT_SEED_LIST="$4"

REPO_ROOT="/home/gaoya/Code_Video/DiffTrack-main"
RESULT_ROOT="/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1/stage3_discovery_videos"
OUTPUT_BASE="/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1/stage3_metrics"
BENCH_MISSING="${REPO_ROOT}/AAA_my_test/object_query_ablation_metrics/bench_missing.sh"

if [[ "${GPU_ID}" == "4" ]]; then
  echo "GPU4 is forbidden" >&2
  exit 2
fi
if [[ ! -f "${CURRENT_SEED_LIST}" ]]; then
  echo "Missing shard list: ${CURRENT_SEED_LIST}" >&2
  exit 2
fi

cd "${REPO_ROOT}"

failures=0
echo "[current-pass] GPU${GPU_ID} shard ${SHARD_INDEX}/${SHARD_COUNT}"
while IFS= read -r seed_dir; do
  [[ -n "${seed_dir}" ]] || continue
  echo "[seed-start] ${seed_dir}"
  if bash "${BENCH_MISSING}" "${seed_dir}" \
    --gpu "${GPU_ID}" \
    --output-base "${OUTPUT_BASE}" \
    --stages trajectory,survival; then
    echo "[seed-complete] ${seed_dir}"
  else
    echo "[seed-failed] ${seed_dir}"
    failures=$((failures + 1))
  fi
done < "${CURRENT_SEED_LIST}"
echo "[current-pass-done] failures=${failures}"

echo "[wait-generation]"
while tmux has-session -t oqif_stage3_gpu2 2>/dev/null \
  || tmux has-session -t oqif_stage3_gpu3 2>/dev/null; do
  sleep 60
done

echo "[final-pass] GPU${GPU_ID} shard ${SHARD_INDEX}/${SHARD_COUNT}"
bash "${BENCH_MISSING}" "${RESULT_ROOT}" \
  --gpu "${GPU_ID}" \
  --output-base "${OUTPUT_BASE}" \
  --stages fast,trajectory,survival \
  --num-shards "${SHARD_COUNT}" \
  --shard-index "${SHARD_INDEX}"
echo "[all-done] GPU${GPU_ID}"
