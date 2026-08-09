#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: $0 GPU SHARD_INDEX [NUM_SHARDS]}"
SHARD_INDEX="${2:?usage: $0 GPU SHARD_INDEX [NUM_SHARDS]}"
NUM_SHARDS="${3:-2}"

ROOT=/home/gaoya/Code_Video/DiffTrack-main
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
PREPARE="${ROOT}/AAA_my_test/prepare_legacy_physiciq67_ablation_vbench.py"
BENCH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/bench.py
OUTPUT=/data/gaoya/agent-data/outputs/legacy_physiciq67_attention_ablation_vbench
INDEX="${OUTPUT}/index"
LOCK="${OUTPUT}/prepare.lock"
METRICS=(
  vbench_subject_consistency
  vbench_background_consistency
  vbench_temporal_flickering
  vbench_motion_smoothness
  vbench_dynamic_degree
  vbench_aesthetic_quality
  vbench_imaging_quality
)

if [[ "${GPU}" == "4" ]]; then
  echo "GPU 4 is forbidden by /home/gaoya/AGENTS.md" >&2
  exit 2
fi

mkdir -p "${OUTPUT}"
echo "[vbench:wait] completed-video snapshot waits for fixed and tube generation"
while pgrep -u gaoya -f 'run_legacy_ti2v_firstlatent_physiciq67_attention_zero_ablations.py|run_legacy_ti2v_temporal_object_tube_ablations.py' >/dev/null; do
  sleep 30
done

echo "[vbench:wait] gpu=${GPU} shard=${SHARD_INDEX}/${NUM_SHARDS}"
while true; do
  used=$(nvidia-smi -i "${GPU}" --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{sum += $1} END {print sum + 0}')
  if (( used < 4096 )); then
    break
  fi
  sleep 30
done

(
  flock -x 9
  if [[ ! -f "${OUTPUT}/snapshot.json" ]]; then
    "${PYTHON}" "${PREPARE}" --output-root "${OUTPUT}"
  fi
) 9>"${LOCK}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/Code_data/Code_try0526${PYTHONPATH:+:${PYTHONPATH}}"

for metric in "${METRICS[@]}"; do
  echo "[vbench:start] gpu=${GPU} shard=${SHARD_INDEX}/${NUM_SHARDS} metric=${metric}"
  "${PYTHON}" "${BENCH}" \
    --metric "${metric}" \
    --result-root "${INDEX}" \
    --num-shards "${NUM_SHARDS}" \
    --shard-index "${SHARD_INDEX}" \
    --vbench-output-root "${OUTPUT}/raw" \
    --vbench-device cuda
  echo "[vbench:done] gpu=${GPU} shard=${SHARD_INDEX}/${NUM_SHARDS} metric=${metric}"
done

touch "${OUTPUT}/shard_${SHARD_INDEX}_of_${NUM_SHARDS}.complete"
