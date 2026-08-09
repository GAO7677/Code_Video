#!/usr/bin/env bash
set -euo pipefail

worker_id="${1:?usage: $0 WORKER_ID NUM_WORKERS PHYSICAL_GPU}"
num_workers="${2:?usage: $0 WORKER_ID NUM_WORKERS PHYSICAL_GPU}"
physical_gpu="${3:?usage: $0 WORKER_ID NUM_WORKERS PHYSICAL_GPU}"

if [[ "${physical_gpu}" == "4" ]]; then
  echo "GPU 4 is prohibited in this workspace." >&2
  exit 2
fi
if ! [[ "${worker_id}" =~ ^[0-9]+$ && "${num_workers}" =~ ^[1-9][0-9]*$ ]]; then
  echo "worker id and worker count must be non-negative integers" >&2
  exit 2
fi
if (( worker_id >= num_workers )); then
  echo "worker id must be smaller than worker count" >&2
  exit 2
fi

repo=/home/gaoya/Code_Video/DiffTrack-main
python_bin=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
batch_root=/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/attention_zero_seed47326
manifest="${batch_root}/cases_other10_6seeds.json"
fixed_root="${batch_root}/attention_matrix_ablations_v2"
tube_root="${batch_root}/attention_matrix_ablations_temporal_tube_v1"
log_root="${batch_root}/other10_6seed_logs"

mkdir -p "${log_root}"
exec > >(tee -a "${log_root}/worker${worker_id}_gpu${physical_gpu}.log") 2>&1
export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${repo}"

echo "[$(date -u +%FT%TZ)] worker ${worker_id}/${num_workers} GPU${physical_gpu} start Fixed"
"${python_bin}" -u AAA_my_test/run_legacy_ti2v_firstlatent_physiciq67_attention_zero_ablations.py \
  --worker-id "${worker_id}" \
  --num-workers "${num_workers}" \
  --manifest-path "${manifest}" \
  --output-root "${fixed_root}" \
  --top-counts 100 \
  --generate-missing-baselines

echo "[$(date -u +%FT%TZ)] worker ${worker_id}/${num_workers} GPU${physical_gpu} start Tube"
"${python_bin}" -u AAA_my_test/run_legacy_ti2v_temporal_object_tube_ablations.py \
  --all-samples \
  --worker-id "${worker_id}" \
  --num-workers "${num_workers}" \
  --manifest-path "${manifest}" \
  --output-root "${tube_root}" \
  --device cuda \
  --mask-modes \
    self_only incoming_only outgoing_only query_row key_value_column \
    cross_boundary row_and_column literal_kv_zero

echo "[$(date -u +%FT%TZ)] worker ${worker_id}/${num_workers} GPU${physical_gpu} complete"
