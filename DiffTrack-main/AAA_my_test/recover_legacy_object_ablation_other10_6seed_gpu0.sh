#!/usr/bin/env bash
set -euo pipefail

repo=/home/gaoya/Code_Video/DiffTrack-main
python_bin=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
batch_root=/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/attention_zero_seed47326
manifest="${batch_root}/cases_other10_6seeds.json"
fixed_root="${batch_root}/attention_matrix_ablations_v2"
tube_root="${batch_root}/attention_matrix_ablations_temporal_tube_v1"
log="${batch_root}/other10_6seed_logs/recovery_worker0_gpu0.log"

exec > >(tee -a "${log}") 2>&1
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${repo}"

echo "[$(date -u +%FT%TZ)] fill every remaining baseline"
"${python_bin}" -u AAA_my_test/run_legacy_ti2v_firstlatent_physiciq67_attention_zero_ablations.py \
  --worker-id 0 --num-workers 1 \
  --manifest-path "${manifest}" \
  --output-root "${fixed_root}" \
  --baselines-only

echo "[$(date -u +%FT%TZ)] scan and fill every remaining Fixed Top100 task"
"${python_bin}" -u AAA_my_test/run_legacy_ti2v_firstlatent_physiciq67_attention_zero_ablations.py \
  --worker-id 0 --num-workers 1 \
  --manifest-path "${manifest}" \
  --output-root "${fixed_root}" \
  --top-counts 100

echo "[$(date -u +%FT%TZ)] resume Tube worker 0/5"
"${python_bin}" -u AAA_my_test/run_legacy_ti2v_temporal_object_tube_ablations.py \
  --all-samples --worker-id 0 --num-workers 5 \
  --manifest-path "${manifest}" \
  --output-root "${tube_root}" \
  --device cuda \
  --mask-modes \
    self_only incoming_only outgoing_only query_row key_value_column \
    cross_boundary row_and_column literal_kv_zero

echo "[$(date -u +%FT%TZ)] recovery worker0 complete"
