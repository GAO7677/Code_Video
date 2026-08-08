#!/usr/bin/env bash
set -euo pipefail

worker_id="${1:?usage: $0 WORKER_ID NUM_WORKERS PHYSICAL_GPU}"
num_workers="${2:?usage: $0 WORKER_ID NUM_WORKERS PHYSICAL_GPU}"
physical_gpu="${3:?usage: $0 WORKER_ID NUM_WORKERS PHYSICAL_GPU}"

if [[ "$physical_gpu" == "4" ]]; then
  echo "GPU 4 is prohibited in this workspace." >&2
  exit 2
fi

python_bin=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
runner=/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/run_legacy_ti2v_firstlatent_physiciq67_attention_zero_ablations.py
requested_manifest=/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/attention_zero_seed47326/cases.json
requested_output=/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/attention_zero_seed47326/attention_matrix_ablations_v2

CUDA_VISIBLE_DEVICES="$physical_gpu" "$python_bin" -u "$runner" \
  --worker-id "$worker_id" \
  --num-workers "$num_workers"

CUDA_VISIBLE_DEVICES="$physical_gpu" "$python_bin" -u "$runner" \
  --worker-id "$worker_id" \
  --num-workers "$num_workers" \
  --manifest-path "$requested_manifest" \
  --output-root "$requested_output"
