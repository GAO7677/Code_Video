#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: $0 GPU [extra capture arguments...]}"
shift

DIFFTRACK="/home/gaoya/Code_Video/DiffTrack-main"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
SCRIPT="${DIFFTRACK}/AAA_my_test/object_query_ablation_metrics/training_free_m1_control/capture_phase_b_top100_attention_overlays.py"
OUTPUT="/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1/training_free_m1_direct_enhancement_v2/seed90094_top100_attention_overlays"

if [[ "${GPU}" == "4" ]]; then
  echo "GPU 4 is forbidden by workspace policy" >&2
  exit 2
fi

mkdir -p "${OUTPUT}/logs"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
cd "${DIFFTRACK}"

exec "${PYTHON}" "${SCRIPT}" "$@"
