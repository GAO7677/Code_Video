#!/usr/bin/env bash
# Run step-500 and step-1000 on nine deterministic training samples.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
OUTPUT_ROOT="/data/gaoya/agent-data/outputs/full_sa_no_object_xssc_loss_train_cases"
CHECKPOINT_ROOT="/data/gaoya/agent-data/checkpoints/xssc_feature_loss/full_sa_no_object_xssc_loss_dinov3_movic_step50000/formal_gpu01/checkpoints"
INPUT_LIST="${OUTPUT_ROOT}/inputs/cases.txt"
INFERENCE_ROOT="${OUTPUT_ROOT}/inference"
GPU_ID="${GPU_ID:-1}"

if [[ "${GPU_ID}" == "4" ]]; then
  echo "GPU4 is prohibited by workspace rules." >&2
  exit 2
fi
if [[ "$(nvidia-smi -i "${GPU_ID}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" -gt 2000 ]]; then
  echo "GPU${GPU_ID} is already in use; refusing to overlap a new model process." >&2
  exit 2
fi

export PYTHONNOUSERSITE=1
export PYTHONPATH="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt"
mkdir -p "${OUTPUT_ROOT}/logs" "${INFERENCE_ROOT}"
"${PYTHON}" "${PROJECT_DIR}/prepare_full_sa_no_object_xssc_train_cases.py" --gpu "${GPU_ID}"
"${PYTHON}" "${PROJECT_DIR}/build_full_sa_no_object_xssc_train_case_gallery.py"
trap '"${PYTHON}" "${PROJECT_DIR}/build_full_sa_no_object_xssc_train_case_gallery.py" || true' EXIT

for step in 000500 001000; do
  checkpoint="${CHECKPOINT_ROOT}/step-${step}"
  output_name="step-${step}_steps40_512x896_ctx08_49f"
  echo "[$(date -u +%FT%TZ)] start ${output_name} on GPU${GPU_ID}"
  TEST_LIST="${INPUT_LIST}" \
  NUM_INFERENCE_STEPS=40 \
  STEP_OUTPUT_DIR_NAME="${output_name}" \
  TRACE_ROOT="${OUTPUT_ROOT}/numeric_traces/${output_name}" \
  bash "${EXPERIMENT_ROOT}/run_infer_from_experiment.sh" \
    "${checkpoint}" "${GPU_ID}" "${INFERENCE_ROOT}" \
    2>&1 | tee -a "${OUTPUT_ROOT}/logs/${output_name}.log"
  "${PYTHON}" "${PROJECT_DIR}/build_full_sa_no_object_xssc_train_case_gallery.py"
  echo "[$(date -u +%FT%TZ)] finish ${output_name}"
done
