#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
PROJECT=/home/gaoya/Code_Video/TextOCVP-PyBullet-smoke
INDEX_ROOT=/data/gaoya/AAA_test_video/0623_savi/indices
PROBE_ROOT=/data/gaoya/AAA_test_video/0623_savi/outputs/memory_probe
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-/data/gaoya/AAA_test_video/0623_savi/experiments/comparison_3way_216x384_slot256_accum2_val500_${RUN_TAG}}"
WANDB_PROJECT="${WANDB_PROJECT:-textocvp_savi_stage1}"
WANDB_GROUP="${WANDB_GROUP:-$(basename "${RUN_ROOT}")}"
EFFECTIVE_BATCH="${EFFECTIVE_BATCH:-16}"
VALIDATION_FREQUENCY_STEPS="${VALIDATION_FREQUENCY_STEPS:-500}"
EPOCHS="${EPOCHS:-1000}"

if [[ ! -f "${PROBE_ROOT}/selected_micro_global_batch.txt" ]]; then
  echo "Missing successful memory-probe selection" >&2
  exit 1
fi
MICRO_BATCH="$(<"${PROBE_ROOT}/selected_micro_global_batch.txt")"
if (( EFFECTIVE_BATCH % MICRO_BATCH != 0 )); then
  echo "Selected micro batch ${MICRO_BATCH} does not divide effective batch ${EFFECTIVE_BATCH}" >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
printf '%s\n' \
  "gpus=0,1,2,3" \
  "micro_global_batch=${MICRO_BATCH}" \
  "accumulation_steps=$((EFFECTIVE_BATCH / MICRO_BATCH))" \
  "effective_global_batch=${EFFECTIVE_BATCH}" \
  "epochs=${EPOCHS}" \
  "validation_frequency_steps=${VALIDATION_FREQUENCY_STEPS}" \
  "checkpoint_frequency_steps=${VALIDATION_FREQUENCY_STEPS}" \
  "resolution_hw=216,384" \
  "num_slots=8" \
  "slot_dim=256" \
  "wandb_project=${WANDB_PROJECT}" \
  "wandb_group=${WANDB_GROUP}" \
  > "${RUN_ROOT}/run_config.txt"

for mode in pybullet kubric mixed; do
  echo "[three-way] starting ${mode}"
  "${PYTHON}" "${PROJECT}/launch_stage1_experiment.py" \
    --dataset-mode "${mode}" \
    --index-root "${INDEX_ROOT}" \
    --output-dir "${RUN_ROOT}/${mode}" \
    --gpus 0,1,2,3 \
    --micro-global-batch-size "${MICRO_BATCH}" \
    --effective-batch-size "${EFFECTIVE_BATCH}" \
    --epochs "${EPOCHS}" \
    --validation-frequency-steps "${VALIDATION_FREQUENCY_STEPS}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-group "${WANDB_GROUP}" \
    2>&1 | tee "${RUN_ROOT}/${mode}.launch.log"
  "${PYTHON}" "${PROJECT}/compare_stage1_convergence.py" \
    --run-root "${RUN_ROOT}" \
    --output-dir "${RUN_ROOT}/comparison"
done

echo "[three-way] complete: ${RUN_ROOT}"
