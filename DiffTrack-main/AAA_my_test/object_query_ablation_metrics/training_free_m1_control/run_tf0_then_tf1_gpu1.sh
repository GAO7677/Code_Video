#!/usr/bin/env bash
set -euo pipefail

TF_GPU="${TF_GPU:-1}"
TF_SHARD_COUNT="${TF_SHARD_COUNT:-1}"
TF_SHARD_INDEX="${TF_SHARD_INDEX:-0}"
TF_UNIT_INDICES="${TF_UNIT_INDICES:-}"
if [[ "${TF_GPU}" == "4" ]]; then
  echo "Refusing to run on forbidden GPU 4." >&2
  exit 2
fi
if [[ "${CUDA_VISIBLE_DEVICES:-}" != "${TF_GPU}" ]]; then
  echo "Refusing to run: set CUDA_VISIBLE_DEVICES=${TF_GPU} exactly." >&2
  exit 2
fi
if ! [[ "${TF_SHARD_COUNT}" =~ ^[1-9][0-9]*$ ]] ||
   ! [[ "${TF_SHARD_INDEX}" =~ ^[0-9]+$ ]] ||
   (( TF_SHARD_INDEX >= TF_SHARD_COUNT )); then
  echo "Invalid shard: index=${TF_SHARD_INDEX}, count=${TF_SHARD_COUNT}." >&2
  exit 2
fi
if [[ -n "${TF_UNIT_INDICES}" ]] && ! [[ "${TF_UNIT_INDICES}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "Invalid TF_UNIT_INDICES=${TF_UNIT_INDICES}; expected comma-separated integers." >&2
  exit 2
fi

PYTHON_BIN="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
REPO_ROOT="/home/gaoya/Code_Video/DiffTrack-main"
WORK_DIR="${REPO_ROOT}/AAA_my_test/object_query_ablation_metrics"
CONTROL_DIR="${WORK_DIR}/training_free_m1_control"
OUTPUT_ROOT="/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1/training_free_m1_control_v1"
MANIFEST="/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1/training_free_top100_m23_guidance_v1/guidance_grid_manifest.json"
RANKING="/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1/head_scopes_latest3350_with_random100.json"
TRACKS="/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1/stage4_temporal_v1"
SOFT_RUNNER="${CONTROL_DIR}/run_m1_soft_scaling.py"
GUIDANCE_RUNNER="${WORK_DIR}/run_top100_m1_perturbed_attention_guidance.py"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="/data/gaoya/agent-data/cache/huggingface"
export TORCH_HOME="/data/gaoya/agent-data/cache/torch"

cd "${WORK_DIR}"

tf0_failed() {
  exit_code=$?
  "${PYTHON_BIN}" "${CONTROL_DIR}/write_tf0_failure.py" \
    --root "${OUTPUT_ROOT}" \
    --exit-code "${exit_code}"
  exit "${exit_code}"
}
trap tf0_failed ERR

"${PYTHON_BIN}" -m py_compile \
  "${GUIDANCE_RUNNER}" \
  "${SOFT_RUNNER}" \
  "${CONTROL_DIR}/validate_tf0.py" \
  "${CONTROL_DIR}/build_tf1_inventory.py"
"${PYTHON_BIN}" -m unittest \
  test_top100_m1_perturbed_attention_guidance.py \
  training_free_m1_control/test_m1_soft_scaling.py

if [[ -f "${OUTPUT_ROOT}/tf0/PASS.json" && ! -f "${OUTPUT_ROOT}/tf0/FAIL.json" ]]; then
  echo "[$(date -u +%FT%TZ)] Reusing completed TF-0 gate."
  "${PYTHON_BIN}" "${CONTROL_DIR}/validate_tf0.py" --root "${OUTPUT_ROOT}"
else
  rm -f "${OUTPUT_ROOT}/tf0/PASS.json" "${OUTPUT_ROOT}/tf0/FAIL.json"

  SMOKE_CASE="0613pybullet_sample_001460_w002"
  "${PYTHON_BIN}" "${SOFT_RUNNER}" \
    --case "${SMOKE_CASE}" \
    --seed 47326 \
    --region object_A \
    --alpha 0 \
    --reference-mode clean \
    --manifest-path "${MANIFEST}" \
    --head-ranking-path "${RANKING}" \
    --tracks-root "${TRACKS}" \
    --output-root "${OUTPUT_ROOT}/soft_scaling" \
    --device cuda \
    --overwrite

  "${PYTHON_BIN}" "${SOFT_RUNNER}" \
    --case "${SMOKE_CASE}" \
    --seed 47326 \
    --region object_A \
    --alpha -1 \
    --reference-mode stage3 \
    --manifest-path "${MANIFEST}" \
    --head-ranking-path "${RANKING}" \
    --tracks-root "${TRACKS}" \
    --output-root "${OUTPUT_ROOT}/soft_scaling" \
    --device cuda \
    --record-dose \
    --overwrite

  for alpha in 0 -1; do
    extra_args=()
    if [[ "${alpha}" == "-1" ]]; then
      extra_args+=(--record-dose --audit-decomposition)
    fi
    "${PYTHON_BIN}" "${SOFT_RUNNER}" \
      --case "${SMOKE_CASE}" \
      --seed 47326 \
      --region object_A \
      --alpha "${alpha}" \
      --manifest-path "${MANIFEST}" \
      --head-ranking-path "${RANKING}" \
      --tracks-root "${TRACKS}" \
      --output-root "${OUTPUT_ROOT}/soft_scaling" \
      --device cuda \
      --overwrite \
      "${extra_args[@]}"
  done

  "${PYTHON_BIN}" "${CONTROL_DIR}/validate_tf0.py" --root "${OUTPUT_ROOT}"
fi
trap - ERR

cases=(
  "0613pybullet_sample_001460_w002"
  "0613pybullet_sample_000331_w001"
  "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed-ball-and-block-fall_motion_to_end"
)
seeds=(47326 42)
alphas=(-1 -0.5 0 0.5 1)
lambdas=(-1 -0.5)

unit_index=0
for case_name in "${cases[@]}"; do
  for seed_value in "${seeds[@]}"; do
    current_unit_index=${unit_index}
    unit_index=$((unit_index + 1))
    if [[ -n "${TF_UNIT_INDICES}" ]]; then
      if [[ ",${TF_UNIT_INDICES}," != *",${current_unit_index},"* ]]; then
        continue
      fi
    elif (( current_unit_index % TF_SHARD_COUNT != TF_SHARD_INDEX )); then
        continue
    fi
    echo "[$(date -u +%FT%TZ)] shard ${TF_SHARD_INDEX}/${TF_SHARD_COUNT}: ${case_name} seed=${seed_value}"
    for alpha in "${alphas[@]}"; do
      "${PYTHON_BIN}" "${SOFT_RUNNER}" \
        --case "${case_name}" \
        --seed "${seed_value}" \
        --region object_A \
        --alpha "${alpha}" \
        --manifest-path "${MANIFEST}" \
        --head-ranking-path "${RANKING}" \
        --tracks-root "${TRACKS}" \
        --output-root "${OUTPUT_ROOT}/soft_scaling" \
        --device cuda \
        --record-dose
    done
    for lambda in "${lambdas[@]}"; do
      "${PYTHON_BIN}" "${GUIDANCE_RUNNER}" \
        --case "${case_name}" \
        --seed "${seed_value}" \
        --target-scope single_object \
        --region object_A \
        --flow m1 \
        --time-scope all_time \
        --pag-scale "${lambda}" \
        --cfg-scale 5 \
        --sampling-steps 40 \
        --manifest-path "${MANIFEST}" \
        --head-ranking-path "${RANKING}" \
        --tracks-root "${TRACKS}" \
        --output-root "${OUTPUT_ROOT}/contrast_raw" \
        --device cuda \
        --record-dose
    done
  done
done

"${PYTHON_BIN}" "${CONTROL_DIR}/build_tf1_inventory.py"
