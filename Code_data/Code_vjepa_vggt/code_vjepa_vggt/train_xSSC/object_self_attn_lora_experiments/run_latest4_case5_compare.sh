#!/usr/bin/env bash
# Run:
# GPU_ID=3 NUM_INFERENCE_STEPS=8 RUN_ROOT=/data/gaoya/agent-data/outputs/xssc_lora_latest_case5_compare/run_TAG \
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/run_latest4_case5_compare.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_ID="${GPU_ID:-3}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-8}"
INPUT_LIST="${INPUT_LIST:-${ROOT}/case5_latest_compare_inputs.txt}"
RUN_ROOT="${RUN_ROOT:?Set RUN_ROOT to the comparison output directory}"
GENERATION_ROOT="${RUN_ROOT}/generations"
LOG_ROOT="${RUN_ROOT}/logs"

if [[ "${GPU_ID}" == "4" ]]; then
  echo "GPU 4 is prohibited by workspace rules." >&2
  exit 2
fi
if [[ ! -s "${INPUT_LIST}" ]]; then
  echo "Input list is missing or empty: ${INPUT_LIST}" >&2
  exit 2
fi

METHOD_IDS=(
  object_only_step000529
  full_sa_step001000
  s_head59_step001000
  t_head70_step001000
)
CHECKPOINT_DIRS=(
  /data/gaoya/agent-data/checkpoints/xssc_object_self_attn_lora/object_only_gpu1_formal/formal_20260729T184553Z/checkpoints/interrupted-latest
  /data/gaoya/agent-data/checkpoints/xssc_object_self_attn_lora/full_sa_gpu12_ddp_resume/ddp2_resume_20260730T055600Z/checkpoints/step-001000
  /data/gaoya/agent-data/checkpoints/xssc_object_self_attn_lora/same_frame_s_head_full59_gpu05_ddp_resume/ddp2_resume_int32_20260730T061200Z/checkpoints/step-001000
  /data/gaoya/agent-data/checkpoints/xssc_object_self_attn_lora/common_t_head_full70_gpu67_ddp_resume/ddp2_resume_fixint32_20260730T060100Z/checkpoints/step-001000
)

mkdir -p "${GENERATION_ROOT}" "${LOG_ROOT}"
cp "${INPUT_LIST}" "${RUN_ROOT}/input_cases.txt"

for index in "${!METHOD_IDS[@]}"; do
  method_id="${METHOD_IDS[${index}]}"
  checkpoint_dir="${CHECKPOINT_DIRS[${index}]}"
  output_dir_name="${method_id}_steps${NUM_INFERENCE_STEPS}_512x896_ctx08_49f"
  log_path="${LOG_ROOT}/${method_id}.log"
  if [[ ! -s "${checkpoint_dir}/checkpoint.safetensors" ]]; then
    echo "Checkpoint is missing: ${checkpoint_dir}/checkpoint.safetensors" >&2
    exit 2
  fi
  printf '[%s] start method=%s checkpoint=%s\n' \
    "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "${method_id}" "${checkpoint_dir}"
  TEST_LIST="${INPUT_LIST}" \
    NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}" \
    STEP_OUTPUT_DIR_NAME="${output_dir_name}" \
    TRACE_ROOT="${GENERATION_ROOT}/_numeric_traces/${output_dir_name}" \
    bash "${ROOT}/run_infer_from_experiment.sh" \
      "${checkpoint_dir}" "${GPU_ID}" "${GENERATION_ROOT}" \
      >"${log_path}" 2>&1
  printf '[%s] complete method=%s log=%s\n' \
    "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "${method_id}" "${log_path}"
done

printf '[%s] all comparisons complete: %s\n' \
  "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "${GENERATION_ROOT}"
