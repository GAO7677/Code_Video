#!/usr/bin/env bash
# Run in tmux:
# tmux new-session -d -s xssc_lora_smoke_gate \
#   "bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/run_smoke_then_formal_gpu1267.sh"
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_LAUNCHER="${ROOT}/run_train_from_config.sh"
INFER_LAUNCHER="${ROOT}/run_infer_from_experiment.sh"
STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
SMOKE_TAG="smoke_${STAMP}"
FORMAL_TAG="formal_${STAMP}"
FORMAL_SESSION="xssc_lora_formal_${STAMP}"

SMOKE_OUTPUT_ROOT=/data/gaoya/agent-data/checkpoints/xssc_object_self_attn_lora_smoke
SMOKE_INFER_ROOT="/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_smoke/${SMOKE_TAG}"
CONTROL_ROOT="/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_control/${STAMP}"
TEST_SOURCE=/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt
SMOKE_TEST_LIST="${CONTROL_ROOT}/test_5_first_case.txt"
mkdir -p "${CONTROL_ROOT}/logs/smoke" "${CONTROL_ROOT}/logs/infer" "${SMOKE_INFER_ROOT}"

NAMES=(object_only full_sa s_head t_head)
GPUS=(1 2 6 7)
SMOKE_CONFIGS=(
  "${ROOT}/configs/smoke_object_only_gpu1.json"
  "${ROOT}/configs/smoke_full_sa_gpu2.json"
  "${ROOT}/configs/smoke_s_head_gpu6.json"
  "${ROOT}/configs/smoke_t_head_gpu7.json"
)
SMOKE_EXPERIMENTS=(
  smoke_object_only_gpu1
  smoke_full_sa_gpu2
  smoke_same_frame_s_head_full59_gpu6
  smoke_common_t_head_full70_gpu7
)
FORMAL_CONFIGS=(
  "${ROOT}/configs/formal_object_only_gpu1.json"
  "${ROOT}/configs/formal_full_sa_gpu2.json"
  "${ROOT}/configs/formal_s_head_gpu6.json"
  "${ROOT}/configs/formal_t_head_gpu7.json"
)

assert_gpu_idle() {
  local gpu_id="$1"
  local pids
  pids="$(nvidia-smi -i "${gpu_id}" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
  if [[ -n "${pids}" ]]; then
    echo "GPU ${gpu_id} is no longer idle; active PIDs: ${pids}" >&2
    return 1
  fi
}

for gpu_id in "${GPUS[@]}"; do
  if [[ "${gpu_id}" == "4" ]]; then
    echo "GPU 4 is prohibited by workspace rules." >&2
    exit 2
  fi
  assert_gpu_idle "${gpu_id}"
done
if [[ ! -s "${TEST_SOURCE}" ]]; then
  echo "Missing smoke inference source list: ${TEST_SOURCE}" >&2
  exit 2
fi
awk 'NF {print; exit}' "${TEST_SOURCE}" > "${SMOKE_TEST_LIST}"

echo "[gate] smoke training tag=${SMOKE_TAG}"
declare -a train_pids=()
for index in "${!NAMES[@]}"; do
  name="${NAMES[$index]}"
  log="${CONTROL_ROOT}/logs/smoke/${name}.log"
  echo "[gate] launch smoke ${name} on GPU ${GPUS[$index]} -> ${log}"
  bash "${TRAIN_LAUNCHER}" "${SMOKE_CONFIGS[$index]}" \
    --run-tag "${SMOKE_TAG}" > "${log}" 2>&1 &
  train_pids+=("$!")
done

status=0
for index in "${!train_pids[@]}"; do
  if ! wait "${train_pids[$index]}"; then
    echo "[gate] smoke training failed: ${NAMES[$index]}" >&2
    status=1
  fi
done
if (( status != 0 )); then
  echo "[gate] formal training blocked by smoke training failure." >&2
  exit 1
fi

declare -a checkpoint_dirs=()
for index in "${!NAMES[@]}"; do
  checkpoint_dir="${SMOKE_OUTPUT_ROOT}/${SMOKE_EXPERIMENTS[$index]}/${SMOKE_TAG}/checkpoints/step-000001"
  if [[ ! -s "${checkpoint_dir}/checkpoint.safetensors" ]]; then
    echo "[gate] missing smoke checkpoint: ${checkpoint_dir}" >&2
    exit 1
  fi
  if [[ ! -s "${checkpoint_dir}/training_state.pt" ]]; then
    echo "[gate] missing smoke training state: ${checkpoint_dir}" >&2
    exit 1
  fi
  checkpoint_dirs+=("${checkpoint_dir}")
done

echo "[gate] smoke checkpoints complete; launching one-case inference"
declare -a infer_pids=()
for index in "${!NAMES[@]}"; do
  name="${NAMES[$index]}"
  method="${name}_smoke_step-000001"
  log="${CONTROL_ROOT}/logs/infer/${name}.log"
  echo "[gate] launch inference ${name} on GPU ${GPUS[$index]} -> ${log}"
  TEST_LIST="${SMOKE_TEST_LIST}" \
  NUM_INFERENCE_STEPS=2 \
  STEP_OUTPUT_DIR_NAME="${method}" \
  bash "${INFER_LAUNCHER}" \
    "${checkpoint_dirs[$index]}" \
    "${GPUS[$index]}" \
    "${SMOKE_INFER_ROOT}" > "${log}" 2>&1 &
  infer_pids+=("$!")
done

status=0
for index in "${!infer_pids[@]}"; do
  if ! wait "${infer_pids[$index]}"; then
    echo "[gate] smoke inference failed: ${NAMES[$index]}" >&2
    status=1
  fi
done
if (( status != 0 )); then
  echo "[gate] formal training blocked by smoke inference failure." >&2
  exit 1
fi

for name in "${NAMES[@]}"; do
  method_dir="${SMOKE_INFER_ROOT}/${name}_smoke_step-000001"
  if ! find "${method_dir}" -type f -name '*.mp4' -size +0c -print -quit | grep -q .; then
    echo "[gate] no generated video found under ${method_dir}" >&2
    exit 1
  fi
  if ! find "${method_dir}" -type f -name '*.json' -size +0c -print -quit | grep -q .; then
    echo "[gate] no inference JSON found under ${method_dir}" >&2
    exit 1
  fi
done

echo "[gate] all smoke inference outputs validated"
for gpu_id in "${GPUS[@]}"; do
  assert_gpu_idle "${gpu_id}"
done
if tmux has-session -t "${FORMAL_SESSION}" 2>/dev/null; then
  echo "[gate] formal tmux session already exists: ${FORMAL_SESSION}" >&2
  exit 1
fi

for index in "${!NAMES[@]}"; do
  name="${NAMES[$index]}"
  log="${CONTROL_ROOT}/logs/formal_${name}.log"
  command="set -o pipefail; bash '${TRAIN_LAUNCHER}' '${FORMAL_CONFIGS[$index]}' --run-tag '${FORMAL_TAG}' 2>&1 | tee '${log}'"
  if (( index == 0 )); then
    tmux new-session -d -s "${FORMAL_SESSION}" -n "${name}" -c "${ROOT}"
  else
    tmux new-window -d -t "${FORMAL_SESSION}" -n "${name}" -c "${ROOT}"
  fi
  tmux send-keys -t "${FORMAL_SESSION}:${name}" "${command}" C-m
done

{
  echo "status=formal_started"
  echo "formal_session=${FORMAL_SESSION}"
  echo "formal_tag=${FORMAL_TAG}"
  echo "smoke_tag=${SMOKE_TAG}"
  echo "smoke_inference_root=${SMOKE_INFER_ROOT}"
  echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${CONTROL_ROOT}/gate_result.txt"

echo "[gate] formal training started in tmux session: ${FORMAL_SESSION}"
echo "[gate] attach with: tmux attach -t ${FORMAL_SESSION}"
echo "[gate] control artifacts: ${CONTROL_ROOT}"
