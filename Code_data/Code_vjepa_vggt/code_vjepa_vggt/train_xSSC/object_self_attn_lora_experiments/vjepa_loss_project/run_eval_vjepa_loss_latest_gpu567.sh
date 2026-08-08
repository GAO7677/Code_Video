#!/usr/bin/env bash
# Run: RESUME_WATCHER_PID=<pid> bash run_eval_vjepa_loss_latest_gpu567.sh

set -u -o pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${PROJECT_DIR}/.." && pwd)"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
CONFIG="${ROOT}/xssc_lora_three_train_watch_config_with_t_head.json"
METHOD="full_sa_no_object_vjepa_loss"
CHECKPOINT_ROOT="/data/gaoya/agent-data/checkpoints/xssc_object_self_attn_lora/full_sa_no_object_gpu01_formal_vjepa_loss/20260805T180305Z/checkpoints"
WATCH_ROOT="/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_three_run_watch"
LOG_ROOT="${WATCH_ROOT}/logs/vjepa_loss_latest_gpu567"
RESUME_WATCHER_PID="${RESUME_WATCHER_PID:-}"
mkdir -p "${LOG_ROOT}"
export PYTHONNOUSERSITE=1

STEP="$(${PYTHON} - "${CHECKPOINT_ROOT}" <<'PY'
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
steps = []
for path in root.glob("step-*"):
    match = re.fullmatch(r"step-(\d+)", path.name)
    if not match:
        continue
    if not (path / "checkpoint.safetensors").is_file():
        continue
    if not (path / "training_state.pt").is_file():
        continue
    steps.append(int(match.group(1)))
if not steps:
    raise SystemExit("No complete checkpoint found")
print(max(steps))
PY
)"

resume_watcher() {
  if [[ -n "${RESUME_WATCHER_PID}" ]] && kill -0 "${RESUME_WATCHER_PID}" 2>/dev/null; then
    kill -CONT "${RESUME_WATCHER_PID}" 2>/dev/null || true
    echo "[cleanup] resumed watcher pid=${RESUME_WATCHER_PID}"
  fi
}
trap resume_watcher EXIT

echo "[latest] method=${METHOD} step=${STEP}"
echo "[generation] test_5 on GPU6; PhysicIQ shards on GPU5/7"
"${PYTHON}" "${ROOT}/run_missing_checkpoint_generation.py" \
  --config "${CONFIG}" --gpus 6 --methods "${METHOD}" --steps "${STEP}" \
  --test5-only 2>&1 | tee "${LOG_ROOT}/step-${STEP}_test5_gpu6.log" &
test5_pid=$!
"${PYTHON}" "${ROOT}/xssc_lora_physiciq_sharded_catchup.py" \
  --config "${CONFIG}" --gpus 5,7 --methods "${METHOD}" --steps "${STEP}" \
  2>&1 | tee "${LOG_ROOT}/step-${STEP}_physiciq_gpu57.log" &
phys_pid=$!

generation_status=0
wait "${test5_pid}" || generation_status=1
wait "${phys_pid}" || generation_status=1
if [[ "${generation_status}" -ne 0 ]]; then
  echo "[generation] one or more generation jobs failed" >&2
  exit 1
fi

echo "[metrics] CPU metrics"
"${PYTHON}" "${ROOT}/xssc_lora_checkpoint_filtered_cpu_metrics.py" \
  --config "${CONFIG}" --methods "${METHOD}" --steps "${STEP}" \
  --workers 4 --refresh 2>&1 | tee "${LOG_ROOT}/step-${STEP}_test5_cpu.log" &
test5_cpu_pid=$!
"${PYTHON}" "${ROOT}/xssc_lora_physiciq_parallel_metrics.py" \
  --config "${CONFIG}" --kind cpu --cpu-workers 4 --methods "${METHOD}" \
  --steps "${STEP}" --skip-locked --refresh-plots \
  2>&1 | tee "${LOG_ROOT}/step-${STEP}_physiciq_cpu.log" &
phys_cpu_pid=$!

echo "[metrics] test_5 GPU metrics: GPUs 5/6/7, three workers per GPU"
"${PYTHON}" "${ROOT}/xssc_lora_checkpoint_parallel_metrics.py" \
  --config "${CONFIG}" --gpus 5,6,7 --methods "${METHOD}" --steps "${STEP}" \
  --workers-per-gpu 3 --refresh \
  2>&1 | tee "${LOG_ROOT}/step-${STEP}_test5_gpu567x3.log" || true

echo "[metrics] PhysicIQ GPU metrics: GPUs 5/6/7, three workers per GPU"
"${PYTHON}" "${ROOT}/xssc_lora_physiciq_parallel_metrics.py" \
  --config "${CONFIG}" --kind gpu --gpus 5,6,7 --workers-per-gpu 3 \
  --methods "${METHOD}" --steps "${STEP}" --skip-locked --refresh-plots \
  2>&1 | tee "${LOG_ROOT}/step-${STEP}_physiciq_gpu567x3.log" || true

wait "${test5_cpu_pid}" || true
wait "${phys_cpu_pid}" || true
"${PYTHON}" "${ROOT}/build_xssc_lora_checkpoint_dashboard.py" \
  --config "${CONFIG}" 2>&1 | tee "${LOG_ROOT}/step-${STEP}_dashboard.log"
echo "[complete] method=${METHOD} step=${STEP}"
