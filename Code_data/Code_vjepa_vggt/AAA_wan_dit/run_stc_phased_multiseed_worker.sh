#!/usr/bin/env bash
set -uo pipefail

# GPU=0 NUM_WORKERS=7 bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_stc_phased_multiseed_worker.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:?set GPU}"
NUM_WORKERS="${NUM_WORKERS:-7}"
INPUT_LIST="${INPUT_LIST:-${SCRIPT_DIR}/common22_public_head_ablation_case025.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/agent-data/outputs/wan_dit_common22_public_head_ablation_case025}"
CHECK_PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
NEW_SEEDS=(3278 11395 20379 28221 32098)
failed=0
task_index=0

run_task() {
  local seed="$1" step_start="$2" step_end="$3" role="$4" model="$5"
  local variant="${role}_steps$(printf '%02d' "${step_start}")_$(printf '%02d' "${step_end}")"
  local label="${model}_seed$(printf '%06d' "${seed}")_${role}_steps$(printf '%02d' "${step_start}")_$(printf '%02d' "${step_end}")"
  if (( task_index % NUM_WORKERS == GPU )); then
    if "${CHECK_PYTHON}" "${SCRIPT_DIR}/check_phased_ablation_task_complete.py" \
      --output-root "${OUTPUT_ROOT}" --model "${model}" --seed "${seed}" \
      --variant "${variant}"; then
      echo "[stc-multiseed-worker] skip-complete ${label} on GPU${GPU}"
      task_index=$((task_index + 1))
      return
    fi
    echo "[stc-multiseed-worker] start ${label} on GPU${GPU} at $(date -u +%FT%TZ)"
    if MODEL="${model}" ROLE="${role}" GPU="${GPU}" SEED="${seed}" \
      STEP_START="${step_start}" STEP_END="${step_end}" \
      INPUT_LIST="${INPUT_LIST}" OUTPUT_ROOT="${OUTPUT_ROOT}" \
      bash "${SCRIPT_DIR}/run_common22_public_head_ablation_job.sh"; then
      echo "[stc-multiseed-worker] complete ${label} at $(date -u +%FT%TZ)"
    else
      status=$?
      failed=1
      echo "[stc-multiseed-worker] failed ${label} status=${status} at $(date -u +%FT%TZ)" >&2
    fi
  fi
  task_index=$((task_index + 1))
}

# Seed 851 already has the four non-overlapping windows; only add [0,15).
for role in S T C; do
  for model in wan_lora xssc physrvg; do
    run_task 851 0 15 "${role}" "${model}"
  done
done

for seed in "${NEW_SEEDS[@]}"; do
  for step_range in 0:10 10:20 20:30 30:40 0:15; do
    step_start="${step_range%%:*}"
    step_end="${step_range##*:}"
    for role in S T C; do
      for model in wan_lora xssc physrvg; do
        run_task "${seed}" "${step_start}" "${step_end}" "${role}" "${model}"
      done
    done
  done
done

echo "[stc-multiseed-worker] GPU${GPU} exhausted assigned tasks; total_matrix=${task_index}"
exit "${failed}"
