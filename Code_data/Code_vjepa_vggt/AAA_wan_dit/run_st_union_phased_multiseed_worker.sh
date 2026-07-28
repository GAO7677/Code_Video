#!/usr/bin/env bash
set -uo pipefail

# GPU=0 NUM_WORKERS=7 bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_st_union_phased_multiseed_worker.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:?set GPU}"
NUM_WORKERS="${NUM_WORKERS:-7}"
INPUT_LIST="${INPUT_LIST:-${SCRIPT_DIR}/common22_public_head_ablation_case025.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/agent-data/outputs/wan_dit_common22_public_head_ablation_case025}"
CHECK_PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
SEEDS=(851 3278 11395 20379 28221 32098)
STEP_RANGES=(0:5 5:10 5:15 0:10 0:15 10:20 20:30 30:40)
failed=0
task_index=0

for step_range in "${STEP_RANGES[@]}"; do
  step_start="${step_range%%:*}"
  step_end="${step_range##*:}"
  variant="ST_steps$(printf '%02d' "${step_start}")_$(printf '%02d' "${step_end}")"
  for seed in "${SEEDS[@]}"; do
    for model in wan_lora xssc physrvg; do
      if (( task_index % NUM_WORKERS == GPU )); then
        label="${model}_seed$(printf '%06d' "${seed}")_${variant}"
        if "${CHECK_PYTHON}" "${SCRIPT_DIR}/check_phased_ablation_task_complete.py" \
          --output-root "${OUTPUT_ROOT}" --model "${model}" --seed "${seed}" \
          --variant "${variant}"; then
          echo "[st-union-worker] skip-complete ${label} on GPU${GPU}"
          task_index=$((task_index + 1))
          continue
        fi
        echo "[st-union-worker] start ${label} on GPU${GPU} at $(date -u +%FT%TZ)"
        if MODEL="${model}" ROLE=ST GPU="${GPU}" SEED="${seed}" \
          STEP_START="${step_start}" STEP_END="${step_end}" \
          INPUT_LIST="${INPUT_LIST}" OUTPUT_ROOT="${OUTPUT_ROOT}" \
          bash "${SCRIPT_DIR}/run_common22_public_head_ablation_job.sh"; then
          echo "[st-union-worker] complete ${label} at $(date -u +%FT%TZ)"
        else
          status=$?
          failed=1
          echo "[st-union-worker] failed ${label} status=${status} at $(date -u +%FT%TZ)" >&2
        fi
      fi
      task_index=$((task_index + 1))
    done
  done
done

echo "[st-union-worker] GPU${GPU} exhausted assigned tasks; total_matrix=${task_index}"
exit "${failed}"
