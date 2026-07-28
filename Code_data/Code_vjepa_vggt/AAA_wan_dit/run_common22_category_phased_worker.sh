#!/usr/bin/env bash
set -uo pipefail

# GPU=0 NUM_WORKERS=7 bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_common22_category_phased_worker.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:?set GPU}"
NUM_WORKERS="${NUM_WORKERS:-7}"
SEED="${SEED:-851}"
INPUT_LIST="${INPUT_LIST:-${SCRIPT_DIR}/common22_public_head_ablation_case025.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/agent-data/outputs/wan_dit_common22_public_head_ablation_case025}"
failed=0
task_index=0

for step_start in 0 10 20 30; do
  step_end=$((step_start + 10))
  for role in S T P C G; do
    for model in wan_lora xssc physrvg; do
      if (( task_index % NUM_WORKERS == GPU )); then
        label="${model}_${role}_steps$(printf '%02d' "${step_start}")_$(printf '%02d' "${step_end}")"
        echo "[category-phased-worker] start ${label} on GPU${GPU} at $(date -u +%FT%TZ)"
        if MODEL="${model}" ROLE="${role}" GPU="${GPU}" SEED="${SEED}" \
          STEP_START="${step_start}" STEP_END="${step_end}" \
          INPUT_LIST="${INPUT_LIST}" OUTPUT_ROOT="${OUTPUT_ROOT}" \
          bash "${SCRIPT_DIR}/run_common22_public_head_ablation_job.sh"; then
          echo "[category-phased-worker] complete ${label} at $(date -u +%FT%TZ)"
        else
          status=$?
          failed=1
          echo "[category-phased-worker] failed ${label} status=${status} at $(date -u +%FT%TZ)" >&2
        fi
      fi
      task_index=$((task_index + 1))
    done
  done
done

exit "${failed}"
