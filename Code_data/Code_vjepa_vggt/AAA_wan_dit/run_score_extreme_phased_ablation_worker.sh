#!/usr/bin/env bash
set -uo pipefail

# GPU=0 NUM_WORKERS=7 bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_score_extreme_phased_ablation_worker.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:?set GPU}"
NUM_WORKERS="${NUM_WORKERS:-7}"
SEED="${SEED:-851}"
failed=0
task_index=0

for step_start in 0 10 20 30; do
  step_end=$((step_start + 10))
  for group in top bottom; do
    for model in wan_lora xssc physrvg; do
      if (( task_index % NUM_WORKERS == GPU )); then
        label="${model}_${group}_steps$(printf '%02d' "${step_start}")_$(printf '%02d' "${step_end}")"
        echo "[phased-worker] start ${label} on GPU${GPU} at $(date -u +%FT%TZ)"
        if MODEL="${model}" GROUP="${group}" GPU="${GPU}" SEED="${SEED}" \
          STEP_START="${step_start}" STEP_END="${step_end}" \
          bash "${SCRIPT_DIR}/run_score_extreme_head_ablation_job.sh"; then
          echo "[phased-worker] complete ${label} at $(date -u +%FT%TZ)"
        else
          status=$?
          failed=1
          echo "[phased-worker] failed ${label} status=${status} at $(date -u +%FT%TZ)" >&2
        fi
      fi
      task_index=$((task_index + 1))
    done
  done
done

exit "${failed}"
