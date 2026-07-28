#!/usr/bin/env bash
set -uo pipefail

# GPU=0 NUM_WORKERS=7 bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_stc_steps05_15_multiseed_worker.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:?set GPU}"
NUM_WORKERS="${NUM_WORKERS:-7}"
INPUT_LIST="${INPUT_LIST:-${SCRIPT_DIR}/common22_public_head_ablation_case025.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/agent-data/outputs/wan_dit_common22_public_head_ablation_case025}"
SEEDS=(851 3278 11395 20379 28221 32098)
failed=0
task_index=0

for seed in "${SEEDS[@]}"; do
  for role in S T C; do
    for model in wan_lora xssc physrvg; do
      if (( task_index % NUM_WORKERS == GPU )); then
        label="${model}_seed$(printf '%06d' "${seed}")_${role}_steps05_15"
        echo "[stc-steps05-15-worker] start ${label} on GPU${GPU} at $(date -u +%FT%TZ)"
        if MODEL="${model}" ROLE="${role}" GPU="${GPU}" SEED="${seed}" \
          STEP_START=5 STEP_END=15 INPUT_LIST="${INPUT_LIST}" \
          OUTPUT_ROOT="${OUTPUT_ROOT}" \
          bash "${SCRIPT_DIR}/run_common22_public_head_ablation_job.sh"; then
          echo "[stc-steps05-15-worker] complete ${label} at $(date -u +%FT%TZ)"
        else
          status=$?
          failed=1
          echo "[stc-steps05-15-worker] failed ${label} status=${status} at $(date -u +%FT%TZ)" >&2
        fi
      fi
      task_index=$((task_index + 1))
    done
  done
done

echo "[stc-steps05-15-worker] GPU${GPU} exhausted assigned tasks; total_matrix=${task_index}"
exit "${failed}"
