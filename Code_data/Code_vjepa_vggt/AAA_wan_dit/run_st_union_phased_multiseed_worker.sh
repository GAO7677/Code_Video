#!/usr/bin/env bash
set -uo pipefail

# GPU=0 NUM_WORKERS=7 bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_st_union_phased_multiseed_worker.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU="${GPU:?set GPU}"
NUM_WORKERS="${NUM_WORKERS:-7}"
INPUT_LIST="${INPUT_LIST:-${SCRIPT_DIR}/common22_public_head_ablation_case025.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/agent-data/outputs/wan_dit_common22_public_head_ablation_case025}"
BARRIER_ROOT="${BARRIER_ROOT:?set BARRIER_ROOT}"
CHECK_PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
SEEDS=(851 3278 11395 20379 28221 32098)
STEP_RANGES=(0:5 5:10 5:15 0:10 0:15 10:20 20:30 30:40)
failed=0
total_tasks=0
mkdir -p "${BARRIER_ROOT}"

for seed in "${SEEDS[@]}"; do
  seed_key="$(printf '%06d' "${seed}")"
  task_index=0
  seed_failed=0
  # Finish the full page row for one seed before advancing. ST is scheduled first.
  for role in ST S T; do
    for step_range in "${STEP_RANGES[@]}"; do
      step_start="${step_range%%:*}"
      step_end="${step_range##*:}"
      variant="${role}_steps$(printf '%02d' "${step_start}")_$(printf '%02d' "${step_end}")"
      for model in wan_lora xssc physrvg; do
        if (( task_index % NUM_WORKERS == GPU )); then
          label="${model}_seed$(printf '%06d' "${seed}")_${variant}"
          if "${CHECK_PYTHON}" "${SCRIPT_DIR}/check_phased_ablation_task_complete.py" \
            --output-root "${OUTPUT_ROOT}" --model "${model}" --seed "${seed}" \
            --variant "${variant}"; then
            echo "[st-union-worker] skip-complete ${label} on GPU${GPU}"
          else
            echo "[st-union-worker] start ${label} on GPU${GPU} at $(date -u +%FT%TZ)"
            if MODEL="${model}" ROLE="${role}" GPU="${GPU}" SEED="${seed}" \
              STEP_START="${step_start}" STEP_END="${step_end}" \
              INPUT_LIST="${INPUT_LIST}" OUTPUT_ROOT="${OUTPUT_ROOT}" \
              bash "${SCRIPT_DIR}/run_common22_public_head_ablation_job.sh"; then
              echo "[st-union-worker] complete ${label} at $(date -u +%FT%TZ)"
            else
              status=$?
              failed=1
              seed_failed=1
              echo "[st-union-worker] failed ${label} status=${status} at $(date -u +%FT%TZ)" >&2
            fi
          fi
        fi
        task_index=$((task_index + 1))
        total_tasks=$((total_tasks + 1))
      done
    done
  done

  if (( seed_failed == 0 )); then
    touch "${BARRIER_ROOT}/seed-${seed_key}-gpu${GPU}.done"
  else
    touch "${BARRIER_ROOT}/seed-${seed_key}-gpu${GPU}.failed"
  fi

  echo "[st-union-worker] GPU${GPU} waiting at seed ${seed} barrier"
  while true; do
    barrier_ready=1
    barrier_failed=0
    for peer in 0 1 2 3 4 5 6; do
      if [[ -f "${BARRIER_ROOT}/seed-${seed_key}-gpu${peer}.failed" ]]; then
        barrier_failed=1
      fi
      if [[ ! -f "${BARRIER_ROOT}/seed-${seed_key}-gpu${peer}.done" ]]; then
        barrier_ready=0
      fi
    done
    if (( barrier_failed == 1 )); then
      echo "[st-union-worker] seed ${seed} barrier failed; refusing to advance" >&2
      exit 1
    fi
    if (( barrier_ready == 1 )); then
      echo "[st-union-worker] seed ${seed} complete on all GPUs at $(date -u +%FT%TZ)"
      break
    fi
    sleep 10
  done
done

echo "[st-union-worker] GPU${GPU} exhausted assigned tasks; total_matrix=${total_tasks}"
exit "${failed}"
