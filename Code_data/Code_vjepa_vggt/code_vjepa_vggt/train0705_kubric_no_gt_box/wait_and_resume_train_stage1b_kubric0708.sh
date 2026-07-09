#!/usr/bin/env bash
set -euo pipefail

TARGET_GPUS=(1 2 3 5 6 7)
RESUME_STATE="/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-003500/training_state.pt"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[wait] waiting for physical GPUs ${TARGET_GPUS[*]} to become idle at $(date -u '+%F %T UTC')"

while true; do
  mapfile -t BUS_IDS < <(
    nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader \
      | awk -F', ' '
          $1==1 || $1==2 || $1==3 || $1==5 || $1==6 || $1==7 { print $2 }
        '
  )

  if [ "${#BUS_IDS[@]}" -eq 0 ]; then
    echo "[wait] failed to query target GPU bus ids; retrying in 30s"
    sleep 30
    continue
  fi

  ACTIVE_LINES=()
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    for bus_id in "${BUS_IDS[@]}"; do
      if [[ "$line" == "$bus_id"* ]]; then
        ACTIVE_LINES+=("$line")
        break
      fi
    done
  done < <(nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true)

  if [ "${#ACTIVE_LINES[@]}" -eq 0 ]; then
    echo "[wait] target GPUs are idle at $(date -u '+%F %T UTC'), starting resume"
    break
  fi

  echo "[wait] still busy at $(date -u '+%F %T UTC')"
  printf '%s\n' "${ACTIVE_LINES[@]}"
  sleep 60
done

cd "${SCRIPT_DIR}"
WANDB_RUN_ID=467j6zus \
WANDB_RESUME=must \
WANDB_DIR=/home/gaoya/wandb \
VISIBLE_GPU_IDS=1,2,3,5,6,7 \
RESUME="${RESUME_STATE}" \
bash run_train_stage1b_context_only_no_gt_box_v_newtrain_kubric.sh
