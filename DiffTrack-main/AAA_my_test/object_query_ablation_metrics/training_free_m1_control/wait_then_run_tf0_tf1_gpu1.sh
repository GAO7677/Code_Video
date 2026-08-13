#!/usr/bin/env bash
set -euo pipefail

TF_GPU="${TF_GPU:-1}"
BLOCKING_SESSION="${TF_BLOCKING_SESSION:-gt_stc_hyperparam_stage1a_gpu12}"
RUNNER="/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/object_query_ablation_metrics/training_free_m1_control/run_tf0_then_tf1_gpu1.sh"
MAX_USED_MIB=1024
STABLE_POLLS_REQUIRED=3
POLL_SECONDS=30

if [[ "${TF_GPU}" == "4" ]]; then
  echo "GPU 4 is forbidden by workspace policy" >&2
  exit 2
fi

if [[ -n "${BLOCKING_SESSION}" ]]; then
  echo "[$(date -u +%FT%TZ)] GPU${TF_GPU}: waiting for ${BLOCKING_SESSION} to finish"
  while tmux has-session -t "${BLOCKING_SESSION}" 2>/dev/null; do
    sleep "${POLL_SECONDS}"
  done
fi

echo "[$(date -u +%FT%TZ)] blocker finished; waiting for stable free GPU${TF_GPU}"
stable_polls=0
while (( stable_polls < STABLE_POLLS_REQUIRED )); do
  used_mib="$({
    nvidia-smi --id="${TF_GPU}" --query-gpu=memory.used --format=csv,noheader,nounits
  } | tr -d '[:space:]')"
  if [[ "${used_mib}" =~ ^[0-9]+$ ]] && (( used_mib <= MAX_USED_MIB )); then
    stable_polls=$((stable_polls + 1))
  else
    stable_polls=0
  fi
  echo "[$(date -u +%FT%TZ)] GPU${TF_GPU} used=${used_mib} MiB stable=${stable_polls}/${STABLE_POLLS_REQUIRED}"
  if (( stable_polls < STABLE_POLLS_REQUIRED )); then
    sleep "${POLL_SECONDS}"
  fi
done

echo "[$(date -u +%FT%TZ)] GPU${TF_GPU} is stably free; starting TF-0 -> TF-1"
exec env CUDA_VISIBLE_DEVICES="${TF_GPU}" TF_GPU="${TF_GPU}" bash "${RUNNER}"
