#!/usr/bin/env bash
# Run on SSH 118 in the foreground:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/run_train_no_object_dataset_ablation_ssh118_gpu67_serial.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHON_BIN="/mnt/data/gaoya/agent-data/envs/wan-cu128/bin/python"
export PYTHONPATH="/home/gaoya/Code_Video/vjepa2-main:${PYTHONPATH:-}"
PYBULLET_CONFIG="${SCRIPT_DIR}/configs/ssh118_full_sa_no_object_pybullet100_gpu67_resume500.json"
KUBRIC_CONFIG="${SCRIPT_DIR}/configs/ssh118_full_sa_no_object_kubric100_gpu67_1000steps.json"
KUBRIC_READY="/mnt/data/gaoya/dataset/nnsriram97-phyco_kubric/.ssh118_rsync_complete"

echo "[1/2] Resuming 100% PyBullet from step 500 on GPU6,7"
bash "${SCRIPT_DIR}/run_train_from_config.sh" \
  "${PYBULLET_CONFIG}" --run-tag "ssh118_resume500_retry4_20260804"

echo "PyBullet completed. Waiting for the Kubric transfer marker."
while [[ ! -f "${KUBRIC_READY}" ]]; do
  sleep 60
done

echo "[2/2] Starting 100% Kubric on GPU6,7"
bash "${SCRIPT_DIR}/run_train_from_config.sh" \
  "${KUBRIC_CONFIG}" --run-tag "ssh118_after_pybullet_20260804"
