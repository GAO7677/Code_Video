#!/usr/bin/env bash
# Run in the foreground:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/run_train_no_object_dataset_ablation_gpu67_serial.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TAG="${RUN_TAG:-serial_20260804T115337Z}"
PYBULLET_CONFIG="${SCRIPT_DIR}/configs/formal_full_sa_no_object_pybullet100_gpu67_1000steps.json"
KUBRIC_CONFIG="${SCRIPT_DIR}/configs/formal_full_sa_no_object_kubric100_gpu67_1000steps.json"

echo "[1/2] Starting 100% PyBullet on GPU6,7 with run tag ${RUN_TAG}"
bash "${SCRIPT_DIR}/run_train_from_config.sh" "${PYBULLET_CONFIG}" --run-tag "${RUN_TAG}"

echo "[2/2] PyBullet completed successfully; starting 100% Kubric on GPU6,7"
bash "${SCRIPT_DIR}/run_train_from_config.sh" "${KUBRIC_CONFIG}" --run-tag "${RUN_TAG}"

echo "Both dataset ablation runs completed successfully."
