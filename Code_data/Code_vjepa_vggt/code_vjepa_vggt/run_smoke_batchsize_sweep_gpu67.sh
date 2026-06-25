#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
ACCELERATE_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/accelerate
TRAIN_SCRIPT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_context_video_wan.py
BASE_CONFIG=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_wan_lora_monitor_gpu67_pilot.yaml
OUTPUT_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/smoke_batchsize_sweep_gpu67
TMP_ROOT=/dev/shm/pybullet0624_batchsize_sweep_gpu67
WANDB_DIR=/data/gaoya/AAA_test_video/0623/train/train0624/logs/wandb
DATASET_ROOT=/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/episodes_v1/industrial_s1_scale2_256x144_s8_f16_n6_h264_batch1500
WAN_ROOT=/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B
JEPA_CKPT=/data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384
VGGT_MODEL=/data/gaoya/ckpt/facebook-VGGT-1B
COTRACKER_CKPT=/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth

mkdir -p "${OUTPUT_ROOT}" "${TMP_ROOT}" "${WANDB_DIR}"

BATCH_SIZES=(1 2 3 4)
SUMMARY_PATH="${OUTPUT_ROOT}/batchsize_sweep_summary.json"

${PYTHON_BIN} - <<'PY' > "${SUMMARY_PATH}"
import json
from pathlib import Path
print(json.dumps({"runs": []}, indent=2))
PY

run_one() {
  local batch_size="$1"
  local config_path="${TMP_ROOT}/config_bs${batch_size}.yaml"
  local output_dir="${OUTPUT_ROOT}/bs${batch_size}"
  local wandb_name="pybullet0624_wan_lora_monitor_gpu67_bs${batch_size}_smoke"

  mkdir -p "${output_dir}"
  ${PYTHON_BIN} - <<PY
from pathlib import Path
import yaml

base = Path("${BASE_CONFIG}")
cfg = yaml.safe_load(base.read_text(encoding="utf-8"))
cfg["data"]["batch_size"] = int(${batch_size})
cfg["optimization"]["max_steps"] = 2
cfg["logging"]["save_every"] = 2
cfg["logging"]["wandb_run_name"] = "${wandb_name}"
cfg["experiment"]["name"] = "${wandb_name}"
cfg["experiment"]["output_dir"] = "${output_dir}"
Path("${config_path}").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
PY

  echo "[batch_size=${batch_size}] launching smoke..."
  set +e
  PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main \
  CUDA_VISIBLE_DEVICES=6,7 "${ACCELERATE_BIN}" launch --multi_gpu --num_processes 2 --num_machines 1 "${TRAIN_SCRIPT}" \
    --config "${config_path}"
  local exit_code=$?
  set -e

  ${PYTHON_BIN} - <<PY
import json
from pathlib import Path

summary_path = Path("${SUMMARY_PATH}")
payload = json.loads(summary_path.read_text(encoding="utf-8"))
payload["runs"].append({
    "batch_size": int(${batch_size}),
    "config_path": "${config_path}",
    "output_dir": "${output_dir}",
    "wandb_name": "${wandb_name}",
    "exit_code": int(${exit_code}),
})
summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

  if [[ "${exit_code}" -ne 0 ]]; then
    echo "[batch_size=${batch_size}] exited with code ${exit_code}"
  else
    echo "[batch_size=${batch_size}] completed successfully"
  fi
}

for bs in "${BATCH_SIZES[@]}"; do
  run_one "${bs}"
done

echo "summary written to ${SUMMARY_PATH}"
