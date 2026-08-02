#!/usr/bin/env bash
set -euo pipefail

cd /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/xssc_physics_representation

GPU=3
MAX_USED_MIB=6000
POLL_SECONDS=60

while true; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${GPU}" | tr -d ' ')
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "${GPU}" | tr -d ' ')
  printf '[gpu-wait] gpu=%s used_mib=%s util_pct=%s\n' "${GPU}" "${used}" "${util}"
  if (( used <= MAX_USED_MIB )); then
    break
  fi
  sleep "${POLL_SECONDS}"
done

SMOKE_ROOT=/data/gaoya/agent-data/outputs/xssc_physics_representation/phase1_smoke
/home/gaoya/miniconda3/envs/wan-cu128/bin/python extract_phase1_features.py \
  --stage all \
  --physical-gpu "${GPU}" \
  --batch-size 16 \
  --case-limit 1 \
  --model-names dinov3_movic_step044000 \
  --output-dir "${SMOKE_ROOT}"

/home/gaoya/miniconda3/envs/wan-cu128/bin/python extract_phase1_features.py \
  --stage all \
  --physical-gpu "${GPU}" \
  --batch-size 16 \
  --output-dir /data/gaoya/agent-data/outputs/xssc_physics_representation/phase1

/home/gaoya/miniconda3/envs/wan-cu128/bin/python report_phase1.py \
  --root /data/gaoya/agent-data/outputs/xssc_physics_representation/phase1
