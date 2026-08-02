#!/usr/bin/env bash
set -euo pipefail

cd /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/xssc_physics_representation

GPU=2
SMOKE_ROOT=/data/gaoya/agent-data/outputs/xssc_physics_representation/phase1_smoke
FORMAL_ROOT=/data/gaoya/agent-data/outputs/xssc_physics_representation/phase1

echo "[smoke] physical_gpu=${GPU}"
/home/gaoya/miniconda3/envs/wan-cu128/bin/python extract_phase1_features.py \
  --stage all \
  --physical-gpu "${GPU}" \
  --batch-size 16 \
  --case-limit 1 \
  --model-names dinov3_movic_step044000 \
  --output-dir "${SMOKE_ROOT}"

echo "[formal] physical_gpu=${GPU}"
/home/gaoya/miniconda3/envs/wan-cu128/bin/python extract_phase1_features.py \
  --stage all \
  --physical-gpu "${GPU}" \
  --batch-size 16 \
  --output-dir "${FORMAL_ROOT}"

echo "[report]"
/home/gaoya/miniconda3/envs/wan-cu128/bin/python report_phase1.py \
  --root "${FORMAL_ROOT}"
