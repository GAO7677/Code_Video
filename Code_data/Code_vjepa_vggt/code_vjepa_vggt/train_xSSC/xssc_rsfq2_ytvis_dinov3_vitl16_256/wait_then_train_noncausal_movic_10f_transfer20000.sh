#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-/data/gaoya/agent-data/checkpoints/xssc_vjepa2_1_video_noncausal_ytvis_hq_10f_ar_bs64_steps20000/rsfq2_r-ytvis_hq-vjepa2_1_vitl16-ar10f-slot512-transfer10000-bs64/42}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${SOURCE_RUN_DIR}/step-020000.pth}"
SOURCE_METADATA="${SOURCE_CHECKPOINT%.pth}.metadata.json"
SOURCE_CONFIG_PATTERN="rsfq2_r-ytvis_hq-vjepa2_1_vitl16-ar10f-slot512-transfer10000-bs64.py"
POLL_SECONDS="${POLL_SECONDS:-30}"

echo "[wait-movic-10f] waiting for ${SOURCE_CHECKPOINT}"
while [[ ! -f "${SOURCE_CHECKPOINT}" || ! -f "${SOURCE_METADATA}" ]]; do
  sleep "${POLL_SECONDS}"
done

"${PYTHON_BIN}" - "${SOURCE_METADATA}" <<'PY'
import json
import sys
from pathlib import Path

metadata = json.loads(Path(sys.argv[1]).read_text())
expected_variant = (
    "vjepa2_1_vitl16_video_ytvis_hq_10f_ar_slot512_transfer10000_bs64"
)
if metadata.get("optimizer_step") != 20000:
    raise SystemExit(f"unexpected source step: {metadata}")
if metadata.get("variant_name") != expected_variant:
    raise SystemExit(f"unexpected source variant: {metadata}")
if metadata.get("world_size") != 2 or metadata.get("effective_global_batch_size") != 384:
    raise SystemExit(f"unexpected source batch topology: {metadata}")
print(f"[wait-movic-10f] verified source metadata: {metadata}", flush=True)
PY

echo "[wait-movic-10f] checkpoint is complete; waiting for YTVIS trainer exit"
while pgrep -u "$(id -u)" -f "train_ddp_ytvis_hq.py .*${SOURCE_CONFIG_PATTERN}" >/dev/null; do
  sleep "${POLL_SECONDS}"
done

echo "[wait-movic-10f] starting noncausal MOVi-C transfer at $(date -u +%FT%TZ)"
exec env \
  YTVIS_CHECKPOINT="${SOURCE_CHECKPOINT}" \
  GPU_IDS=5,6 \
  NPROC_PER_NODE=2 \
  DATA_DIR=/data/gaoya/dataset \
  WANDB_PROJECT=xssc_vjepa2_1_movi_c_10f \
  WANDB_MODE=online \
  bash "${ROOT}/run_train_rsfq2_movi_c_vjepa2_1_vitl16_256_video_10f_transfer20000.sh"
