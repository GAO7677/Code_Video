#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-/data/gaoya/agent-data/checkpoints/xssc_vjepa2_1_video_noncausal_ytvis_hq_bs64_steps10000/rsfq2_r-ytvis_hq-vjepa2_1_vitl16_256-video-slot512/42}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${SOURCE_RUN_DIR}/step-010000.pth}"
SOURCE_METADATA="${SOURCE_CHECKPOINT%.pth}.metadata.json"
SOURCE_CONFIG_PATTERN="rsfq2_r-ytvis_hq-vjepa2_1_vitl16_256-video-slot512.py"
PAIR_WRAPPER_PID="${PAIR_WRAPPER_PID:-}"
POLL_SECONDS="${POLL_SECONDS:-30}"

echo "[wait-movic] waiting for ${SOURCE_CHECKPOINT}"
while [[ ! -f "${SOURCE_CHECKPOINT}" || ! -f "${SOURCE_METADATA}" ]]; do
  sleep "${POLL_SECONDS}"
done

"${PYTHON_BIN}" - "${SOURCE_METADATA}" <<'PY'
import json
import sys
from pathlib import Path

metadata = json.loads(Path(sys.argv[1]).read_text())
expected_variant = "vjepa2_1_vitl16_video_256_ytvis_hq_slot512_native_tubelet"
if metadata.get("optimizer_step") != 10000:
    raise SystemExit(f"unexpected source step: {metadata}")
if metadata.get("variant_name") != expected_variant:
    raise SystemExit(f"unexpected source variant: {metadata}")
print(f"[wait-movic] verified source metadata: {metadata}", flush=True)
PY

echo "[wait-movic] checkpoint exists; waiting for the noncausal trainer to exit"
while pgrep -u "$(id -u)" -f "train_ddp_ytvis_hq.py .*${SOURCE_CONFIG_PATTERN}" >/dev/null; do
  sleep "${POLL_SECONDS}"
done

# The original pair wrapper would start causal YTVIS next. It is deliberately
# stopped by the scheduler while the noncausal child finishes; retire it now.
if [[ -n "${PAIR_WRAPPER_PID}" ]] && kill -0 "${PAIR_WRAPPER_PID}" 2>/dev/null; then
  kill -TERM "${PAIR_WRAPPER_PID}"
  # SIGTERM stays pending for a stopped process until it is continued.
  kill -CONT "${PAIR_WRAPPER_PID}"
  for _ in {1..10}; do
    if ! kill -0 "${PAIR_WRAPPER_PID}" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "${PAIR_WRAPPER_PID}" 2>/dev/null; then
    kill -KILL "${PAIR_WRAPPER_PID}"
  fi
  echo "[wait-movic] retired original pair wrapper pid=${PAIR_WRAPPER_PID}"
fi

echo "[wait-movic] refreshing exact-step DINOv3/V-JEPA comparison"
if ! PYTHONPATH="${ROOT}/upstream" "${PYTHON_BIN}" \
  "${ROOT}/compare_xssc_dinov3_vjepa_same_step_wandb.py"; then
  echo "WARNING: W&B comparison refresh failed; MOVi-C training will continue" >&2
fi

echo "[wait-movic] starting noncausal MOVi-C transfer at $(date -u +%FT%TZ)"
exec env \
  YTVIS_CHECKPOINT="${SOURCE_CHECKPOINT}" \
  GPU_IDS=5,6 \
  NPROC_PER_NODE=2 \
  DATA_DIR=/data/gaoya/dataset \
  WANDB_PROJECT=xssc_vjepa2_1_movi_c \
  WANDB_MODE=online \
  bash "${ROOT}/run_train_rsfq2_movi_c_vjepa2_1_vitl16_256_video_transfer10000.sh"
