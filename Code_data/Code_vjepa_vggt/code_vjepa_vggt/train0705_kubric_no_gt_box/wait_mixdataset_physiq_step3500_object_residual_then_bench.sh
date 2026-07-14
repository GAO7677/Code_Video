#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box
OUTPUT_BASE=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/physicIQ/train_stage1b_mixdataset/step-003500
LOG_ROOT=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/physicIQ/train_stage1b_mixdataset/_sweep_logs
EVAL_ALL="${SCRIPT_DIR}/AAAeval.txt"
EVAL_PHYSIQ="${SCRIPT_DIR}/AAAevalphysiq.txt"

for tag in 1p0 1p5 2p0; do
  marker="${OUTPUT_BASE}/object_residual_${tag}x/worker_complete.txt"
  until [[ -s "${marker}" ]]; do
    echo "[coordinator] waiting for ${marker}"
    sleep 120
  done
done

for tag in 1p0 1p5 2p0; do
  root="${OUTPUT_BASE}/object_residual_${tag}x"
  leaf="$(find "${root}" -mindepth 2 -maxdepth 2 -type f -name result.json -printf '%h\n' | sort | head -n 1)"
  if [[ -z "${leaf}" ]]; then
    echo "[coordinator] missing result leaf: ${root}" >&2
    exit 1
  fi
  flock "${EVAL_ALL}.lock" bash -c 'grep -Fqx -- "$1" "$2" 2>/dev/null || printf "%s\n" "$1" >> "$2"' _ "${leaf}" "${EVAL_ALL}"
  flock "${EVAL_PHYSIQ}.lock" bash -c 'grep -Fqx -- "$1" "$2" 2>/dev/null || printf "%s\n" "$1" >> "$2"' _ "${leaf}" "${EVAL_PHYSIQ}"
done

echo "[coordinator] all workers completed; starting PhysicIQ metrics"
CUDA_VISIBLE_DEVICES=0 BENCH_CUDA_VISIBLE_DEVICES=0 \
  bash "${SCRIPT_DIR}/bench.sh" "${EVAL_PHYSIQ}" 2>&1 \
  | tee -a "${LOG_ROOT}/bench_AAAevalphysiq_step-003500.log"
echo "[coordinator] metrics completed"
