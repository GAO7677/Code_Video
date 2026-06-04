#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/gaoya/Code_Video/Code_data/Code_try0526"
INPUT_ROOT="/data/gaoya/AAA_test_video/Output_try0526/phygenbench"
LOG_ROOT="${INPUT_ROOT}/logs"
FLUX_PY="/home/gaoya/miniconda3/envs/flux/bin/python"
VPHY_PY="/data/gaoya/miniconda3/envs/vphy/bin/python"
METHODS=(wan22-5B-TI2V VACE_1p3B_TI2V)

mkdir -p "${LOG_ROOT}"

wait_for_no_match() {
  local pattern="$1"
  while pgrep -f "${pattern}" >/dev/null 2>&1; do
    sleep 30
  done
}

render_report() {
  python3 - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, "/home/gaoya/Code_Video/Code_data/Code_try0526")
from physv_eval.report import build_html, ABC_REPORT_ROOT, DATA_ROOT, A_OUTPUT, PHYSICSIQ_OUTPUT, PHYGENBENCH_OUTPUT

ABC_REPORT_ROOT.mkdir(parents=True, exist_ok=True)
(ABC_REPORT_ROOT / "index.html").write_text(build_html(), encoding="utf-8")
for name, target in [
    ("dataset_videos", DATA_ROOT / "videos"),
    ("pdi_output", A_OUTPUT),
    ("physicsiq_output", PHYSICSIQ_OUTPUT),
    ("phygenbench_output", PHYGENBENCH_OUTPUT),
]:
    link = ABC_REPORT_ROOT / name
    if not link.exists():
        link.symlink_to(target)
PY
}

run_flux_phase() {
  local metric_name="$1"
  local shards="$2"
  shift 2
  local gpu_ids=("$@")
  local pids=()
  local i=0
  for gpu in "${gpu_ids[@]}"; do
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
      "${FLUX_PY}" "${ROOT}/eval_benchmark_dir_metrics.py" \
      --input-root "${INPUT_ROOT}" \
      --methods "${METHODS[@]}" \
      --metrics "${metric_name}" \
      --num-shards "${shards}" \
      --shard-id "${i}" \
      --continue-on-error \
      --skip-summary \
      > "${LOG_ROOT}/phase_${metric_name}_shard${i}_gpu${gpu}.log" 2>&1 &
    pids+=($!)
    i=$((i + 1))
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
}

run_videophy_phase() {
  local shards="$1"
  shift
  local gpu_ids=("$@")
  local pids=()
  local i=0
  for gpu in "${gpu_ids[@]}"; do
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
      "${VPHY_PY}" "${ROOT}/eval_benchmark_dir_metrics.py" \
      --input-root "${INPUT_ROOT}" \
      --methods "${METHODS[@]}" \
      --metrics videophy2 \
      --num-shards "${shards}" \
      --shard-id "${i}" \
      --continue-on-error \
      --skip-summary \
      > "${LOG_ROOT}/phase_videophy2_shard${i}_gpu${gpu}.log" 2>&1 &
    pids+=($!)
    i=$((i + 1))
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
}

wait_for_no_match "eval_benchmark_dir_metrics.py --input-root ${INPUT_ROOT} --methods wan22-5B-TI2V VACE_1p3B_TI2V --metrics pdi wmreward"

# Retry the slowest metrics once in continue-on-error mode after the initial
# shard swarm finishes. Already-computed samples are skipped, so this mainly
# picks up stragglers from transient CUDA/SAM failures.
run_flux_phase "pdi" 8 0 1 2 3 4 5 6 7
run_flux_phase "wmreward" 4 0 1 2 3

run_flux_phase proxy 4 0 1 2 3
run_flux_phase cosmos 2 4 5
run_videophy_phase 2 6 7

"${FLUX_PY}" "${ROOT}/eval_benchmark_dir_metrics.py" \
  --input-root "${INPUT_ROOT}" \
  --methods "${METHODS[@]}" \
  --summary-only \
  --summary-csv "${INPUT_ROOT}/result/method_metrics_summary.csv"

render_report
