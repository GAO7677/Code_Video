#!/usr/bin/env bash
set -euo pipefail
# CUDA_VISIBLE_DEVICES=5 bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/bench.sh /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/AAAevalphysiq.txt



# CUDA_VISIBLE_DEVICES=5 bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/bench.sh /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/AAAeval.txt



# /home/gaoya/miniconda3/envs/wan-cu128/bin/python /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/summarize_generated_folder_metrics.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRY0526_ROOT="/home/gaoya/Code_Video/Code_data/Code_try0526"

PYTHON_BIN="${PYTHON_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
BENCH_PY="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_phys_papers_compare/bench.py"
BASELINE_LIST="${1:-${SCRIPT_DIR}/baseline.txt}"
SUMMARY_PY="${SCRIPT_DIR}/summarize_benchmark_txt_metrics.py"
RESULT_DIR="${SCRIPT_DIR}/AAAresults"
BENCH_CUDA_VISIBLE_DEVICES="${BENCH_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
BENCH_METRICS_RAW="${BENCH_METRICS:-}"

export PYTHONPATH="${PROJECT_ROOT}:${TRY0526_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${BENCH_CUDA_VISIBLE_DEVICES}" ]]; then
  export CUDA_VISIBLE_DEVICES="${BENCH_CUDA_VISIBLE_DEVICES}"
fi

METRICS=(
  # "physics_iq_with_context"
  # "physics_iq_without_context"
  # "pmf_with_context"
  # "pmf_without_context"
  "wmreward"
  "videophy2"
  "cosmos_reason1"
)

if [[ -n "${BENCH_METRICS_RAW}" ]]; then
  IFS=',' read -r -a METRICS <<< "${BENCH_METRICS_RAW}"
fi

echo "[baseline-bench] python=${PYTHON_BIN}"
echo "[baseline-bench] baseline_list=${BASELINE_LIST}"
echo "[baseline-bench] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[baseline-bench] metrics=${METRICS[*]}"
mkdir -p "${RESULT_DIR}"

for metric in "${METRICS[@]}"; do
  echo "[baseline-bench] start metric=${metric}"
  "${PYTHON_BIN}" "${BENCH_PY}" \
    --metric "${metric}" \
    --baseline-list "${BASELINE_LIST}"
  echo "[baseline-bench] done metric=${metric}"
done

SUMMARY_BASENAME="$(basename "${BASELINE_LIST}")"
SUMMARY_STEM="${SUMMARY_BASENAME%.*}"
OUTPUT_CSV="${RESULT_DIR}/${SUMMARY_STEM}_metric_summary.csv"
echo "[baseline-bench] start export_csv=${OUTPUT_CSV}"
"${PYTHON_BIN}" "${SUMMARY_PY}" \
  --input-txt "${BASELINE_LIST}" \
  --output-csv "${OUTPUT_CSV}"
echo "[baseline-bench] done export_csv=${OUTPUT_CSV}"

echo "[baseline-bench] all metrics completed"
