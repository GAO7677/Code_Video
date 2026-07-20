#!/usr/bin/env bash
set -euo pipefail
# 统计结果
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/bench.sh /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/AAAevalphysiq.txt


# 计算指标+统计结果
# BENCH_RUN_METRICS=1 CUDA_VISIBLE_DEVICES=1 bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/bench.sh /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/AAAevalphysiq1.txt
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
BENCH_RUN_METRICS="${BENCH_RUN_METRICS:-0}"
BENCH_INPUT_JSON_ALLOWLIST="${BENCH_INPUT_JSON_ALLOWLIST:-}"
DEFAULT_PHYSIQ_INPUT_JSON_ALLOWLIST=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt

if [[ -z "${BENCH_INPUT_JSON_ALLOWLIST}" && "$(basename "${BASELINE_LIST}")" == "AAAevalphysiq.txt" ]]; then
  BENCH_INPUT_JSON_ALLOWLIST="${DEFAULT_PHYSIQ_INPUT_JSON_ALLOWLIST}"
fi

export PYTHONPATH="${PROJECT_ROOT}:${TRY0526_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${BENCH_CUDA_VISIBLE_DEVICES}" ]]; then
  export CUDA_VISIBLE_DEVICES="${BENCH_CUDA_VISIBLE_DEVICES}"
fi

METRICS=(
  "physics_iq_with_context"
  "physics_iq_without_context"
  "pmf_with_context"
  "pmf_without_context"
  "wmreward"
  "videophy2"
  "cosmos_reason1"
  "vbench_subject_consistency"
  "vbench_background_consistency"
  "vbench_temporal_flickering"
  "vbench_motion_smoothness"
  "vbench_dynamic_degree"
  "vbench_aesthetic_quality"
  "vbench_imaging_quality"
)

if [[ -n "${BENCH_METRICS_RAW}" ]]; then
  IFS=',' read -r -a METRICS <<< "${BENCH_METRICS_RAW}"
fi

echo "[baseline-bench] python=${PYTHON_BIN}"
echo "[baseline-bench] baseline_list=${BASELINE_LIST}"
echo "[baseline-bench] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[baseline-bench] metrics=${METRICS[*]}"
echo "[baseline-bench] run_metrics=${BENCH_RUN_METRICS}"
echo "[baseline-bench] input_json_allowlist=${BENCH_INPUT_JSON_ALLOWLIST:-<unset>}"
mkdir -p "${RESULT_DIR}"

SUMMARY_BASENAME="$(basename "${BASELINE_LIST}")"
SUMMARY_STEM="${SUMMARY_BASENAME%.*}"
OUTPUT_CSV="${RESULT_DIR}/${SUMMARY_STEM}_metric_summary.csv"

export_summary() {
  local -a summary_args
  echo "[baseline-bench] start export_csv=${OUTPUT_CSV}"
  summary_args=(
    --input-txt "${BASELINE_LIST}"
    --output-csv "${OUTPUT_CSV}"
  )
  if [[ -n "${BENCH_INPUT_JSON_ALLOWLIST}" ]]; then
    summary_args+=(--input-json-allowlist "${BENCH_INPUT_JSON_ALLOWLIST}")
  fi
  "${PYTHON_BIN}" "${SUMMARY_PY}" "${summary_args[@]}"
  echo "[baseline-bench] done export_csv=${OUTPUT_CSV}"
}

if [[ "${BENCH_RUN_METRICS}" != "1" ]]; then
  echo "[baseline-bench] summary-only mode; skip metric checks/runs"
  export_summary
  echo "[baseline-bench] summary completed"
  exit 0
fi

# Keep the CSV synchronized with all metrics that reached disk, including
# partial runs stopped by a failed metric or an interrupt.
trap export_summary EXIT

for metric in "${METRICS[@]}"; do
  echo "[baseline-bench] start metric=${metric}"
  EXTRA_ARGS=()
  if [[ "${metric}" == "wmreward" ]]; then
    EXTRA_ARGS+=(--wmreward-reset-interval 1000000)
  fi
  if [[ -n "${BENCH_INPUT_JSON_ALLOWLIST}" ]]; then
    EXTRA_ARGS+=(--input-json-allowlist "${BENCH_INPUT_JSON_ALLOWLIST}")
  fi
  "${PYTHON_BIN}" "${BENCH_PY}" \
    --metric "${metric}" \
    --baseline-list "${BASELINE_LIST}" \
    "${EXTRA_ARGS[@]}"
  echo "[baseline-bench] done metric=${metric}"
done

export_summary
trap - EXIT

echo "[baseline-bench] all metrics completed"
