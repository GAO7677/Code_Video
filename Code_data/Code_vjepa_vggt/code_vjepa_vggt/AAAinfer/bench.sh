#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TRY0526_ROOT="/home/gaoya/Code_Video/Code_data/Code_try0526"

PYTHON_BIN="${PYTHON_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
BENCH_PY="${SCRIPT_DIR}/bench.py"
REPORT_PY="${SCRIPT_DIR}/render_v2v_metric_report.py"

RESULT_ROOT="${1:-/data/gaoya/AAA_test_video/0623/test/v2v}"
BENCH_CUDA_VISIBLE_DEVICES="${BENCH_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
BENCH_METRICS_RAW="${BENCH_METRICS:-}"

export PYTHONPATH="${PROJECT_ROOT}:${TRY0526_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${BENCH_CUDA_VISIBLE_DEVICES}" ]]; then
  export CUDA_VISIBLE_DEVICES="${BENCH_CUDA_VISIBLE_DEVICES}"
fi

METRICS=(
  "wmreward"
  "physics_iq"
  "physics_iq_with_context"
  "physics_iq_without_context"
  "pmf_with_context"
  "pmf_without_context"
  "videophy2"
  # "phyground"
  "cosmos_reason1"
)

if [[ -n "${BENCH_METRICS_RAW}" ]]; then
  IFS=',' read -r -a METRICS <<< "${BENCH_METRICS_RAW}"
fi

inspect_metric_summary() {
  local metric="$1"
  "${PYTHON_BIN}" - "${RESULT_ROOT}" "${metric}" <<'PY'
import json
import sys
from pathlib import Path

result_root = Path(sys.argv[1]).expanduser().resolve()
metric = sys.argv[2]
summary_path = result_root / f"eval_summary_{metric}.json"

status = "missing_summary"
num_cases = 0
num_success = 0
num_failed = 0
completed = 0
errors_count = 0

if summary_path.is_file():
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        metric_status = payload.get("metric_status") or {}
        num_cases = int(metric_status.get("num_cases") or 0)
        num_success = int(metric_status.get("num_success") or 0)
        num_failed = int(metric_status.get("num_failed") or 0)
        completed = int(metric_status.get("completed") or 0)
        errors = payload.get("errors")
        errors_count = len(errors) if isinstance(errors, list) else 0
        if num_cases <= 0:
            status = "empty"
        elif num_failed > 0 or errors_count > 0:
            status = "failed"
        elif num_success != num_cases or completed != num_cases:
            status = "partial"
        else:
            status = "ok"
    except Exception:
        status = "invalid_summary"

print(f"status={status}")
print(f"summary_path={summary_path}")
print(f"num_cases={num_cases}")
print(f"num_success={num_success}")
print(f"num_failed={num_failed}")
print(f"completed={completed}")
print(f"errors_count={errors_count}")
PY
}

run_metric() {
  local metric="$1"
  local summary_kv
  local status=""
  local summary_path=""
  local num_cases=""
  local num_success=""
  local num_failed=""
  local completed=""
  local errors_count=""

  echo "[bench] start metric=${metric}"
  "${PYTHON_BIN}" "${BENCH_PY}" \
    --metric "${metric}" \
    --result-root "${RESULT_ROOT}"
  echo "[bench] done metric=${metric}"

  summary_kv="$(inspect_metric_summary "${metric}")"
  while IFS='=' read -r key value; do
    case "${key}" in
      status) status="${value}" ;;
      summary_path) summary_path="${value}" ;;
      num_cases) num_cases="${value}" ;;
      num_success) num_success="${value}" ;;
      num_failed) num_failed="${value}" ;;
      completed) completed="${value}" ;;
      errors_count) errors_count="${value}" ;;
    esac
  done <<< "${summary_kv}"

  echo "[bench] metric_signal metric=${metric} status=${status} cases=${num_cases} success=${num_success} failed=${num_failed} completed=${completed} errors=${errors_count}"
  echo "[bench] metric_summary_path=${summary_path}"

  case "${status}" in
    ok)
      return 0
      ;;
    empty)
      return 2
      ;;
    failed|partial|missing_summary|invalid_summary)
      return 1
      ;;
    *)
      return 1
      ;;
  esac
}

print_folder_metric_progress() {
  "${PYTHON_BIN}" - "${RESULT_ROOT}" "${METRICS[@]}" <<'PY'
import json
import sys
from pathlib import Path

result_root = Path(sys.argv[1]).expanduser().resolve()
metrics = list(sys.argv[2:])
excluded_names = {"summary.json", "result.json", "batch_manifest.json", "eval_summary.json"}

folder_rows: dict[Path, dict[str, object]] = {}
for json_path in sorted(result_root.rglob("*.json")):
    if json_path.name in excluded_names or json_path.name.startswith("eval_summary_"):
        continue
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if not isinstance(payload, dict):
        continue
    if "input_json" not in payload and "case_json" not in payload:
        continue
    folder = json_path.parent.resolve()
    row = folder_rows.setdefault(
        folder,
        {
            "num_cases": 0,
            "metric_counts": {metric: 0 for metric in metrics},
        },
    )
    row["num_cases"] = int(row["num_cases"]) + 1
    metric_counts = row["metric_counts"]
    for metric in metrics:
        if payload.get(metric) is not None:
            metric_counts[metric] += 1

print("[bench] per-folder metric progress")
for folder in sorted(folder_rows):
    row = folder_rows[folder]
    num_cases = int(row["num_cases"])
    metric_counts = row["metric_counts"]
    parts = [f"{metric}={metric_counts[metric]}/{num_cases}" for metric in metrics]
    print(f"[bench] folder={folder}")
    print(f"[bench]   {' | '.join(parts)}")
PY
}

echo "[bench] python=${PYTHON_BIN}"
echo "[bench] result_root=${RESULT_ROOT}"
echo "[bench] cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[bench] metrics=${METRICS[*]}"
echo "[bench] input_json_policy=read absolute input_json directly from each result json"
echo "[bench] skip_policy=existing metric fields are preserved unless --overwrite is used in bench.py"

overall_exit_code=0
overall_signal="success"
failed_metric=""

for metric in "${METRICS[@]}"; do
  set +e
  run_metric "${metric}"
  metric_exit_code=$?
  set -e
  if [[ "${metric_exit_code}" -ne 0 ]]; then
    overall_exit_code="${metric_exit_code}"
    failed_metric="${metric}"
    case "${overall_exit_code}" in
      2) overall_signal="empty" ;;
      *) overall_signal="failed" ;;
    esac
    break
  fi
done

if [[ "${overall_exit_code}" -eq 0 ]]; then
  echo "[bench] render report"
  "${PYTHON_BIN}" "${REPORT_PY}" --result-root "${RESULT_ROOT}"
else
  echo "[bench] skip render report due to final_signal=${overall_signal} metric=${failed_metric}"
fi

print_folder_metric_progress

if [[ "${overall_exit_code}" -eq 0 ]]; then
  echo "[bench] final_signal=success result_root=${RESULT_ROOT}"
  echo "[bench] all metrics completed successfully"
else
  echo "[bench] final_signal=${overall_signal} result_root=${RESULT_ROOT} metric=${failed_metric}"
  exit "${overall_exit_code}"
fi
