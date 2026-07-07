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

run_metric() {
  local metric="$1"
  echo "[bench] start metric=${metric}"
  "${PYTHON_BIN}" "${BENCH_PY}" \
    --metric "${metric}" \
    --result-root "${RESULT_ROOT}"
  echo "[bench] done metric=${metric}"
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
    if "input_json" not in payload:
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
echo "[bench] input_json_policy=read absolute input_json directly from each result json"
echo "[bench] skip_policy=existing metric fields are preserved unless --overwrite is used in bench.py"

for metric in "${METRICS[@]}"; do
  run_metric "${metric}"
done

echo "[bench] render report"
"${PYTHON_BIN}" "${REPORT_PY}" --result-root "${RESULT_ROOT}"

print_folder_metric_progress

echo "[bench] all metrics completed successfully"
