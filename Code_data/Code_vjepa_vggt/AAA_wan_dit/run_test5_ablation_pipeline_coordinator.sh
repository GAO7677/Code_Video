#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 5 ]]; then
  echo "Usage: $0 RUN_ROOT OUTPUT_BASE INPUT_LIST NUM_GEN_WORKERS NUM_METRIC_WORKERS" >&2
  exit 2
fi

RUN_ROOT="$1"
OUTPUT_BASE="$2"
INPUT_LIST="$3"
NUM_GEN_WORKERS="$4"
NUM_METRIC_WORKERS="$5"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
SUMMARY=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/summarize_benchmark_txt_metrics.py
VERIFY_METRICS=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/verify_bench_physiq_metrics.py

while true; do
  gen_workers="$(find "${RUN_ROOT}/generation/state" -name '*.complete' -type f | wc -l)"
  gen_done="$(wc -l < "${RUN_ROOT}/generation/completed.tsv")"
  gen_failed="$(wc -l < "${RUN_ROOT}/generation/failed.tsv")"
  printf '[coordinator] generation workers=%s/%s jobs=%s/63 failed=%s\n' \
    "${gen_workers}" "${NUM_GEN_WORKERS}" "${gen_done}" "${gen_failed}"
  if [[ "${gen_workers}" -eq "${NUM_GEN_WORKERS}" ]]; then
    break
  fi
  sleep 30
done

if [[ "$(wc -l < "${RUN_ROOT}/generation/failed.tsv")" -ne 0 ]] || \
   [[ "$(wc -l < "${RUN_ROOT}/generation/completed.tsv")" -ne 63 ]]; then
  touch "${RUN_ROOT}/generation.failed"
  echo "[coordinator] generation validation failed"
  exit 1
fi

"${PYTHON}" - "${RUN_ROOT}" <<'PY'
import json
import pathlib
import sys

run_root = pathlib.Path(sys.argv[1])
queue = []
for line in (run_root / "generation/queue.tsv").read_text().splitlines():
    if line.strip():
        task_id, model, mode, block = line.split("\t")
        queue.append((task_id, model, mode, block))
if len(queue) != 63:
    raise SystemExit(f"expected 63 jobs, got {len(queue)}")

roots = []
manifest = []
for task_id, model, mode, block in queue:
    path = run_root / "generation/validations" / f"{task_id}.json"
    payload = json.loads(path.read_text())
    roots.append(payload["result_root"])
    manifest.append(payload)
if len(set(roots)) != 63:
    raise SystemExit("metric result roots are not unique")
(run_root / "leaf_folders.txt").write_text("\n".join(roots) + "\n")
(run_root / "generation_manifest.json").write_text(
    json.dumps({"num_configs": 63, "num_cases_per_config": 5, "configs": manifest}, indent=2)
    + "\n"
)
PY

METRIC_ROOT="${RUN_ROOT}/metrics"
mkdir -p "${METRIC_ROOT}/queues" "${METRIC_ROOT}/logs" \
  "${METRIC_ROOT}/state" "${METRIC_ROOT}/task_summaries"
: > "${METRIC_ROOT}/completed_tasks.tsv"
: > "${METRIC_ROOT}/failed_tasks.tsv"

CPU_METRICS=(
  physics_iq_with_context physics_iq_without_context
  pmf_with_context pmf_without_context
)
GPU_COMMON_METRICS=(
  wmreward
  vbench_subject_consistency vbench_background_consistency
  vbench_temporal_flickering vbench_motion_smoothness
  vbench_dynamic_degree vbench_aesthetic_quality vbench_imaging_quality
)

for kind in cpu gpu_common videophy2 cosmos; do
  : > "${METRIC_ROOT}/queues/${kind}.tsv"
  printf '1\n' > "${METRIC_ROOT}/queues/${kind}.cursor"
done

task_index=0
while IFS= read -r root; do
  for metric in "${CPU_METRICS[@]}"; do
    printf 'cpu-%04d\t%s\t%s\n' "${task_index}" "${metric}" "${root}" \
      >> "${METRIC_ROOT}/queues/cpu.tsv"
    task_index=$((task_index + 1))
  done
done < "${RUN_ROOT}/leaf_folders.txt"

task_index=0
while IFS= read -r root; do
  for metric in "${GPU_COMMON_METRICS[@]}"; do
    printf 'gpu-%04d\t%s\t%s\n' "${task_index}" "${metric}" "${root}" \
      >> "${METRIC_ROOT}/queues/gpu_common.tsv"
    task_index=$((task_index + 1))
  done
done < "${RUN_ROOT}/leaf_folders.txt"

task_index=0
while IFS= read -r root; do
  printf 'videophy2-%04d\tvideophy2\t%s\n' "${task_index}" "${root}" \
    >> "${METRIC_ROOT}/queues/videophy2.tsv"
  printf 'cosmos-%04d\tcosmos_reason1\t%s\n' "${task_index}" "${root}" \
    >> "${METRIC_ROOT}/queues/cosmos.tsv"
  task_index=$((task_index + 1))
done < "${RUN_ROOT}/leaf_folders.txt"

touch "${RUN_ROOT}/metrics.ready"
echo "[coordinator] generation complete; metric queues released"

while true; do
  metric_workers="$(find "${METRIC_ROOT}/state" -name '*.complete' -type f | wc -l)"
  metric_done="$(wc -l < "${METRIC_ROOT}/completed_tasks.tsv")"
  metric_failed="$(wc -l < "${METRIC_ROOT}/failed_tasks.tsv")"
  printf '[coordinator] metrics workers=%s/%s tasks=%s/882 failed=%s\n' \
    "${metric_workers}" "${NUM_METRIC_WORKERS}" "${metric_done}" "${metric_failed}"
  if [[ "${metric_workers}" -eq "${NUM_METRIC_WORKERS}" ]]; then
    break
  fi
  sleep 30
done

"${PYTHON}" "${SUMMARY}" \
  --input-txt "${RUN_ROOT}/leaf_folders.txt" \
  --output-csv "${METRIC_ROOT}/metric_summary.csv" \
  --input-json-allowlist "${INPUT_LIST}"
"${PYTHON}" "${VERIFY_METRICS}" \
  --baseline-list "${RUN_ROOT}/leaf_folders.txt" \
  --output "${METRIC_ROOT}/verification.json" \
  --input-json-allowlist "${INPUT_LIST}"

if [[ "$(wc -l < "${METRIC_ROOT}/failed_tasks.tsv")" -ne 0 ]]; then
  touch "${RUN_ROOT}/metrics.failed"
  exit 1
fi
touch "${RUN_ROOT}/pipeline.complete"
echo "[coordinator] pipeline complete"
