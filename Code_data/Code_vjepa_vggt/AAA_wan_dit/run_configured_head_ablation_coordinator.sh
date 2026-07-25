#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "Usage: $0 CONFIG NUM_JOBS NUM_GEN_WORKERS NUM_METRIC_WORKERS" >&2
  exit 2
fi

CONFIG="$(realpath "$1")"
NUM_JOBS="$2"
NUM_GEN_WORKERS="$3"
NUM_METRIC_WORKERS="$4"
# shellcheck source=/dev/null
source "${CONFIG}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
SUMMARY=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/summarize_benchmark_txt_metrics.py
VERIFY_METRICS=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/verify_bench_physiq_metrics.py
INPUT_LIST="${RUN_ROOT}/input_unique.txt"
GEN_STATE="${RUN_ROOT}/generation/task_state"

while true; do
  complete="$(find "${GEN_STATE}" -maxdepth 1 -name '*.complete' -type f | wc -l)"
  failed="$(find "${GEN_STATE}" -maxdepth 1 -name '*.failed' -type f | wc -l)"
  workers="$(find "${RUN_ROOT}/generation/state" -name '*.worker.complete' -type f | wc -l)"
  printf '[coordinator] generation configs=%s/%s failed=%s workers=%s/%s\n' \
    "${complete}" "${NUM_JOBS}" "${failed}" "${workers}" "${NUM_GEN_WORKERS}"
  if [[ "${complete}" -eq "${NUM_JOBS}" ]]; then
    break
  fi
  if [[ "${workers}" -eq "${NUM_GEN_WORKERS}" ]]; then
    touch "${RUN_ROOT}/generation.failed"
    echo "[coordinator] workers ended before all configurations completed"
    exit 1
  fi
  sleep "${COORDINATOR_POLL_SECONDS}"
done

"${PYTHON}" - "${RUN_ROOT}" "${NUM_JOBS}" "${EXPECTED_CASES}" <<'PY'
import json
import pathlib
import sys

run_root = pathlib.Path(sys.argv[1])
expected_jobs = int(sys.argv[2])
expected_cases = int(sys.argv[3])
queue = [
    line.split("\t")
    for line in (run_root / "generation/queue.tsv").read_text().splitlines()
    if line.strip()
]
if len(queue) != expected_jobs:
    raise SystemExit(f"expected {expected_jobs} queue rows, got {len(queue)}")

roots = []
manifest = []
for task_id, model, block, head in queue:
    path = run_root / "generation/validations" / f"{task_id}.json"
    payload = json.loads(path.read_text())
    if payload["num_cases"] != expected_cases:
        raise SystemExit(f"{path}: num_cases={payload['num_cases']}")
    roots.append(payload["result_root"])
    manifest.append(payload)
if len(set(roots)) != expected_jobs:
    raise SystemExit("metric result roots are not unique")
(run_root / "leaf_folders.txt").write_text("\n".join(roots) + "\n")
(run_root / "generation_manifest.json").write_text(
    json.dumps(
        {
            "num_configs": expected_jobs,
            "num_cases_per_config": expected_cases,
            "configs": manifest,
        },
        indent=2,
    )
    + "\n"
)
PY

METRIC_ROOT="${RUN_ROOT}/metrics"
mkdir -p "${METRIC_ROOT}/queues" "${METRIC_ROOT}/logs" \
  "${METRIC_ROOT}/state" "${METRIC_ROOT}/task_summaries"
: > "${METRIC_ROOT}/completed_tasks.tsv"
: > "${METRIC_ROOT}/failed_tasks.tsv"
for kind in cpu gpu_common videophy2 cosmos; do
  : > "${METRIC_ROOT}/queues/${kind}.tsv"
  printf '1\n' > "${METRIC_ROOT}/queues/${kind}.cursor"
done

read -r -a cpu_metrics <<< "${CPU_METRICS}"
read -r -a gpu_metrics <<< "${GPU_COMMON_METRICS}"
read -r -a videophy_metrics <<< "${VIDEOPHY2_METRICS}"
read -r -a cosmos_metrics <<< "${COSMOS_METRICS}"

cpu_index=0
gpu_index=0
videophy_index=0
cosmos_index=0
while IFS= read -r result_root; do
  for metric in "${cpu_metrics[@]}"; do
    printf 'cpu-%06d\t%s\t%s\n' "${cpu_index}" "${metric}" "${result_root}" \
      >> "${METRIC_ROOT}/queues/cpu.tsv"
    cpu_index=$((cpu_index + 1))
  done
  for metric in "${gpu_metrics[@]}"; do
    printf 'gpu-%06d\t%s\t%s\n' "${gpu_index}" "${metric}" "${result_root}" \
      >> "${METRIC_ROOT}/queues/gpu_common.tsv"
    gpu_index=$((gpu_index + 1))
  done
  for metric in "${videophy_metrics[@]}"; do
    printf 'videophy2-%06d\t%s\t%s\n' "${videophy_index}" "${metric}" "${result_root}" \
      >> "${METRIC_ROOT}/queues/videophy2.tsv"
    videophy_index=$((videophy_index + 1))
  done
  for metric in "${cosmos_metrics[@]}"; do
    printf 'cosmos-%06d\t%s\t%s\n' "${cosmos_index}" "${metric}" "${result_root}" \
      >> "${METRIC_ROOT}/queues/cosmos.tsv"
    cosmos_index=$((cosmos_index + 1))
  done
done < "${RUN_ROOT}/leaf_folders.txt"

EXPECTED_METRIC_TASKS=$((cpu_index + gpu_index + videophy_index + cosmos_index))
printf 'cpu=%s\ngpu_common=%s\nvideophy2=%s\ncosmos=%s\ntotal=%s\n' \
  "${cpu_index}" "${gpu_index}" "${videophy_index}" "${cosmos_index}" \
  "${EXPECTED_METRIC_TASKS}" > "${METRIC_ROOT}/queue_summary.txt"
touch "${RUN_ROOT}/metrics.ready"
echo "[coordinator] generation complete; metric tasks=${EXPECTED_METRIC_TASKS}"

while true; do
  workers="$(find "${METRIC_ROOT}/state" -name '*.complete' -type f | wc -l)"
  done_tasks="$(wc -l < "${METRIC_ROOT}/completed_tasks.tsv")"
  failed_tasks="$(wc -l < "${METRIC_ROOT}/failed_tasks.tsv")"
  printf '[coordinator] metrics tasks=%s/%s failed=%s workers=%s/%s\n' \
    "${done_tasks}" "${EXPECTED_METRIC_TASKS}" "${failed_tasks}" \
    "${workers}" "${NUM_METRIC_WORKERS}"
  if [[ "${workers}" -eq "${NUM_METRIC_WORKERS}" ]]; then
    break
  fi
  sleep "${COORDINATOR_POLL_SECONDS}"
done

"${PYTHON}" "${SUMMARY}" \
  --input-txt "${RUN_ROOT}/leaf_folders.txt" \
  --output-csv "${METRIC_ROOT}/metric_summary.csv" \
  --input-json-allowlist "${INPUT_LIST}"
"${PYTHON}" "${VERIFY_METRICS}" \
  --baseline-list "${RUN_ROOT}/leaf_folders.txt" \
  --output "${METRIC_ROOT}/verification.json" \
  --input-json-allowlist "${INPUT_LIST}"

done_tasks="$(wc -l < "${METRIC_ROOT}/completed_tasks.tsv")"
failed_tasks="$(wc -l < "${METRIC_ROOT}/failed_tasks.tsv")"
if [[ "${failed_tasks}" -ne 0 || "${done_tasks}" -ne "${EXPECTED_METRIC_TASKS}" ]]; then
  touch "${RUN_ROOT}/metrics.failed"
  echo "[coordinator] metric completion mismatch"
  exit 1
fi
touch "${RUN_ROOT}/pipeline.complete"
echo "[coordinator] pipeline complete"
