#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 7 ]]; then
  echo "Usage: $0 RUN_ROOT ROOTS_FILE INPUT_ALLOWLIST CPU_WORKERS VIDEO_WORKERS EXPECTED_CASES ARTIFACT_ROOT" >&2
  exit 2
fi

RUN_ROOT="$1"
ROOTS_FILE="$2"
INPUT_ALLOWLIST="$3"
CPU_WORKERS="$4"
VIDEO_WORKERS="$5"
EXPECTED_CASES="$6"
ARTIFACT_ROOT="$7"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
SUMMARY=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/summarize_benchmark_txt_metrics.py

while true; do
  cpu_done="$(find "${RUN_ROOT}/state" -maxdepth 1 -type f -name 'cpu*.complete' | wc -l)"
  video_done="$(find "${RUN_ROOT}/state" -maxdepth 1 -type f -name 'g*_vp*.complete' | wc -l)"
  cpu_claimed="$(( $(<"${RUN_ROOT}/queues/cpu.cursor") - 1 ))"
  video_claimed="$(( $(<"${RUN_ROOT}/queues/videophy2.cursor") - 1 ))"
  completed="$(wc -l < "${RUN_ROOT}/completed_tasks.tsv")"
  failed="$(wc -l < "${RUN_ROOT}/failed_tasks.tsv")"
  printf '[coordinator] cpu_workers=%s/%s video_workers=%s/%s cpu_claimed=%s/152 video_claimed=%s/38 completed=%s failed=%s\n' \
    "${cpu_done}" "${CPU_WORKERS}" "${video_done}" "${VIDEO_WORKERS}" \
    "${cpu_claimed}" "${video_claimed}" "${completed}" "${failed}"
  if [[ "${cpu_done}" -eq "${CPU_WORKERS}" && "${video_done}" -eq "${VIDEO_WORKERS}" ]]; then
    break
  fi
  sleep 30
done

if [[ -s "${RUN_ROOT}/failed_tasks.tsv" ]]; then
  echo "[coordinator] task failures detected; see ${RUN_ROOT}/failed_tasks.tsv" >&2
  exit 1
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/verify_physics_iq_deterministic.py" \
  --result-roots "${ROOTS_FILE}" \
  --input-json-allowlist "${INPUT_ALLOWLIST}" \
  --expected-cases "${EXPECTED_CASES}" \
  --output "${RUN_ROOT}/verification_physics_iq.json"

"${PYTHON_BIN}" "${SCRIPT_DIR}/verify_pmf_time_alignment.py" \
  --result-roots "${ROOTS_FILE}" \
  --input-json-allowlist "${INPUT_ALLOWLIST}" \
  --expected-cases "${EXPECTED_CASES}" \
  --output "${RUN_ROOT}/verification_pmf.json"

"${PYTHON_BIN}" "${SCRIPT_DIR}/verify_videophy2_generated_only.py" \
  --result-roots "${ROOTS_FILE}" \
  --input-json-allowlist "${INPUT_ALLOWLIST}" \
  --expected-cases "${EXPECTED_CASES}" \
  --expected-context-frames 8 \
  --output "${RUN_ROOT}/verification_videophy2.json"

"${PYTHON_BIN}" "${SUMMARY}" \
  --input-txt "${ROOTS_FILE}" \
  --output-csv "${RUN_ROOT}/metric_summary_after_recompute.csv" \
  --input-json-allowlist "${INPUT_ALLOWLIST}"

touch "${RUN_ROOT}/pipeline.complete"
echo "[coordinator] all five metrics recomputed and verified"
echo "[coordinator] artifacts=${ARTIFACT_ROOT}"
exec bash
