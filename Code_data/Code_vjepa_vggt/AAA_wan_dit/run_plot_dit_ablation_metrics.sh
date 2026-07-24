#!/usr/bin/env bash
# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_plot_dit_ablation_metrics.sh
#
# Run from a leaf-folder txt:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_plot_dit_ablation_metrics.sh \
#   /data/gaoya/AAA_test_video/0623/test/v2v_wan/PhyRVG/rvg_leaf_folders.txt
#
# Optional second argument: output directory.

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
INPUT_TXT="${1:-${INPUT_TXT:-}}"
OUTPUT_DIR="${2:-${OUTPUT_DIR:-}}"
INPUT_JSON_ALLOWLIST="${INPUT_JSON_ALLOWLIST:-/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt}"
EXPECTED_CASES="${EXPECTED_CASES:-67}"

args=(
  --input-json-allowlist "${INPUT_JSON_ALLOWLIST}"
  --expected-cases "${EXPECTED_CASES}"
)

if [[ -n "${INPUT_TXT}" ]]; then
  if [[ ! -s "${INPUT_TXT}" ]]; then
    echo "Missing or empty input txt: ${INPUT_TXT}" >&2
    exit 2
  fi
  if [[ -z "${OUTPUT_DIR}" ]]; then
    txt_dir="$(cd -- "$(dirname -- "${INPUT_TXT}")" && pwd)"
    txt_name="$(basename -- "${INPUT_TXT}")"
    OUTPUT_DIR="${txt_dir}/_metric_plots/${txt_name%.*}"
  fi
  args+=(--input-txt "${INPUT_TXT}")
else
  args+=(--result-root /data/gaoya/AAA_test_video/0623/test/v2v_wan)
  OUTPUT_DIR="${OUTPUT_DIR:-/data/gaoya/AAA_test_video/0623/test/v2v_wan/_metric_plots}"
fi

args+=(--output-dir "${OUTPUT_DIR}")
"${PYTHON_BIN}" "${SCRIPT_DIR}/plot_dit_ablation_metrics.py" "${args[@]}"
