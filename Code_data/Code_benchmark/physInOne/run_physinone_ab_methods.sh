#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-wan}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/AAA_test_video/Benchmark/physInOne_AB}"
DATASET_ROOT="${DATASET_ROOT:-/data/gaoya/dataset/vLAR-PhysInOne}"
WAN_ROOT="${WAN_ROOT:-/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B}"
BENCH_CONFIG="${BENCH_CONFIG:-/home/gaoya/Code_Video/Code_data/Code_benchmark/physInOne/bench_paths.local.yaml}"
SELECTION_MODE="${SELECTION_MODE:-contains}"
CAMERA_NAME="${CAMERA_NAME:-CineCamera_0}"
GROUPS="${GROUPS:-A B}"

TI2V_PY="/home/gaoya/Code_Video/Code_data/Code_benchmark/physInOne/wan22_ti2v_physinone_benchmark.py"
TV2V_PY="/home/gaoya/Code_Video/Code_data/Code_benchmark/physInOne/wan22_tv2v_physinone_benchmark.py"
BUILD_PY="/home/gaoya/Code_Video/Code_data/Code_benchmark/physInOne/build_physinone_ab_benchmark.py"
SUMMARY_PY="/home/gaoya/Code_Video/Code_data/Code_benchmark/physInOne/summarize_physinone_benchmark.py"

mkdir -p "${OUTPUT_ROOT}"

conda run -n "${CONDA_ENV}" python "${BUILD_PY}" \
  --dataset_root "${DATASET_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --selection_mode "${SELECTION_MODE}" \
  --camera_name "${CAMERA_NAME}" \
  --groups ${GROUPS}

for GROUP in ${GROUPS}; do
  SOURCE_MANIFEST="${OUTPUT_ROOT}/prepared/group_${GROUP}_${SELECTION_MODE}_source_manifest.jsonl"

  conda run -n "${CONDA_ENV}" python "${TI2V_PY}" \
    --source_manifest "${SOURCE_MANIFEST}" \
    --model_root "${WAN_ROOT}" \
    --bench_config "${BENCH_CONFIG}" \
    --output_root "${OUTPUT_ROOT}/${GROUP}/ti2v" \
    --model_name "wan22_ti2v_5b_physinone_${GROUP}_${SELECTION_MODE}"

  conda run -n "${CONDA_ENV}" python "${TV2V_PY}" \
    --source_manifest "${SOURCE_MANIFEST}" \
    --wan_root "${WAN_ROOT}" \
    --bench_config "${BENCH_CONFIG}" \
    --output_root "${OUTPUT_ROOT}/${GROUP}/tv2v" \
    --model_name "wan22_tv2v_pretrain_ctx8_physinone_${GROUP}_${SELECTION_MODE}"

  conda run -n "${CONDA_ENV}" python "${SUMMARY_PY}" \
    --output_root "${OUTPUT_ROOT}/${GROUP}/ti2v/benchmarks" \
    --summary_csv "${OUTPUT_ROOT}/${GROUP}/ti2v/benchmarks/summary.csv"

  conda run -n "${CONDA_ENV}" python "${SUMMARY_PY}" \
    --output_root "${OUTPUT_ROOT}/${GROUP}/tv2v/benchmarks" \
    --summary_csv "${OUTPUT_ROOT}/${GROUP}/tv2v/benchmarks/summary.csv"
done
