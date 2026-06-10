#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/gaoya/Code_Video/Code_data/Code_try0526"
GEN_PY="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
FLUX_PY="/home/gaoya/miniconda3/envs/flux/bin/python"
VPHY_PY="/data/gaoya/miniconda3/envs/vphy/bin/python"

BENCH_ROOT="/data/gaoya/AAA_test_video/Output_try0526/Dataset_physV_B_benchmark"
ABD_B_ROOT="/data/gaoya/AAA_test_video/Output_try0526/ABD_test/B"
SUMMARY_CSV="${ABD_B_ROOT}/_meta/method_metrics_summary.csv"
LOG_ROOT="/data/gaoya/AAA_test_video/Output_try0526/Dataset_physV_B_benchmark_eval_logs"

METHODS=(GT wan22-5B-TI2V VACE_1p3B_TI2V VACE_1p3B_ctx08)
GPU_IDS=(5 6 7)
NUM_SHARDS="${#GPU_IDS[@]}"

mkdir -p "${LOG_ROOT}"

wait_for_no_match() {
  local pattern="$1"
  while pgrep -f "${pattern}" >/dev/null 2>&1; do
    sleep 30
  done
}

run_flux_phase() {
  local metric_name="$1"
  local pids=()
  local i=0
  for gpu in "${GPU_IDS[@]}"; do
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
      "${FLUX_PY}" "${ROOT}/eval_benchmark_dir_metrics.py" \
      --input-root "${ABD_B_ROOT}" \
      --methods "${METHODS[@]}" \
      --metrics "${metric_name}" \
      --num-shards "${NUM_SHARDS}" \
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
  local pids=()
  local i=0
  for gpu in "${GPU_IDS[@]}"; do
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
      "${VPHY_PY}" "${ROOT}/eval_benchmark_dir_metrics.py" \
      --input-root "${ABD_B_ROOT}" \
      --methods "${METHODS[@]}" \
      --metrics videophy2 \
      --num-shards "${NUM_SHARDS}" \
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

echo "[step] regenerate B benchmark videos"
CUDA_VISIBLE_DEVICES=7 PYTHONUNBUFFERED=1 \
  "${GEN_PY}" "${ROOT}/generate_dataset_physv_b_methods.py" \
  --wan-backend legacy \
  > "/data/gaoya/AAA_test_video/Output_try0526/Dataset_physV_B_benchmark.run.log" 2>&1

echo "[step] sync methods into ABD_test/B"
PYTHONUNBUFFERED=1 \
  "${GEN_PY}" "${ROOT}/sync_dataset_physv_b_to_abd_test.py" \
  > "/data/gaoya/AAA_test_video/Output_try0526/Dataset_physV_B_benchmark.sync.log" 2>&1

echo "[step] rebuild ABD B GT + _meta"
PYTHONUNBUFFERED=1 \
  python3 "${ROOT}/organize_output_try0526_abd.py" \
  > "/data/gaoya/AAA_test_video/Output_try0526/Dataset_physV_B_benchmark.organize.log" 2>&1

wait_for_no_match "eval_benchmark_dir_metrics.py --input-root ${ABD_B_ROOT}"

echo "[step] refresh pdi"
run_flux_phase pdi
echo "[step] refresh wmreward"
run_flux_phase wmreward
echo "[step] refresh proxy"
run_flux_phase proxy
echo "[step] refresh cosmos"
run_flux_phase cosmos
echo "[step] refresh videophy2"
run_videophy_phase

echo "[step] write summary csv"
PYTHONUNBUFFERED=1 \
  "${FLUX_PY}" "${ROOT}/eval_benchmark_dir_metrics.py" \
  --input-root "${ABD_B_ROOT}" \
  --methods "${METHODS[@]}" \
  --summary-only \
  --summary-csv "${SUMMARY_CSV}" \
  > "${LOG_ROOT}/summary.log" 2>&1

echo "[done] B benchmark refresh complete"
