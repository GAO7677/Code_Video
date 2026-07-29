#!/usr/bin/env bash
set -euo pipefail
# tmux new-session -d -s recompute_AAAevalphysiq1_metrics "bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_recompute_AAAevalphysiq1_metrics.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_SH="${SCRIPT_DIR}/bench.sh"
INPUT_LIST="${SCRIPT_DIR}/AAAevalphysiq1.txt"
ALLOWLIST="/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt"
LOG_DIR="/data/gaoya/agent-data/outputs/metric_recompute_20260729"
ALLOWED_GPUS=(0 1 2 3 5 6 7)
MIN_FREE_MIB="${MIN_FREE_MIB:-30000}"
MAX_UTIL_PERCENT="${MAX_UTIL_PERCENT:-10}"
POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p "${LOG_DIR}"

echo "[recompute] input_list=${INPUT_LIST}"
echo "[recompute] allowlist=${ALLOWLIST}"
echo "[recompute] allowed_gpus=${ALLOWED_GPUS[*]}"
echo "[recompute] source hashes:"
sha256sum \
  "${BENCH_SH}" \
  "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/bench.py" \
  "/home/gaoya/Code_Video/Code_data/Code_try0526/physv_eval/single_case/physics_iq.py" \
  "/home/gaoya/Code_Video/Code_data/Code_try0526/physv_eval/single_case/videophy2.py"

echo "[recompute] phase=physics_iq start"
BENCH_RUN_METRICS=1 \
BENCH_OVERWRITE=1 \
BENCH_METRICS="physics_iq_with_context,physics_iq_without_context" \
BENCH_INPUT_JSON_ALLOWLIST="${ALLOWLIST}" \
CUDA_VISIBLE_DEVICES="" \
bash "${BENCH_SH}" "${INPUT_LIST}"
echo "[recompute] phase=physics_iq done"

select_available_gpu() {
  local index free_mib util allowed
  while IFS=',' read -r index free_mib util; do
    index="${index//[[:space:]]/}"
    free_mib="${free_mib//[[:space:]]/}"
    util="${util//[[:space:]]/}"
    allowed=0
    for candidate in "${ALLOWED_GPUS[@]}"; do
      if [[ "${index}" == "${candidate}" ]]; then
        allowed=1
        break
      fi
    done
    if (( allowed == 1 && free_mib >= MIN_FREE_MIB && util <= MAX_UTIL_PERCENT )); then
      printf '%s\n' "${index}"
      return 0
    fi
  done < <(
    nvidia-smi \
      --query-gpu=index,memory.free,utilization.gpu \
      --format=csv,noheader,nounits
  )
  return 1
}

echo "[recompute] phase=videophy2 waiting_for_gpu min_free_mib=${MIN_FREE_MIB} max_util=${MAX_UTIL_PERCENT}"
while ! SELECTED_GPU="$(select_available_gpu)"; do
  date -u '+[recompute] %Y-%m-%dT%H:%M:%SZ no allowed GPU is ready'
  sleep "${POLL_SECONDS}"
done

echo "[recompute] phase=videophy2 start physical_gpu=${SELECTED_GPU}"
BENCH_RUN_METRICS=1 \
BENCH_OVERWRITE=1 \
BENCH_METRICS="videophy2" \
BENCH_INPUT_JSON_ALLOWLIST="${ALLOWLIST}" \
BENCH_CUDA_VISIBLE_DEVICES="${SELECTED_GPU}" \
CUDA_VISIBLE_DEVICES="${SELECTED_GPU}" \
bash "${BENCH_SH}" "${INPUT_LIST}"
echo "[recompute] phase=videophy2 done"
echo "[recompute] all requested metrics completed"
