#!/usr/bin/env bash
set -euo pipefail

gpu=2
memory_threshold_mb="${GPU_WAIT_THRESHOLD_MB:-12000}"
repo=/home/gaoya/Code_Video/DiffTrack-main
python_bin=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
driver=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/bench.py
output=/data/gaoya/agent-data/outputs/wan_context_point_guidance_head_compare/direct_attention_tv_v1
state_dir="${output}/multicase_pilot_state"
vbench_root="${output}/vbench_multicase"
log_dir="${output}/logs"
mkdir -p "${state_dir}" "${log_dir}"
rm -f "${state_dir}/vbench.done" "${state_dir}/vbench.failed"
exec > >(tee -a "${log_dir}/multicase_vbench_gpu2.log") 2>&1

on_error() {
  status=$?
  printf '%s\n' "exit_status=${status}" > "${state_dir}/vbench.failed"
  exit "${status}"
}
trap on_error ERR

while [[ ! -f "${state_dir}/gpu2.done" || ! -f "${state_dir}/gpu1.done" ]]; do
  if [[ -f "${state_dir}/gpu2.failed" || -f "${state_dir}/gpu1.failed" ]]; then
    echo "generation shard failed; refusing to run incomplete VBench cohort"
    exit 1
  fi
  echo "[$(date -u +%FT%TZ)] waiting for both generation shards"
  sleep 30
done

stable_checks=0
while true; do
  used_mb="$(nvidia-smi --id="${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  if [[ "${used_mb}" =~ ^[0-9]+$ ]] && (( used_mb <= memory_threshold_mb )); then
    stable_checks=$((stable_checks + 1))
    if (( stable_checks >= 3 )); then
      break
    fi
    echo "[$(date -u +%FT%TZ)] GPU${gpu} low-memory stability ${stable_checks}/3: ${used_mb} MiB"
  else
    stable_checks=0
    echo "[$(date -u +%FT%TZ)] waiting for GPU${gpu}: ${used_mb} MiB used; threshold=${memory_threshold_mb} MiB"
  fi
  sleep 30
done

export CUDA_VISIBLE_DEVICES="${gpu}"
export TOKENIZERS_PARALLELISM=false
cd "${repo}"
"${python_bin}" -u AAA_my_test/wan_context_point_guidance/prepare_direct_attention_vbench.py

metrics=(
  vbench_subject_consistency
  vbench_background_consistency
  vbench_temporal_flickering
  vbench_motion_smoothness
  vbench_dynamic_degree
  vbench_aesthetic_quality
  vbench_imaging_quality
)
for metric in "${metrics[@]}"; do
  echo "[$(date -u +%FT%TZ)] GPU${gpu} starting ${metric}"
  env PYTHONNOUSERSITE=1 "${python_bin}" -u "${driver}" \
    --metric "${metric}" \
    --result-root "${vbench_root}/index" \
    --output-summary "${vbench_root}/eval_summary_${metric}.json" \
    --vbench-output-root "${vbench_root}/raw" \
    --vbench-device cuda
done

date -u +%FT%TZ > "${state_dir}/vbench.done"
echo "[$(date -u +%FT%TZ)] all seven VBench dimensions complete"
