#!/usr/bin/env bash
set -euo pipefail

wait_pid="${WAIT_PID:-2590613}"
repo=/home/gaoya/Code_Video/DiffTrack-main
runner="${repo}/AAA_my_test/wan_context_point_guidance/run_direct_attention_protocol.py"
python_bin=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
output=/data/gaoya/agent-data/outputs/wan_context_point_guidance_head_compare/direct_attention_tv_v1
log_dir="${output}/logs"
mkdir -p "${log_dir}"
exec > >(tee -a "${log_dir}/gpu2.log") 2>&1

while [[ -r "/proc/${wait_pid}/cmdline" ]]; do
  current_cmd="$(tr '\0' ' ' < "/proc/${wait_pid}/cmdline")"
  if [[ "${current_cmd}" != *"run_dual_protocol.py"* ]]; then
    break
  fi
  echo "[$(date -u +%FT%TZ)] waiting for GPU2 PID ${wait_pid}: ${current_cmd}"
  sleep 30
done

export CUDA_VISIBLE_DEVICES=2
export TOKENIZERS_PARALLELISM=false
cd "${repo}"
echo "[$(date -u +%FT%TZ)] starting direct-attention sanity + focal matrix on physical GPU2"
"${python_bin}" -u "${runner}" \
  --backend firstframe_ti2v \
  --stage all \
  --device cuda:0 \
  --case-keys 0613pybullet_sample_001460_w002 \
  --head-groups top100 bottom100 random100 \
  --directions context_to_future future_to_context bidirectional \
  --attention-tv-budget 0.10 \
  --guidance-start 0 \
  --guidance-end 39 \
  --attention-capture-steps 5 10 15 20 25 30 35 40 \
  --output-root "${output}"
