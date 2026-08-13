#!/usr/bin/env bash
set -euo pipefail

gpu=3
memory_threshold_mb="${GPU_WAIT_THRESHOLD_MB:-12000}"
repo=/home/gaoya/Code_Video/DiffTrack-main
runner="${repo}/AAA_my_test/wan_context_point_guidance/run_direct_attention_protocol.py"
python_bin=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
output=/data/gaoya/agent-data/outputs/wan_context_point_guidance_head_compare/direct_attention_tv_v1
state_dir="${output}/multicase_pilot_state"
log_dir="${output}/logs"
mkdir -p "${state_dir}" "${log_dir}"
rm -f "${state_dir}/gpu3.done" "${state_dir}/gpu3.failed"
exec > >(tee -a "${log_dir}/multicase_gpu3.log") 2>&1

on_error() {
  status=$?
  printf '%s\n' "exit_status=${status}" > "${state_dir}/gpu3.failed"
  exit "${status}"
}
trap on_error ERR

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

cases=(
  0613pybullet_sample_001460_w002
  0613pybullet_sample_001455_w000
  0613pybullet_sample_000336_w001
  phyco_kubric_ball_wall_collision_2025-08-08_00ac15
  physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed_crop_top60px
)

echo "[$(date -u +%FT%TZ)] GPU${gpu} starting seed 90094"
"${python_bin}" -u "${runner}" \
  --backend firstframe_ti2v \
  --stage all \
  --device cuda:0 \
  --seed 90094 \
  --case-keys "${cases[@]}" \
  --head-groups top100 bottom100 random100 \
  --directions context_to_future future_to_context bidirectional \
  --attention-tv-budget 0.10 \
  --guidance-start 0 \
  --guidance-end 39 \
  --attention-capture-steps 5 10 15 20 25 30 35 40 \
  --output-root "${output}"

date -u +%FT%TZ > "${state_dir}/gpu3.done"
echo "[$(date -u +%FT%TZ)] GPU${gpu} shard complete"
