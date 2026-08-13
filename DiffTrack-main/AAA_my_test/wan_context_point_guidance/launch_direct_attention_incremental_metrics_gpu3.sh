#!/usr/bin/env bash
set -euo pipefail

gpu=3
repo=/home/gaoya/Code_Video/DiffTrack-main
python_bin=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
runner="${repo}/AAA_my_test/wan_context_point_guidance/run_direct_attention_protocol.py"
vbench_driver=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/bench.py
output=/data/gaoya/agent-data/outputs/wan_context_point_guidance_head_compare/direct_attention_tv_v1
state_dir="${output}/multicase_pilot_state"
vbench_root="${output}/vbench_multicase"
log_dir="${output}/logs"
snapshot_manifest="${state_dir}/gpu3_incremental_metrics_snapshot.json"

mkdir -p "${state_dir}" "${log_dir}"
rm -f "${state_dir}/gpu3_metrics.done" "${state_dir}/gpu3_metrics.failed"
exec > >(tee -a "${log_dir}/incremental_metrics_gpu3.log") 2>&1

on_error() {
  status=$?
  printf '%s\n' "exit_status=${status}" > "${state_dir}/gpu3_metrics.failed"
  exit "${status}"
}
trap on_error ERR

export CUDA_VISIBLE_DEVICES="${gpu}"
export TOKENIZERS_PARALLELISM=false
cd "${repo}"

# GPU2 is still generating seed 47326, so this pass handles only the videos
# already complete when each case is visited.  The original GPU2 job later
# skips these files and fills any remaining trajectory reports.
echo "[$(date -u +%FT%TZ)] GPU${gpu} filling available seed 47326 trajectory metrics"
"${python_bin}" -u "${runner}" \
  --backend firstframe_ti2v \
  --stage evaluate \
  --device cuda:0 \
  --seed 47326 \
  --case-keys \
    0613pybullet_sample_001460_w002 \
    0613pybullet_sample_001455_w000 \
    0613pybullet_sample_000336_w001 \
    phyco_kubric_ball_wall_collision_2025-08-08_00ac15 \
    physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed_crop_top60px \
  --output-root "${output}"

# Freeze the VBench cohort after the trajectory pass.  Rebuilding this index
# preserves prior scores, so the final GPU2 pass only evaluates later videos.
echo "[$(date -u +%FT%TZ)] freezing current complete-video VBench cohort"
"${python_bin}" -u AAA_my_test/wan_context_point_guidance/prepare_direct_attention_vbench.py
cp "${vbench_root}/snapshot.json" "${snapshot_manifest}"

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
  env PYTHONNOUSERSITE=1 "${python_bin}" -u "${vbench_driver}" \
    --metric "${metric}" \
    --result-root "${vbench_root}/index" \
    --output-summary "${vbench_root}/incremental_gpu3_${metric}.json" \
    --vbench-output-root "${vbench_root}/raw" \
    --vbench-device cuda
done

date -u +%FT%TZ > "${state_dir}/gpu3_metrics.done"
echo "[$(date -u +%FT%TZ)] GPU${gpu} incremental trajectory and seven-dimension VBench pass complete"
