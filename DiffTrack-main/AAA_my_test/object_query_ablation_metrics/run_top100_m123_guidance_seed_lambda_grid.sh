#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-1}"
wait_session="${2:-top100_m23_guidance_pilot_20260813}"
if [[ "$gpu" == "4" ]]; then
  echo "GPU 4 is forbidden by workspace policy." >&2
  exit 2
fi

repo=/home/gaoya/Code_Video/DiffTrack-main
python_bin=/data/gaoya/miniconda3/envs/wan/bin/python
runner=AAA_my_test/object_query_ablation_metrics/run_top100_m1_perturbed_attention_guidance.py
baseline_runner=AAA_my_test/run_legacy_ti2v_firstlatent_physiciq67_attention_zero_ablations.py
manifest_builder=AAA_my_test/object_query_ablation_metrics/build_top100_m123_guidance_grid_manifest.py
output_root=/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1/training_free_top100_m23_guidance_v1
manifest="$output_root/guidance_grid_manifest.json"
tracks_root=/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1/stage4_temporal_v1
log_root="$output_root/grid_logs"
mkdir -p "$log_root"

cd "$repo"
"$python_bin" "$manifest_builder"

while tmux has-session -t "$wait_session" 2>/dev/null; do
  echo "[$(date -u +%FT%TZ)] waiting for $wait_session before reusing GPU $gpu"
  sleep 30
done

baseline_log="$log_root/seed42_baselines.log"
echo "[$(date -u +%FT%TZ)] start three seed-42 baselines on gpu=$gpu" | tee "$baseline_log"
CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u "$baseline_runner" \
  --worker-id 0 --num-workers 1 \
  --manifest-path "$manifest" \
  --baselines-only 2>&1 | tee -a "$baseline_log"

cases=(
  0613pybullet_sample_001460_w002
  0613pybullet_sample_000331_w001
  physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed-ball-and-block-fall_motion_to_end
)
flows=(m1 m2 m3)

run_guidance() {
  local case_id="$1" seed="$2" flow="$3" scale="$4"
  local scale_tag="${scale/./p}"
  local log="$log_root/${case_id}__seed${seed}__${flow}__lambda${scale_tag}.log"
  echo "[$(date -u +%FT%TZ)] start case=$case_id seed=$seed flow=$flow lambda=$scale gpu=$gpu" | tee "$log"
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u "$runner" \
    --case "$case_id" \
    --seed "$seed" \
    --manifest-path "$manifest" \
    --tracks-root "$tracks_root" \
    --output-root "$output_root" \
    --region object_A \
    --flow "$flow" \
    --time-scope all_time \
    --cfg-scale 5 \
    --pag-scale "$scale" \
    --sampling-steps 40 \
    --prepare-tracks \
    --record-dose 2>&1 | tee -a "$log"
  local rc=${PIPESTATUS[0]}
  set -e
  if (( rc != 0 )); then
    local variant="single_object__object_A__${flow}_all_time__top100__pag${scale_tag}"
    local error_dir="$output_root/$case_id/seed_$(printf '%05d' "$seed")/$variant"
    mkdir -p "$error_dir"
    tail -n 300 "$log" > "$error_dir/error.txt"
    return "$rc"
  fi
  echo "[$(date -u +%FT%TZ)] complete case=$case_id seed=$seed flow=$flow lambda=$scale" | tee -a "$log"
}

# First add the requested second seed at the original pilot strength.
for case_id in "${cases[@]}"; do
  for flow in "${flows[@]}"; do
    run_guidance "$case_id" 42 "$flow" 0.5
  done
done

# Then complete lambda=1 for every case x seed x flow.
for seed in 47326 42; do
  for case_id in "${cases[@]}"; do
    for flow in "${flows[@]}"; do
      run_guidance "$case_id" "$seed" "$flow" 1
    done
  done
done
