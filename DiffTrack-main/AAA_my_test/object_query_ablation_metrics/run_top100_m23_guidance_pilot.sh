#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-1}"
if [[ "$gpu" == "4" ]]; then
  echo "GPU 4 is forbidden by workspace policy." >&2
  exit 2
fi

repo=/home/gaoya/Code_Video/DiffTrack-main
python_bin=/data/gaoya/miniconda3/envs/wan/bin/python
runner=AAA_my_test/object_query_ablation_metrics/run_top100_m1_perturbed_attention_guidance.py
output_root=/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1/training_free_top100_m23_guidance_v1
log_root="$output_root/logs"
mkdir -p "$log_root"

cases=(
  0613pybullet_sample_001460_w002
  0613pybullet_sample_000331_w001
  physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed-ball-and-block-fall_motion_to_end
)
flows=(m2 m3)

cd "$repo"
for case_id in "${cases[@]}"; do
  for flow in "${flows[@]}"; do
    log="$log_root/${case_id}__${flow}.log"
    echo "[$(date -u +%FT%TZ)] start case=$case_id flow=$flow gpu=$gpu" | tee "$log"
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u "$runner" \
      --case "$case_id" \
      --seed 47326 \
      --region object_A \
      --flow "$flow" \
      --time-scope all_time \
      --cfg-scale 5 \
      --pag-scale 0.5 \
      --sampling-steps 40 \
      --record-dose \
      --output-root "$output_root" 2>&1 | tee -a "$log"
    rc=${PIPESTATUS[0]}
    set -e
    if (( rc != 0 )); then
      variant="single_object__object_A__${flow}_all_time__top100__pag0p5"
      error_dir="$output_root/$case_id/seed_47326/$variant"
      mkdir -p "$error_dir"
      tail -n 300 "$log" > "$error_dir/error.txt"
      exit "$rc"
    fi
    echo "[$(date -u +%FT%TZ)] complete case=$case_id flow=$flow" | tee -a "$log"
  done
done
