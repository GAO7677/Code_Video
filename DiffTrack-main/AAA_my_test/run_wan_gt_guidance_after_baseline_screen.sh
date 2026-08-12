#!/usr/bin/env bash
set -euo pipefail

repo=/home/gaoya/Code_Video/DiffTrack-main
python_bin=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
output_root=/data/gaoya/agent-data/outputs/wan_gt_spatiotemporal_correspondence_guidance/latest3350_top100_cotracker_sam2_v2
screen_json="$output_root/screening/seed_47326/baseline_eligibility.json"
log_root="$output_root/logs"
mkdir -p "$log_root"

cd "$repo"
export PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main:/home/gaoya/Code_Video/Code_data/Code_train/train_0419
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "[wait] waiting for a complete 20-case Baseline screen: $screen_json"
while ! "$python_bin" - "$screen_json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
payload = json.loads(path.read_text(encoding="utf-8"))
raise SystemExit(0 if payload.get("missing_case_count") == 0 else 1)
PY
do
    sleep 30
done

eligible_count=$("$python_bin" - "$screen_json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(int(payload["eligible_target_count"]))
PY
)
echo "[screen] eligible targets: $eligible_count"

if [[ "$eligible_count" -gt 0 ]]; then
    "$python_bin" AAA_my_test/run_wan_gt_spatiotemporal_correspondence_guidance.py \
        --stage generate \
        --device cuda:0 \
        --seed 47326 \
        --target-map "$screen_json" \
        --no-baseline \
        --loss-modes region point combined \
        --guidance-scale 0.1 \
        2>&1 | tee "$log_root/full20_guidance_lambda0p1.log"

    "$python_bin" AAA_my_test/run_wan_gt_spatiotemporal_correspondence_guidance.py \
        --stage evaluate \
        --device cuda:0 \
        --seed 47326 \
        --target-map "$screen_json" \
        2>&1 | tee "$log_root/full20_guidance_lambda0p1_evaluate.log"
fi

"$python_bin" AAA_my_test/analyze_wan_gt_guidance_frozen_validation.py \
    --output-root "$output_root" \
    --seed 47326 \
    --lambdas 0.1 \
    --strict \
    2>&1 | tee "$log_root/full20_primary_analysis.log"

report_json="$output_root/final_analysis/seed_47326/frozen_validation_report.json"
mapfile -t trigger_modes < <("$python_bin" - "$report_json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for mode in payload["trigger_modes"]:
    print(mode)
PY
)

if [[ "${#trigger_modes[@]}" -eq 0 ]]; then
    echo "[sensitivity] frozen trigger not met; lambda sweep is not run"
    exit 0
fi

echo "[sensitivity] trigger modes: ${trigger_modes[*]}"
for guidance_scale in 0.05 0.2; do
    "$python_bin" AAA_my_test/run_wan_gt_spatiotemporal_correspondence_guidance.py \
        --stage generate \
        --device cuda:0 \
        --seed 47326 \
        --target-map "$screen_json" \
        --no-baseline \
        --loss-modes "${trigger_modes[@]}" \
        --guidance-scale "$guidance_scale" \
        2>&1 | tee "$log_root/full20_guidance_lambda${guidance_scale/./p}.log"
done

"$python_bin" AAA_my_test/run_wan_gt_spatiotemporal_correspondence_guidance.py \
    --stage evaluate \
    --device cuda:0 \
    --seed 47326 \
    --target-map "$screen_json" \
    2>&1 | tee "$log_root/full20_sensitivity_evaluate.log"

"$python_bin" AAA_my_test/analyze_wan_gt_guidance_frozen_validation.py \
    --output-root "$output_root" \
    --seed 47326 \
    --lambdas 0.05 0.1 0.2 \
    --strict \
    2>&1 | tee "$log_root/full20_final_analysis.log"
