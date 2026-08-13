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

# Guidance peaks near the full 48 GiB card.  Do not race an unrelated user
# workload: wait for the physical GPU exposed by CUDA_VISIBLE_DEVICES, and
# retry only a confirmed CUDA OOM.  All completed variants are protected by
# the Python runner's atomic complete.json markers.
physical_gpu_index="${CUDA_VISIBLE_DEVICES:-7}"
physical_gpu_index="${physical_gpu_index%%,*}"
required_free_mib=47000
required_stable_checks=5

wait_for_guidance_capacity() {
    local free_mib
    local stable_checks=0
    while true; do
        free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$physical_gpu_index" | head -n 1 | tr -d ' ')
        if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= required_free_mib )); then
            stable_checks=$((stable_checks + 1))
            echo "[gpu-capacity] physical GPU $physical_gpu_index candidate: ${free_mib} MiB free (${stable_checks}/${required_stable_checks} stable checks)"
            if (( stable_checks >= required_stable_checks )); then
                echo "[gpu-capacity] physical GPU $physical_gpu_index ready after stable-idle window"
                return 0
            fi
        else
            stable_checks=0
            echo "[gpu-capacity] physical GPU $physical_gpu_index has ${free_mib:-unknown} MiB free; waiting for >= ${required_free_mib} MiB"
        fi
        sleep 30
    done
}

run_gpu_python_with_oom_retry() {
    local log_path="$1"
    shift
    local attempt=0
    local attempt_log
    local status
    while true; do
        attempt=$((attempt + 1))
        wait_for_guidance_capacity
        attempt_log="${log_path%.log}.attempt_$(date -u +%Y%m%dT%H%M%SZ)_${attempt}.log"
        echo "[gpu-run] attempt=$attempt log=$attempt_log" | tee -a "$log_path"
        set +e
        "$python_bin" "$@" 2>&1 | tee -a "$attempt_log" "$log_path"
        status=${PIPESTATUS[0]}
        set -e
        if (( status == 0 )); then
            return 0
        fi
        if rg -q 'torch\.OutOfMemoryError|CUDA out of memory' "$attempt_log"; then
            echo "[gpu-run] confirmed CUDA OOM; preserving completed variants and retrying after capacity is available" | tee -a "$log_path"
            sleep 30
            continue
        fi
        echo "[gpu-run] non-OOM failure (status=$status); refusing to hide it" | tee -a "$log_path"
        return "$status"
    done
}

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

primary_report="$output_root/final_analysis/seed_47326/frozen_validation_report.json"
primary_complete=false
if [[ -f "$primary_report" ]] && "$python_bin" - "$primary_report" <<'PY'
import json
import math
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
rows = [row for row in payload.get("aggregate", []) if math.isclose(float(row["lambda"]), 0.1)]
ok = len(rows) == 3 and all(
    int(row["completed_target_count"]) == int(row["eligible_target_count"])
    for row in rows
)
raise SystemExit(0 if ok else 1)
PY
then
    primary_complete=true
fi

if [[ "$eligible_count" -gt 0 && "$primary_complete" != true ]]; then
    run_gpu_python_with_oom_retry "$log_root/full20_guidance_lambda0p1.log" \
        AAA_my_test/run_wan_gt_spatiotemporal_correspondence_guidance.py \
        --stage generate \
        --device cuda:0 \
        --seed 47326 \
        --target-map "$screen_json" \
        --no-baseline \
        --loss-modes region point combined \
        --guidance-scale 0.1

    run_gpu_python_with_oom_retry "$log_root/full20_guidance_lambda0p1_evaluate.log" \
        AAA_my_test/run_wan_gt_spatiotemporal_correspondence_guidance.py \
        --stage evaluate \
        --device cuda:0 \
        --seed 47326 \
        --target-map "$screen_json"
elif [[ "$primary_complete" == true ]]; then
    echo "[primary] 30/30 generated variants and trajectory metrics already complete; skipping model reload"
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
    run_gpu_python_with_oom_retry "$log_root/full20_guidance_lambda${guidance_scale/./p}.log" \
        AAA_my_test/run_wan_gt_spatiotemporal_correspondence_guidance.py \
        --stage generate \
        --device cuda:0 \
        --seed 47326 \
        --target-map "$screen_json" \
        --no-baseline \
        --loss-modes "${trigger_modes[@]}" \
        --guidance-scale "$guidance_scale"
done

run_gpu_python_with_oom_retry "$log_root/full20_sensitivity_evaluate.log" \
    AAA_my_test/run_wan_gt_spatiotemporal_correspondence_guidance.py \
    --stage evaluate \
    --device cuda:0 \
    --seed 47326 \
    --target-map "$screen_json"

"$python_bin" AAA_my_test/analyze_wan_gt_guidance_frozen_validation.py \
    --output-root "$output_root" \
    --seed 47326 \
    --lambdas 0.05 0.1 0.2 \
    --strict \
    2>&1 | tee "$log_root/full20_final_analysis.log"

"$python_bin" AAA_my_test/summarize_wan_gt_guidance_optimization_audit.py \
    --output-root "$output_root" \
    --seed 47326 \
    --lambdas 0.05 0.1 0.2 \
    2>&1 | tee "$log_root/full20_optimization_audit.log"

"$python_bin" - "$output_root" <<'PY'
import datetime
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
final_dir = root / "final_analysis" / "seed_47326"
report = json.loads((final_dir / "frozen_validation_report.json").read_text(encoding="utf-8"))
audit = json.loads((final_dir / "optimization_audit.json").read_text(encoding="utf-8"))
expected_pairs = {(0.1, mode) for mode in ("region", "point", "combined")}
expected_pairs |= {
    (guidance_lambda, mode)
    for guidance_lambda in (0.05, 0.2)
    for mode in report["trigger_modes"]
}
actual_pairs = {(float(row["lambda"]), row["mode"]) for row in report["aggregate"]}
if actual_pairs != expected_pairs:
    raise SystemExit(f"registered aggregate mismatch: {actual_pairs} != {expected_pairs}")
if any(
    int(row["completed_target_count"]) != int(row["eligible_target_count"])
    for row in report["aggregate"]
):
    raise SystemExit("final outcome report has incomplete registered variants")
if any(
    int(row["complete_variants"]) != int(row["expected_variants"])
    or not row["all_complete_variants_have_40_steps"]
    or not row["all_complete_variants_finite"]
    for row in audit["aggregate"]
):
    raise SystemExit("final optimization audit is incomplete or non-finite")
marker = {
    "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "protocol": report["protocol"],
    "seed": report["seed"],
    "eligible_case_count": report["eligible_case_count"],
    "eligible_target_count": report["eligible_target_count"],
    "trigger_modes": report["trigger_modes"],
    "registered_variant_count": sum(
        int(row["eligible_target_count"]) for row in report["aggregate"]
    ),
    "outcome_report": str(final_dir / "frozen_validation_report.json"),
    "optimization_audit": str(final_dir / "optimization_audit.json"),
}
temporary = final_dir / "PIPELINE_COMPLETE.json.tmp"
temporary.write_text(json.dumps(marker, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
temporary.replace(final_dir / "PIPELINE_COMPLETE.json")
print(json.dumps(marker, indent=2, ensure_ascii=False))
PY
