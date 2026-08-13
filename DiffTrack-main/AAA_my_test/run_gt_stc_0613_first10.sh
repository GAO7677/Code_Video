#!/usr/bin/env bash
set -euo pipefail

repo=/home/gaoya/Code_Video/DiffTrack-main
python_bin=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
source_root=/data/gaoya/agent-data/outputs/wan_gt_spatiotemporal_correspondence_guidance/latest3350_top100_cotracker_sam2_v2
output_root=/data/gaoya/agent-data/outputs/wan_gt_spatiotemporal_correspondence_guidance/latest3350_top100_first10_0613_v1
target_map="$source_root/screening/seed_47326/baseline_eligibility.json"
log_root="$output_root/logs"
cases=(
    0613pybullet_sample_000336_w001
    0613pybullet_sample_001455_w000
    0613pybullet_sample_001460_w002
)
allowed_gpus=(2)
required_free_mib=47000
required_idle_checks=3
last_gpu_index=

mkdir -p "$output_root" "$log_root" "$output_root/generations"

link_exact() {
    local source_path="$1"
    local destination_path="$2"
    if [[ -L "$destination_path" ]]; then
        [[ "$(readlink -f "$destination_path")" == "$(readlink -f "$source_path")" ]] || {
            echo "unexpected symlink target: $destination_path" >&2
            return 1
        }
        return 0
    fi
    [[ ! -e "$destination_path" ]] || {
        echo "refusing to replace existing path: $destination_path" >&2
        return 1
    }
    ln -s "$source_path" "$destination_path"
}

link_exact "$source_root/gt_tubes" "$output_root/gt_tubes"
for case_name in "${cases[@]}"; do
    seed_root="$output_root/generations/$case_name/seed_47326"
    mkdir -p "$seed_root"
    link_exact \
        "$source_root/generations/$case_name/seed_47326/baseline" \
        "$seed_root/baseline"
done

wait_for_idle_gpu() {
    local gpu_index
    local gpu_uuid
    local free_mib
    local process_count
    local idle_checks
    while true; do
        for gpu_index in "${allowed_gpus[@]}"; do
            gpu_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$gpu_index" | head -n 1 | tr -d ' ')
            idle_checks=0
            while (( idle_checks < required_idle_checks )); do
                free_mib=$(nvidia-smi \
                    --query-gpu=memory.free \
                    --format=csv,noheader,nounits \
                    -i "$gpu_index" | head -n 1 | tr -d ' ')
                process_count=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null \
                    | awk -v uuid="$gpu_uuid" '$1 == uuid {count += 1} END {print count + 0}')
                if [[ "$free_mib" =~ ^[0-9]+$ ]] \
                    && (( free_mib >= required_free_mib )) \
                    && (( process_count == 0 )); then
                    idle_checks=$((idle_checks + 1))
                    if (( idle_checks == required_idle_checks )); then
                        printf '%s\n' "$gpu_index"
                        return 0
                    fi
                    sleep 30
                else
                    break
                fi
            done
        done
        echo "[wait] no allowed GPU is process-free with ${required_free_mib} MiB free; retrying in 30s" >&2
        sleep 30
    done
}

run_stage() {
    local stage="$1"
    local log_path="$log_root/${stage}.log"
    local attempt=0
    local gpu_index
    local status
    while true; do
        attempt=$((attempt + 1))
        gpu_index=$(wait_for_idle_gpu)
        echo "[$stage] attempt=$attempt physical_gpu=$gpu_index" | tee -a "$log_path"
        set +e
        CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" -u \
            AAA_my_test/run_wan_gt_spatiotemporal_correspondence_guidance.py \
            --stage "$stage" \
            --device cuda:0 \
            --seed 47326 \
            --output-root "$output_root" \
            --target-map "$target_map" \
            --case-keys "${cases[@]}" \
            --no-baseline \
            --loss-modes region point combined \
            --guidance-scale 0.1 \
            --guidance-start 0 \
            --guidance-end 9 \
            2>&1 | tee -a "$log_path"
        status=${PIPESTATUS[0]}
        set -e
        if (( status == 0 )); then
            last_gpu_index="$gpu_index"
            return 0
        fi
        if tail -n 300 "$log_path" | rg -q 'torch\.OutOfMemoryError|CUDA out of memory'; then
            echo "[$stage] CUDA OOM; completed variants are retained, waiting to retry" | tee -a "$log_path"
            continue
        fi
        echo "[$stage] non-OOM failure status=$status" | tee -a "$log_path"
        return "$status"
    done
}

cd "$repo"
export PYTHONPATH="$repo":/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main:/home/gaoya/Code_Video/Code_data/Code_train/train_0419
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

guided_complete=$(find "$output_root/generations" -type f -path '*__lambda0p1/complete.json' -size +0c 2>/dev/null | wc -l)
if (( guided_complete < 9 )); then
    run_stage generate
else
    echo "[generate] skip: guided videos already complete ($guided_complete/9)"
fi

metric_complete=$(find "$output_root/generations" -type f -path '*__lambda0p1/trajectory_metrics.json' -size +0c 2>/dev/null | wc -l)
if (( metric_complete < 9 )); then
    run_stage evaluate
else
    echo "[evaluate] skip: trajectory metrics already complete ($metric_complete/9)"
fi

if [[ -z "$last_gpu_index" ]]; then
    last_gpu_index=$(wait_for_idle_gpu)
fi

for case_name in "${cases[@]}"; do
    echo "[overlay] case=$case_name physical_gpu=$last_gpu_index" | tee -a "$log_root/overlays.log"
    CUDA_VISIBLE_DEVICES="$last_gpu_index" "$python_bin" -u \
        AAA_my_test/render_gt_stc_trajectory_overlays.py \
        --output-root "$output_root" \
        --target-map "$target_map" \
        --primary-only \
        --case "$case_name" \
        --device cuda:0 \
        2>&1 | tee -a "$log_root/overlays.log"
done

"$python_bin" - "$output_root" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
cases = (
    "0613pybullet_sample_000336_w001",
    "0613pybullet_sample_001455_w000",
    "0613pybullet_sample_001460_w002",
)
guided = []
for case in cases:
    seed_root = root / "generations" / case / "seed_47326"
    for manifest_path in sorted(seed_root.glob("*__lambda0p1/manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metrics_path = manifest_path.with_name("trajectory_metrics.json")
        assert manifest["guidance_step_range_inclusive"] == [0, 9], manifest_path
        assert len(manifest["audit"]) == 40, manifest_path
        assert sum(bool(row["guided"]) for row in manifest["audit"]) == 10, manifest_path
        assert metrics_path.is_file(), metrics_path
        overlay_path = (
            root
            / "trajectory_overlays"
            / case
            / "seed_47326"
            / f"{manifest_path.parent.name}__{manifest['target']}.mp4"
        )
        assert overlay_path.is_file() and overlay_path.stat().st_size > 0, overlay_path
        guided.append(str(manifest_path.parent))
    assert (seed_root / "comparison_to_baseline.json").is_file(), seed_root
assert len(guided) == 9, len(guided)
marker = {
    "protocol": "wan_gt_stc_latest3350_top100_first10_0613_v1",
    "seed": 47326,
    "cases": list(cases),
    "guided_variants_complete": len(guided),
    "guidance_step_range_inclusive": [0, 9],
    "guided_denoising_step_count": 10,
    "variants": guided,
}
path = root / "PIPELINE_COMPLETE.json"
path.write_text(json.dumps(marker, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(marker, indent=2, ensure_ascii=False))
PY
