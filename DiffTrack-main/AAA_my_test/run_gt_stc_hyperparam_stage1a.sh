#!/usr/bin/env bash
set -euo pipefail

repo=/home/gaoya/Code_Video/DiffTrack-main
python_bin=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
source_root=/data/gaoya/agent-data/outputs/wan_gt_spatiotemporal_correspondence_guidance/latest3350_top100_cotracker_sam2_v2
output_root=/data/gaoya/agent-data/outputs/wan_gt_stc_hyperparam_search/latest3350_top100_v1/stage1a_first10
target_map="$source_root/screening/seed_47326/baseline_eligibility.json"
case_name=0613pybullet_sample_001460_w002
target_name=object_A
read -r -a allowed_gpus <<< "${GT_STC_SEARCH_GPUS:-1 2}"
required_free_mib=47000
required_idle_checks=3
scales=(0.005 0.01 0.02 0.05)

for gpu_index in "${allowed_gpus[@]}"; do
    if [[ "$gpu_index" == "4" ]]; then
        echo "GPU 4 is forbidden by workspace policy" >&2
        exit 2
    fi
done

mkdir -p "$output_root/generations/$case_name/seed_47326" "$output_root/logs"
link_exact() {
    local source_path="$1" destination_path="$2"
    if [[ -L "$destination_path" ]]; then
        [[ "$(readlink -f "$destination_path")" == "$(readlink -f "$source_path")" ]] || {
            echo "unexpected symlink target: $destination_path" >&2
            exit 1
        }
    elif [[ -e "$destination_path" ]]; then
        echo "refusing to replace existing path: $destination_path" >&2
        exit 1
    else
        ln -s "$source_path" "$destination_path"
    fi
}
link_exact "$source_root/gt_tubes" "$output_root/gt_tubes"
link_exact \
    "$source_root/generations/$case_name/seed_47326/baseline" \
    "$output_root/generations/$case_name/seed_47326/baseline"

cd "$repo"
export PYTHONPATH="$repo":/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main:/home/gaoya/Code_Video/Code_data/Code_train/train_0419
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

wait_for_idle_gpu() {
    local gpu_index gpu_uuid free_mib process_count idle_checks
    while true; do
        for gpu_index in "${allowed_gpus[@]}"; do
            gpu_uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$gpu_index" | head -n 1 | tr -d ' ')
            idle_checks=0
            while (( idle_checks < required_idle_checks )); do
                free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_index" | head -n 1 | tr -d ' ')
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
                    sleep 15
                else
                    break
                fi
            done
        done
        echo "[wait] GPU1/2 are not stably idle with ${required_free_mib} MiB free; retrying" >&2
        sleep 30
    done
}

run_generate_scale() {
    local scale="$1" gpu_index status attempt=0 scale_log
    scale_log="$output_root/logs/generate_lambda_${scale//./p}.log"
    while true; do
        attempt=$((attempt + 1))
        gpu_index=$(wait_for_idle_gpu)
        echo "[stage1a] attempt=$attempt lambda=$scale window=0-9 modes=region,point,combined gpu=$gpu_index" \
            | tee -a "$scale_log" "$output_root/logs/generate.log"
        set +e
        CUDA_VISIBLE_DEVICES="$gpu_index" "$python_bin" -u \
            AAA_my_test/run_wan_gt_spatiotemporal_correspondence_guidance.py \
            --stage generate \
            --device cuda:0 \
            --seed 47326 \
            --output-root "$output_root" \
            --target-map "$target_map" \
            --case-keys "$case_name" \
            --no-baseline \
            --loss-modes region point combined \
            --guidance-scale "$scale" \
            --guidance-start 0 \
            --guidance-end 9 \
            2>&1 | tee -a "$scale_log" "$output_root/logs/generate.log"
        status=${PIPESTATUS[0]}
        set -e
        if (( status == 0 )); then
            last_gpu_index="$gpu_index"
            return 0
        fi
        if tail -n 160 "$scale_log" | rg -q 'torch\.OutOfMemoryError|CUDA out of memory'; then
            echo "[stage1a] OOM on GPU $gpu_index; completed variants are retained, retrying when a GPU is idle" \
                | tee -a "$scale_log" "$output_root/logs/generate.log"
            continue
        fi
        echo "[stage1a] non-OOM failure status=$status" | tee -a "$scale_log" "$output_root/logs/generate.log"
        return "$status"
    done
}

last_gpu_index=
for scale in "${scales[@]}"; do
    run_generate_scale "$scale"
done

if [[ -z "$last_gpu_index" ]]; then
    last_gpu_index=$(wait_for_idle_gpu)
fi
echo "[stage1a] CoTracker evaluation on GPU $last_gpu_index"
CUDA_VISIBLE_DEVICES="$last_gpu_index" "$python_bin" -u \
    AAA_my_test/run_wan_gt_spatiotemporal_correspondence_guidance.py \
    --stage evaluate \
    --device cuda:0 \
    --seed 47326 \
    --output-root "$output_root" \
    --target-map "$target_map" \
    --case-keys "$case_name" \
    2>&1 | tee -a "$output_root/logs/evaluate.log"

echo "[stage1a] offline GT MSE + trajectory ranking"
"$python_bin" -u AAA_my_test/evaluate_gt_stc_hyperparam_search.py \
    --output-root "$output_root" \
    --tube-root "$source_root/gt_tubes" \
    --case "$case_name" \
    --target "$target_name" \
    --seed 47326 \
    2>&1 | tee -a "$output_root/logs/ranking.log"

echo "[stage1a] complete: $output_root/STAGE1A_RANKING.md"
