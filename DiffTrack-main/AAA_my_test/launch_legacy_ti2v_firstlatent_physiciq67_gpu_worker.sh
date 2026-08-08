#!/usr/bin/env bash
set -euo pipefail

worker_id=${1:?usage: $0 WORKER_ID PHYSICAL_GPU_ID}
physical_gpu_id=${2:?usage: $0 WORKER_ID PHYSICAL_GPU_ID}
repo_root=/home/gaoya/Code_Video/DiffTrack-main
region_python=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
pck_python=/data/gaoya/miniconda3/envs/wan/bin/python
cache_root=/data/gaoya/agent-data/cache/wan22_ti2v_legacy_firstlatent_physiciq67_regions_704x1280
task_file=/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/missing_tasks.jsonl

if [[ "$worker_id" != "0" && "$worker_id" != "1" ]]; then
    echo "worker_id must be 0 or 1" >&2
    exit 2
fi
if [[ "$physical_gpu_id" != "6" && "$physical_gpu_id" != "7" ]]; then
    echo "physical_gpu_id must be 6 or 7" >&2
    exit 2
fi
if [[ ! -s "$task_file" ]]; then
    echo "missing task file: $task_file" >&2
    exit 2
fi

cd "$repo_root"
export CUDA_VISIBLE_DEVICES="$physical_gpu_id"
export PYTHONPATH="$repo_root:/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Grounded-SAM-2-main${PYTHONPATH:+:$PYTHONPATH}"

"$region_python" AAA_my_test/precompute_legacy_ti2v_firstlatent_physiciq67_regions.py \
    --worker-id "$worker_id" \
    --num-workers 2 \
    --device cuda:0

while true; do
    complete_count=$(find "$cache_root" -mindepth 2 -maxdepth 2 -name complete.json -type f | wc -l)
    if [[ "$complete_count" -ge 67 ]]; then
        break
    fi
    echo "region barrier: $complete_count/67 complete"
    sleep 30
done

"$pck_python" AAA_my_test/run_legacy_ti2v_firstlatent_physiciq67_pck_task_worker.py \
    --task-jsonl "$task_file" \
    --worker-id "$worker_id" \
    --num-workers 2 \
    --device cuda:0
