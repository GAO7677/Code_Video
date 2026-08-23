#!/usr/bin/env bash
set -u

BASE="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
CONFIG="$BASE/xssc_lora_three_train_watch_config_with_t_head.json"
LOG="/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_three_run_watch/logs/vbench_model_queue_8844.log"

mkdir -p "$(dirname "$LOG")"
while true; do
    printf '[%s] vbench supervisor starting queue workers_per_gpu=3\n' "$(date -u +%FT%TZ)" >> "$LOG"
    set +e
    env PYTHONNOUSERSITE=1 "$PYTHON" "$BASE/xssc_lora_vbench_queue.py" \
        --config "$CONFIG" \
        --gpus 0,1,2,3,5,6,7 \
        --workers-per-gpu 3 \
        --poll-seconds 30 \
        --idle-rounds 3 >> "$LOG" 2>&1
    rc=$?
    set -u
    printf '[%s] vbench supervisor queue_exit rc=%s\n' "$(date -u +%FT%TZ)" "$rc" >> "$LOG"
    if tail -n 20 "$LOG" | rg -q 'vbench queue drained'; then
        printf '[%s] vbench supervisor drained\n' "$(date -u +%FT%TZ)" >> "$LOG"
        exit 0
    fi
    sleep 10
done
