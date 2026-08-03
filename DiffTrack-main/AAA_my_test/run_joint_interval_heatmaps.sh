#!/usr/bin/env bash
set -uo pipefail

PROJECT=/home/gaoya/Code_Video/DiffTrack-main
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
WORKER="$PROJECT/AAA_my_test/capture_stable_heads_alltoken_qk_worker.py"
OUTPUT=${OUTPUT:-/data/gaoya/agent-data/outputs/three_model_joint_interval_samples_alltoken_qk_case001}
COMBINATIONS=$(<"$OUTPUT/combinations.txt")
mapfile -t LAYERS < <(tr ',' '\n' <<<"$COMBINATIONS" | cut -d: -f1 | sort -n -u)
mkdir -p "$OUTPUT/logs"

run_model() {
    local model=$1
    local gpu=$2
    local extra=()
    if [[ "$model" != gt ]]; then
        extra+=(--analysis-no-cotracker)
    fi
    echo "[$(date -Is)] start model=$model gpu=$gpu"
    CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ALLTOKEN_COMBINATIONS="$COMBINATIONS" \
    PYTHONPATH="$PROJECT:/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main:/home/gaoya/Code_Video/Code_data/Code_train/train_0419" \
    "$PYTHON" "$WORKER" \
        --model-kind "$model" \
        --worker-id 0 \
        --num-workers 1 \
        --output-dir "$OUTPUT/$model" \
        --sampling-steps 40 \
        --analysis-matching-mode q_to_k \
        --analysis-layers "${LAYERS[@]}" \
        --analysis-step-indices 39 \
        --analysis-no-hidden \
        --analysis-no-video \
        --case-keys case_001_ball_roll \
        --overwrite \
        "${extra[@]}" \
        >"$OUTPUT/logs/${model}_gpu${gpu}.log" 2>&1
    local status=$?
    echo "[$(date -Is)] finish model=$model gpu=$gpu status=$status"
    return "$status"
}

run_model gt 2 &
pid_gt=$!
run_model lora 5 &
pid_lora=$!

finished_pid=
wait -n -p finished_pid "$pid_gt" "$pid_lora"
first_status=$?
if [[ "$finished_pid" == "$pid_gt" ]]; then
    freed_gpu=2
    remaining_pid=$pid_lora
else
    freed_gpu=5
    remaining_pid=$pid_gt
fi

run_model baseline "$freed_gpu" &
pid_baseline=$!
wait "$remaining_pid"
remaining_status=$?
wait "$pid_baseline"
baseline_status=$?

if (( first_status != 0 || remaining_status != 0 || baseline_status != 0 )); then
    echo "one or more model captures failed; inspect $OUTPUT/logs" >&2
    exit 1
fi
echo "all interval heatmaps complete: $OUTPUT"
