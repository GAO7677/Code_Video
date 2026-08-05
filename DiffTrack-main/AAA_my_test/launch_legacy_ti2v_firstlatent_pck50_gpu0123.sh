#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/gaoya/Code_Video/DiffTrack-main
PY=/data/gaoya/miniconda3/envs/wan/bin/python
OUT=/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_pck50
CACHE=/data/gaoya/agent-data/cache/wan22_ti2v_legacy_firstlatent_regions_704x1280
SELF="$ROOT/AAA_my_test/launch_legacy_ti2v_firstlatent_pck50_gpu0123.sh"

wait_old_queues() {
  local gpu=$1
  for session in "step_align_g${gpu}" "official_ti2v_firstframe_g${gpu}"; do
    while tmux has-session -t "$session" 2>/dev/null; do
      echo "[$(date -Is)] GPU${gpu} waiting for tmux ${session}"
      sleep 60
    done
  done
}

wait_gpu_free() {
  local gpu=$1
  while true; do
    local used
    used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    if [[ "$used" -lt 2500 ]]; then
      return
    fi
    echo "[$(date -Is)] GPU${gpu} memory=${used}MiB, waiting"
    sleep 60
  done
}

mode=${1:-launch}
if [[ "$mode" == "regions" ]]; then
  wait_old_queues 1
  wait_gpu_free 1
  cd "$ROOT"
  export CUDA_VISIBLE_DEVICES=1
  exec "$PY" AAA_my_test/precompute_legacy_ti2v_firstlatent_regions.py --device cuda:0
fi
if [[ "$mode" == "pck" ]]; then
  gpu=$2
  wait_old_queues "$gpu"
  while [[ ! -f "$CACHE/all_complete.json" ]]; do sleep 30; done
  wait_gpu_free "$gpu"
  cd "$ROOT"
  export CUDA_VISIBLE_DEVICES="$gpu"
  exec "$PY" AAA_my_test/run_legacy_ti2v_firstlatent_pck_worker.py \
    --worker-id "$gpu" --num-workers 4 --device cuda:0
fi
if [[ "$mode" == "heatmap" ]]; then
  gpu=$2
  while [[ ! -f "$OUT/aggregate/final_top10.json" ]]; do sleep 60; done
  wait_gpu_free "$gpu"
  cd "$ROOT"
  export CUDA_VISIBLE_DEVICES="$gpu"
  exec "$PY" AAA_my_test/run_legacy_ti2v_firstlatent_top10_heatmaps_worker.py \
    --worker-id "$gpu" --num-workers 4
fi

mkdir -p "$OUT/logs"
tmux has-session -t legacy_ti2v_pck50_regions 2>/dev/null || \
  tmux new-session -d -s legacy_ti2v_pck50_regions \
  "bash '$SELF' regions 2>&1 | tee -a '$OUT/logs/regions.log'"
for gpu in 0 1 2 3; do
  tmux has-session -t "legacy_ti2v_pck50_g${gpu}" 2>/dev/null || \
    tmux new-session -d -s "legacy_ti2v_pck50_g${gpu}" \
    "bash '$SELF' pck '$gpu' 2>&1 | tee -a '$OUT/logs/pck_gpu${gpu}.log'"
done
tmux has-session -t legacy_ti2v_pck50_aggregate 2>/dev/null || \
  tmux new-session -d -s legacy_ti2v_pck50_aggregate \
  "cd '$ROOT' && '$PY' AAA_my_test/aggregate_legacy_ti2v_firstlatent_pck50.py --watch --interval 60 2>&1 | tee -a '$OUT/logs/aggregate.log'"
for gpu in 0 1 2 3; do
  tmux has-session -t "legacy_ti2v_pck50_heatmap_g${gpu}" 2>/dev/null || \
    tmux new-session -d -s "legacy_ti2v_pck50_heatmap_g${gpu}" \
    "bash '$SELF' heatmap '$gpu' 2>&1 | tee -a '$OUT/logs/heatmap_gpu${gpu}.log'"
done

printf 'Queued legacy TI2V first-latent PCK50 on GPU0,1,2,3.\n'
