#!/usr/bin/env bash
set -u

GPU="${1:?Usage: $0 GPU_ID}"
ROOT="/data/gaoya/agent-data/outputs/attention_lora_neighbor_ranking_seed_sweep_metrics_case001460"
BENCH="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/bench.sh"
HERE="/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test"

while true; do
  /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
    "${HERE}/prepare_attention_lora_neighbor_ranking_benchmark.py" \
    >> "${ROOT}/prepare.log" 2>&1 || { sleep 60; continue; }
  methods="${ROOT}/bench_methods_gpu${GPU}.txt"
  if find "${ROOT}/methods" -name 'seed_*.json' -print -quit 2>/dev/null | grep -q .; then
    mkdir -p "${ROOT}/logs" "${ROOT}/status" "${ROOT}/summaries/gpu${GPU}"
    BENCH_RUN_METRICS=1 \
    BENCH_CUDA_VISIBLE_DEVICES="${GPU}" \
    BENCH_INPUT_JSON_ALLOWLIST="${ROOT}/input_json_allowlist.txt" \
    BENCH_RESULT_DIR="${ROOT}/summaries/gpu${GPU}" \
    CUDA_VISIBLE_DEVICES="${GPU}" \
      bash "${BENCH}" "${methods}" >> "${ROOT}/logs/gpu${GPU}.log" 2>&1 || true
  fi
  sleep 60
done
