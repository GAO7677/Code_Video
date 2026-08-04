#!/usr/bin/env bash
set -u

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 GPU_ID" >&2
  exit 2
fi

GPU="$1"
HERE="/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test"
QUEUE="${HERE}/attention_lora_multicase_queue.tsv"
SOURCE_BASE="/data/gaoya/agent-data/outputs/attention_lora_seed_sweep_multicase"
METRIC_BASE="/data/gaoya/agent-data/outputs/attention_lora_seed_sweep_metrics_multicase"
BENCH="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/bench.sh"

while true; do
  while IFS=$'\t' read -r case_key input_json; do
    [[ -n "${case_key}" ]] || continue
    source_root="${SOURCE_BASE}/${case_key}"
    bench_root="${METRIC_BASE}/${case_key}"
    [[ -s "${source_root}/seeds.txt" ]] || continue
    mkdir -p "${bench_root}/logs" "${bench_root}/status" "${bench_root}/summaries/gpu${GPU}"
    ATTENTION_SEED_SWEEP_SOURCE_ROOT="${source_root}" \
    ATTENTION_SEED_SWEEP_BENCH_ROOT="${bench_root}" \
    ATTENTION_SEED_SWEEP_INPUT_JSON="${input_json}" \
    ATTENTION_SEED_SWEEP_CASE_KEY="${case_key}" \
      /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
      "${HERE}/prepare_attention_lora_seed_sweep_benchmark.py" \
      >> "${bench_root}/logs/prepare.log" 2>&1
    methods="${bench_root}/bench_methods_gpu${GPU}.txt"
    allowlist="${bench_root}/input_json_allowlist.txt"
    if ! find "${bench_root}/methods" -name 'seed_*.json' -print -quit 2>/dev/null | grep -q .; then
      continue
    fi
    printf 'gpu=%s\nstarted=%s\n' "${GPU}" "$(date -u +%FT%TZ)" \
      > "${bench_root}/status/gpu${GPU}.running"
    BENCH_RUN_METRICS=1 \
    BENCH_CUDA_VISIBLE_DEVICES="${GPU}" \
    BENCH_INPUT_JSON_ALLOWLIST="${allowlist}" \
    BENCH_RESULT_DIR="${bench_root}/summaries/gpu${GPU}" \
    CUDA_VISIBLE_DEVICES="${GPU}" \
      bash "${BENCH}" "${methods}" >> "${bench_root}/logs/gpu${GPU}.log" 2>&1 || true
  done < "${QUEUE}"
  sleep 60
done
