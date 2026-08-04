#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-pck_extreme100_all720}"
ROOT="/home/gaoya/Code_Video/DiffTrack-main"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
WORKER="${ROOT}/AAA_my_test/run_pck_extreme_head_zero_ablation_worker.py"
INPUT="/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt"
OUTPUT="/data/gaoya/agent-data/outputs/pck_extreme100_all720_head_zero_ablation_test5"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi
mkdir -p "${OUTPUT}/logs"

worker_cmd() {
  local gpu="$1" shard="$2" first="$3" second="$4"
  printf 'set -euo pipefail; cd %q; ' "${ROOT}"
  for model in "${first}" "${second}"; do
    printf 'CUDA_VISIBLE_DEVICES=%q PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True %q %q --model %q --input-json-list %q --output-root %q --shard-index %q --num-shards 2 --device cuda:0 --ranking-pool all720 --extreme-count 100 2>&1 | tee -a %q; ' \
      "${gpu}" "${PYTHON}" "${WORKER}" "${model}" "${INPUT}" "${OUTPUT}" "${shard}" \
      "${OUTPUT}/logs/${model}_shard${shard}_gpu${gpu}.log"
  done
  printf 'date -u +%%FT%%TZ > %q; exec bash' "${OUTPUT}/GPU${gpu}_SHARD${shard}_COMPLETE.txt"
}

cmd0="$(worker_cmd 0 0 baseline lora)"
cmd2="$(worker_cmd 2 1 lora baseline)"
status_cmd="while true; do clear; date -u; n=\$(find '${OUTPUT}' -type f -name '*.mp4' | wc -l); echo \"Top100/Bottom100 videos: \${n}/440\"; if test \"\${n}\" -ge 440; then echo COMPLETE; exec bash; fi; sleep 10; done"

tmux new-session -d -s "${SESSION}" -n gpu0 "bash -lc $(printf '%q' "${cmd0}")"
tmux set-option -t "${SESSION}" remain-on-exit on
tmux new-window -t "${SESSION}" -n gpu2 "bash -lc $(printf '%q' "${cmd2}")"
tmux new-window -t "${SESSION}" -n status "bash -lc $(printf '%q' "${status_cmd}")"
tmux select-window -t "${SESSION}:status"
echo "started tmux session ${SESSION}"
echo "attach: tmux attach -t ${SESSION}"
