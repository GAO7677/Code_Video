#!/usr/bin/env bash
set -euo pipefail

SESSION="attention_lora_test5_2seed_steps10"
HERE="/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test"
WORKER="${HERE}/run_attention_lora_test5_2seed_10step_gpu.sh"
QUEUE="${HERE}/attention_lora_test5_20case_10seed_queue.tsv"
ROOT40="/data/gaoya/agent-data/outputs/attention_lora_seed_sweep_test5_20case_10seed"
ROOT10="/data/gaoya/agent-data/outputs/attention_lora_seed_sweep_test5_20case_2seed_steps10"

mkdir -p "${ROOT10}/logs"
while IFS=$'\t' read -r case_key input_json; do
  mkdir -p "${ROOT40}/cases/${case_key}" "${ROOT10}/cases/${case_key}"
  printf '90094\n35075\n' > "${ROOT40}/cases/${case_key}/seeds.txt"
  printf '90094\n35075\n' > "${ROOT10}/cases/${case_key}/seeds.txt"
  printf '%s\n' "${input_json}" > "${ROOT10}/cases/${case_key}/case_list.txt"
done < "${QUEUE}"
cat > "${ROOT10}/experiment_manifest.json" <<'JSON'
{
  "model": "Wan+LoRA",
  "num_inference_steps": 10,
  "num_frames": 49,
  "seeds": [90094, 35075],
  "profiles": ["alpha090", "alpha150", "zero", "uniform", "temporal_causal", "strict_past", "strict_future", "head_output_zero"],
  "groups": ["top100", "bottom100"],
  "comparison_root_40step": "/data/gaoya/agent-data/outputs/attention_lora_seed_sweep_test5_20case_10seed"
}
JSON

chmod +x "${WORKER}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi
tmux new-session -d -s "${SESSION}" -n gpu5 "bash '${WORKER}' 5 0 3; exec bash"
tmux new-window -t "${SESSION}" -n gpu6 "bash '${WORKER}' 6 1 3; exec bash"
tmux new-window -t "${SESSION}" -n gpu7 "bash '${WORKER}' 7 2 3; exec bash"
tmux select-window -t "${SESSION}:gpu5"
echo "started ${SESSION} on GPU5/6/7; GPU5 waits while memory is above the configured limit"

