#!/usr/bin/env bash
set -euo pipefail

HERE="/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test"
RUNNER="${HERE}/run_attention_lora_seed_sweep_gpu.sh"
QUEUE="${HERE}/attention_lora_multicase_queue.tsv"
BASE="/data/gaoya/agent-data/outputs/attention_lora_seed_sweep_multicase"
CURRENT="/data/gaoya/agent-data/outputs/attention_lora_seed_sweep_case001460"
CURRENT_CASE_LIST="/data/gaoya/agent-data/outputs/attention_probability_mono_scale_steps40_frames49_case001460/case_list.txt"
CURRENT_CASE_KEY="0613pybullet_sample_001460_w002"

mkdir -p "${BASE}/logs"
exec > >(tee -a "${BASE}/logs/queue.log") 2>&1

run_six_way() {
  local root="$1" case_list="$2" case_key="$3" label="$4"
  local pids=()
  mkdir -p "${root}/logs"
  echo "[$(date -Is)] start ${label}"
  for gpu in 0 1 2 3 4 5; do
    ATTENTION_SEED_SWEEP_ROOT="${root}" \
    ATTENTION_SEED_SWEEP_CASE_LIST="${case_list}" \
    ATTENTION_SEED_SWEEP_CASE_KEY="${case_key}" \
    ATTENTION_SEED_SWEEP_PILOT_SEED=-1 \
      bash "${RUNNER}" "${gpu}" 6 \
      >> "${root}/logs/queue_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  if (( failed )); then
    echo "[$(date -Is)] ${label} failed; retrying incomplete work"
    run_six_way "${root}" "${case_list}" "${case_key}" "${label} retry"
    return
  fi
  printf 'case=%s\ncompleted=%s\n' "${case_key}" "$(date -u +%FT%TZ)" \
    > "${root}/GENERATION_COMPLETE"
  echo "[$(date -Is)] complete ${label}"
}

echo "[$(date -Is)] waiting for current GPU 0-5 legacy shards"
while true; do
  ready=1
  for gpu in 0 1 2 3 4 5; do
    [[ -f "${CURRENT}/logs/gpu${gpu}.complete" ]] || ready=0
  done
  (( ready )) && break
  sleep 60
done

# The original launch used NUM_GPUS=8 while GPU 6/7 were disabled. A safe
# six-way resume fills the omitted shards and skips every completed profile.
run_six_way "${CURRENT}" "${CURRENT_CASE_LIST}" "${CURRENT_CASE_KEY}" "current-case six-way rescue"

while IFS=$'\t' read -r case_key input_json; do
  [[ -n "${case_key}" ]] || continue
  root="${BASE}/${case_key}"
  mkdir -p "${root}"
  printf '%s\n' "${input_json}" > "${root}/case_list.txt"
  if [[ ! -s "${root}/seeds.txt" ]]; then
    /home/gaoya/miniconda3/envs/wan-cu128/bin/python - "${root}/seeds.txt" <<'PY'
import random
import sys
from pathlib import Path

path = Path(sys.argv[1])
seeds = random.SystemRandom().sample(range(100001), 50)
path.write_text("".join(f"{seed}\n" for seed in seeds), encoding="utf-8")
PY
  fi
  if [[ -f "${root}/GENERATION_COMPLETE" ]]; then
    echo "[$(date -Is)] skip complete ${case_key}"
    continue
  fi
  run_six_way "${root}" "${root}/case_list.txt" "${case_key}" "${case_key}"
done < "${QUEUE}"

printf 'completed=%s\n' "$(date -u +%FT%TZ)" > "${BASE}/QUEUE_COMPLETE"
echo "[$(date -Is)] all queued cases complete"
