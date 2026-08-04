#!/usr/bin/env bash
set -euo pipefail

HERE="/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test"
CURRENT_RUNNER="${HERE}/run_attention_lora_seed_sweep_gpu.sh"
RANKING_RUNNER="${HERE}/run_attention_lora_neighbor_ranking_gpu.sh"
CURRENT="/data/gaoya/agent-data/outputs/attention_lora_seed_sweep_case001460"
CURRENT_CASE_LIST="/data/gaoya/agent-data/outputs/attention_probability_mono_scale_steps40_frames49_case001460/case_list.txt"
RANKING_ROOT="/data/gaoya/agent-data/outputs/attention_lora_neighbor_ranking_seed_sweep_case001460"

mkdir -p "${RANKING_ROOT}/logs"
cp -n "${CURRENT}/seeds.txt" "${RANKING_ROOT}/seeds.txt"
exec > >(tee -a "${RANKING_ROOT}/logs/queue.log") 2>&1

echo "[$(date -Is)] waiting for current legacy GPU 0-5 shards"
while true; do
  ready=1
  for gpu in 0 1 2 3 4 5; do
    [[ -f "${CURRENT}/logs/gpu${gpu}.complete" ]] || ready=0
  done
  (( ready )) && break
  sleep 60
done

echo "[$(date -Is)] start current six-way rescue"
pids=()
for gpu in 0 1 2 3 4 5; do
  (while ! ATTENTION_SEED_SWEEP_PILOT_SEED=-1 \
      bash "${CURRENT_RUNNER}" "${gpu}" 6; do sleep 120; done) \
      >> "${RANKING_ROOT}/logs/current_rescue_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "${pid}"; done
printf 'completed=%s\n' "$(date -u +%FT%TZ)" > "${CURRENT}/GENERATION_COMPLETE"

echo "[$(date -Is)] start neighbor-ranking 50-seed matrix"
pids=()
for gpu in 0 1 2 3 4 5; do
  (while ! bash "${RANKING_RUNNER}" "${gpu}" 6; do sleep 120; done) \
      >> "${RANKING_ROOT}/logs/launcher_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "${pid}"; done
printf 'completed=%s\n' "$(date -u +%FT%TZ)" > "${RANKING_ROOT}/GENERATION_COMPLETE"
echo "[$(date -Is)] neighbor-ranking matrix complete"
