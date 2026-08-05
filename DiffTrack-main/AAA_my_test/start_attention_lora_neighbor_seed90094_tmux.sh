#!/usr/bin/env bash
set -euo pipefail

HERE="/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test"
ROOT="/data/gaoya/agent-data/outputs/attention_lora_neighbor_ranking_seed090094_case001460"
METRICS="/data/gaoya/agent-data/outputs/attention_lora_neighbor_ranking_seed090094_metrics_case001460"
GEN_SESSION="attention_lora_neighbor_seed90094"
METRIC_SESSION="attention_lora_neighbor_seed90094_metrics"

mkdir -p "${ROOT}/logs" "${METRICS}"
printf '90094\n' > "${ROOT}/seeds.txt"
for gpu in 0 1 2 3 4; do
  command="ATTENTION_NEIGHBOR_RANKING_ROOT=${ROOT} ATTENTION_RANKING_FIXED_SEED=90094 bash ${HERE}/run_attention_lora_neighbor_ranking_gpu.sh ${gpu} 5; exec bash"
  if [[ "${gpu}" == 0 ]]; then
    tmux new-session -d -s "${GEN_SESSION}" -n "gpu${gpu}" "${command}"
  else
    tmux new-window -t "${GEN_SESSION}" -n "gpu${gpu}" "${command}"
  fi
done
for gpu in 0 1 2 3; do
  command="ATTENTION_NEIGHBOR_RANKING_SOURCE_ROOT=${ROOT} ATTENTION_NEIGHBOR_RANKING_BENCH_ROOT=${METRICS} bash ${HERE}/run_attention_lora_neighbor_ranking_metric_worker.sh ${gpu}; exec bash"
  if [[ "${gpu}" == 0 ]]; then
    tmux new-session -d -s "${METRIC_SESSION}" -n "gpu${gpu}" "${command}"
  else
    tmux new-window -t "${METRIC_SESSION}" -n "gpu${gpu}" "${command}"
  fi
done
echo "started ${GEN_SESSION} on GPU 0-4 and ${METRIC_SESSION} on GPU 0-3"
