#!/usr/bin/env bash
set -euo pipefail

SESSION="${TMUX_SESSION:-pck_extreme100_all720}"
SCRIPT_DIR="/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test"
GEN_LOG_ROOT="/data/gaoya/agent-data/outputs/attention_probability_noise_complete_logs"
BENCH_ROOT="/data/gaoya/agent-data/outputs/attention_probability_noise_metrics_test5"
PYTHON_BIN="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"

mkdir -p "${BENCH_ROOT}/status" "${BENCH_ROOT}/logs"
rm -f "${BENCH_ROOT}/PREPARE_FAILED"

tmux has-session -t "${SESSION}" 2>/dev/null || tmux new-session -d -s "${SESSION}" -n bootstrap

replace_window() {
  local name="$1"
  shift
  tmux kill-window -t "${SESSION}:${name}" 2>/dev/null || true
  tmux new-window -d -t "${SESSION}" -n "${name}" "$*"
}

prepare_command=$(cat <<EOF
set -euo pipefail
echo '[metrics] waiting for generation markers from GPUs 0,1,2,3,5'
while true; do
  ready=1
  for gpu in 0 1 2 3 5; do
    log='${GEN_LOG_ROOT}/gpu'\${gpu}'.log'
    if [[ ! -f \"\${log}\" ]] || ! grep -q \"MATRIX_GPU\${gpu}_COMPLETE\" \"\${log}\"; then
      ready=0
    fi
  done
  [[ \${ready} -eq 1 ]] && break
  date -u '+[metrics] %FT%TZ generation is still running'
  sleep 60
done
echo '[metrics] all generation queues complete; preparing 27 x 20 benchmark tree'
if '${PYTHON_BIN}' '${SCRIPT_DIR}/prepare_attention_noise_benchmark.py' 2>&1 | tee '${BENCH_ROOT}/logs/prepare.log'; then
  echo '[metrics] preparation complete'
else
  touch '${BENCH_ROOT}/PREPARE_FAILED'
  echo '[metrics] preparation failed; see prepare.log'
  exit 1
fi
exec bash
EOF
)
replace_window metrics_prepare "bash -lc $(printf '%q' "${prepare_command}")"

replace_window metrics_g0 \
  "bash -lc 'exec ${SCRIPT_DIR}/run_attention_noise_metric_worker.sh 0 physics_iq_with_context physics_iq_without_context pmf_with_context pmf_without_context'"
replace_window metrics_g1 \
  "bash -lc 'exec ${SCRIPT_DIR}/run_attention_noise_metric_worker.sh 1 wmreward vbench_subject_consistency vbench_background_consistency'"
replace_window metrics_g2 \
  "bash -lc 'exec ${SCRIPT_DIR}/run_attention_noise_metric_worker.sh 2 vbench_temporal_flickering vbench_motion_smoothness vbench_dynamic_degree'"
replace_window metrics_g3 \
  "bash -lc 'exec ${SCRIPT_DIR}/run_attention_noise_metric_worker.sh 3 vbench_aesthetic_quality vbench_imaging_quality videophy2'"
replace_window metrics_g5 \
  "bash -lc 'exec ${SCRIPT_DIR}/run_attention_noise_metric_worker.sh 5 cosmos_reason1'"

echo "Scheduled deferred preparation and all 14 metrics in tmux session ${SESSION}."
