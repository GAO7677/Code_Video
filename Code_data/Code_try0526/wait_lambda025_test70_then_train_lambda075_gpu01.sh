#!/usr/bin/env bash
set -euo pipefail

STATE=/data/gaoya/agent-data/physv_v2v_0819/logs/test70_lineage_watcher/state.json
LOG=/data/gaoya/agent-data/outputs/context_noise_interpolation_20260923/lambda075-gpu01-handoff.log
LAUNCHER=/home/gaoya/Code_Video/Code_data/Code_try0526/run_lambda075_gpu01_interrupt.sh
PYTHON=/data/gaoya/agent-data/envs/physrvg-full-sa/bin/python
WATCHER=/home/gaoya/code_V2V_baselines/PhysRVG-main/scripts_mytrain/evaluation/test70/watchers/watch_lineage_test70.py
MODEL_KEY=full_sa_physrvg_raw_rl_lora_context_noise_lambda025_initial0907_generic_full_2175_clean_context_20260923
RUN=lambda075-initial0907-seed42-gpu01-interrupt
RUN_DIR=/data/gaoya/agent-data/checkpoints/generic2175_initial0907_context_noise_20260923/$RUN
STEPS=500,1000,1500,2000,2500

mkdir -p "$(dirname "$LOG")"
exec >> "$LOG" 2>&1

log() {
  echo "[$(date -u +%FT%TZ)] $*"
}

test70_complete() {
  # Direct backfill workers update metric artifacts; refresh the aggregate state before reading it.
  "$PYTHON" "$WATCHER" --once >/dev/null 2>>"$LOG" || true
  [[ -s "$STATE" ]] || return 1
  ruby -rjson -e '
    state, key, expected = ARGV
    wanted = expected.split(",").map(&:to_i).sort
    rows = (JSON.parse(File.read(state))["tasks"] || []).select { |row| row["model_key"] == key }
    exit 1 unless rows.map { |row| row["step"].to_i }.sort == wanted
    complete = rows.all? do |row|
      progress = row["progress"] || {}
      groups = %w[cpu gpu fvd rigidbench]
      row["generation"].to_i == row["total"].to_i &&
        row["status"] == "complete" &&
        groups.all? do |group|
          item = progress[group] || {}
          item["total"].to_i.positive? && item["done"].to_i == item["total"].to_i
        end
    end
    exit(complete ? 0 : 1)
  ' "$STATE" "$MODEL_KEY" "$STEPS"
}

gpu01_compute_processes() {
  local uuid0 uuid1
  uuid0=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1 == 0 {print $2}')
  uuid1=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1 == 1 {print $2}')
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader 2>/dev/null \
    | awk -F', ' -v u0="$uuid0" -v u1="$uuid1" '$1 == u0 || $1 == u1'
}

if pgrep -af -- "--run_name $RUN" >/dev/null; then
  log "training already active; nothing to launch: $RUN"
  exit 0
fi
if [[ -e "$RUN_DIR/resolved_config.json" ]]; then
  log "run already initialized; refusing duplicate launch: $RUN_DIR"
  exit 0
fi

log "waiting for all five lambda=0.25 Test70 generations and metric groups"
while ! test70_complete; do
  sleep 60
done
log "watcher reports all lambda=0.25 Test70 tasks complete"

while pgrep -f 'run_lambda025_test70_gpu_queue.sh [01]' >/dev/null; do
  log "waiting for GPU0/1 Test70 queue processes to exit"
  sleep 30
done

while [[ -n "$(gpu01_compute_processes)" ]]; do
  log "waiting for GPU0/1 compute processes to release the devices"
  gpu01_compute_processes
  sleep 30
done

log "GPU0/1 are free; starting lambda=0.75 training"
exec bash "$LAUNCHER"
