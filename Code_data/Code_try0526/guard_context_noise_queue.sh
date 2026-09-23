#!/usr/bin/env bash
set -euo pipefail

LOG=/data/gaoya/agent-data/outputs/context_noise_interpolation_20260923/launch.log
SESSION=context-noise-interpolation

while ! pgrep -af 'run_name lambda050-initial0907-seed42' >/dev/null; do
    sleep 60
done
echo "[guard] lambda050 detected; waiting for it to finish"
while pgrep -af 'run_name lambda050-initial0907-seed42' >/dev/null; do
    sleep 60
done

# Stop the old queue before it can launch its obsolete GPU2/7 lambda=.75 job.
tmux kill-session -t "$SESSION" 2>/dev/null || true
pkill -TERM -f 'run_name lambda075-initial0907-seed42' 2>/dev/null || true
sleep 10
bash /home/gaoya/Code_Video/Code_data/Code_try0526/run_lambda075_gpu03.sh >> "$LOG" 2>&1
