#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="tdw_motion_cases"
SCRIPT_PATH="/home/gaoya/Code_Video/Code_data/data0417/tdw_data/run_tdw_motion_cases.py"
LOG_PATH="/data/gaoya/AAA_test_video/Dataset_physV/0505TDW/tdw_motion_cases/tmux_run.log"
DISPLAY_ID=":1"

mkdir -p "$(dirname "$LOG_PATH")"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  tmux kill-session -t "$SESSION_NAME"
fi

tmux new-session -d -s "$SESSION_NAME" "bash -lc '
  source /home/gaoya/.venvs/tdw/bin/activate
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
  export DISPLAY=$DISPLAY_ID
  : > $LOG_PATH
  if ! xdpyinfo >/dev/null 2>&1; then
    echo \"DISPLAY $DISPLAY_ID is not available. Start Xorg first, then rerun.\" | tee -a $LOG_PATH
    exit 1
  fi
  python -u $SCRIPT_PATH 2>&1 | tee -a $LOG_PATH
'"
echo "tmux session started: $SESSION_NAME"
echo "attach: tmux attach -t $SESSION_NAME"
echo "log: $LOG_PATH"
