#!/usr/bin/env bash
set -euo pipefail

wait_pid="${WAIT_PID:-2590621}"
repo=/home/gaoya/Code_Video/DiffTrack-main
launcher="${repo}/AAA_my_test/wan_context_point_guidance/launch_attention_audit_backend.sh"
output=/data/gaoya/agent-data/outputs/wan_context_point_guidance_head_compare/attention_audit_v3
log="${output}/logs/firstframe_move_to_gpu3_queue.log"
mkdir -p "$(dirname "${log}")"
exec > >(tee -a "${log}") 2>&1

while [[ -r "/proc/${wait_pid}/cmdline" ]]; do
  current_cmd="$(tr '\0' ' ' < "/proc/${wait_pid}/cmdline")"
  if [[ "${current_cmd}" != *"run_dual_protocol.py"* ]] || \
     [[ "${current_cmd}" != *"--backend context8_v2v"* ]]; then
    break
  fi
  echo "[$(date -u +%FT%TZ)] waiting for GPU3 context8 PID ${wait_pid}"
  sleep 30
done

echo "[$(date -u +%FT%TZ)] resuming firstframe_ti2v on physical GPU3"
cd "${repo}"
exec "${launcher}" firstframe_ti2v 3 1
