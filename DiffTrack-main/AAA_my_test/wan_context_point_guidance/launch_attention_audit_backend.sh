#!/usr/bin/env bash
set -euo pipefail

backend="${1:?usage: launch_attention_audit_backend.sh BACKEND GPU_ID [COORDINATOR]}"
gpu_id="${2:?usage: launch_attention_audit_backend.sh BACKEND GPU_ID [COORDINATOR]}"
coordinator="${3:-0}"

if [[ "$gpu_id" == "4" ]]; then
  echo "GPU 4 is prohibited by workspace rules." >&2
  exit 2
fi
if [[ "$backend" != "firstframe_ti2v" && "$backend" != "context8_v2v" ]]; then
  echo "Unknown backend: $backend" >&2
  exit 2
fi

repo=/home/gaoya/Code_Video/DiffTrack-main
python_bin=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
runner="$repo/AAA_my_test/wan_context_point_guidance/run_dual_protocol.py"
diagnostics="$repo/AAA_my_test/wan_context_point_guidance/render_constraint_diagnostics.py"
output_root=/data/gaoya/agent-data/outputs/wan_context_point_guidance_head_compare/attention_audit_v3
runtime_root="$output_root/_split_runtime"
log_root="$output_root/logs"
marker_root="$output_root/split_gpu23_status"
other_backend=firstframe_ti2v
if [[ "$backend" == "firstframe_ti2v" ]]; then
  other_backend=context8_v2v
fi

mkdir -p "$log_root" "$marker_root"
rm -f "$marker_root/$backend.done" "$marker_root/$backend.failed"
cd "$repo"
export CUDA_VISIBLE_DEVICES="$gpu_id"
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main:/home/gaoya/Code_Video/DiffTrack-main
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log="$log_root/${backend}_gpu${gpu_id}.log"
echo "[start] $(date -u +%FT%TZ) backend=$backend physical_gpu=$gpu_id" | tee -a "$log"
set +e
"$python_bin" -u "$runner" \
  --backend "$backend" \
  --stage all \
  --device cuda:0 \
  --output-root "$output_root" 2>&1 | tee -a "$log"
rc=${PIPESTATUS[0]}
set -e
if (( rc != 0 )); then
  date -u +%FT%TZ > "$marker_root/$backend.failed"
  echo "[failed] $(date -u +%FT%TZ) backend=$backend rc=$rc" | tee -a "$log"
  exit "$rc"
fi
date -u +%FT%TZ > "$marker_root/$backend.done"
echo "[done] $(date -u +%FT%TZ) backend=$backend" | tee -a "$log"

if [[ "$coordinator" == "1" ]]; then
  echo "[wait] waiting for $other_backend before shared diagnostics" | tee -a "$log"
  while [[ ! -f "$marker_root/$other_backend.done" ]]; do
    if [[ -f "$marker_root/$other_backend.failed" ]]; then
      echo "[failed] $other_backend failed; shared diagnostics not started" | tee -a "$log"
      exit 3
    fi
    sleep 30
  done
  echo "[diagnostics] $(date -u +%FT%TZ) both backends complete" | tee -a "$log"
  "$python_bin" -u "$diagnostics" \
    --backend all \
    --device cuda:0 \
    --output-root "$output_root" 2>&1 | tee -a "$log"
fi
