#!/usr/bin/env bash
set -euo pipefail

# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_s_interim_full_metrics_waiter.sh

ROOT="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
ANALYSIS="/data/gaoya/agent-data/outputs/wan_dit_s_motion_analysis"
VBENCH="${ANALYSIS}/vbench_snapshots/final_20260730T064742Z"
STATE="${ANALYSIS}/state"
OUTPUT_BASE="${ANALYSIS}/full_metric_snapshots"
READY="${ANALYSIS}/interim_full_metrics.complete"
FAILED="${ANALYSIS}/interim_full_metrics.failed"
GALLERY_CONFIG="${ROOT}/head_role_dose_control_pilot.json"
DOMINANT_CONFIG="${ROOT}/head_role_s_dominant_depth_experiment.json"

rm -f "${READY}" "${FAILED}"
echo "[interim-metrics] waiting for current VBench and Motion feature shards"
while [[ ! -f "${VBENCH}/run.complete" ]] || \
      [[ "$(find "${STATE}" -maxdepth 1 -name 'features_shard_*.complete' -type f | wc -l)" -ne 7 ]]; do
  if [[ -f "${VBENCH}/run.failed" ]] || [[ -f "${STATE}/pipeline.failed" ]]; then
    touch "${FAILED}"
    echo "[interim-metrics] prerequisite VBench or Motion pipeline failed" >&2
    exit 1
  fi
  vbench="$(wc -l < "${VBENCH}/completed_tasks.tsv")"
  shards="$(find "${STATE}" -maxdepth 1 -name 'features_shard_*.complete' -type f | wc -l)"
  echo "[interim-metrics] vbench=${vbench}/106 motion_shards=${shards}/7"
  sleep 60
done

"${PYTHON}" "${ROOT}/build_head_role_dose_control_case_gallery.py" \
  --config "${GALLERY_CONFIG}" \
  --s-dominant-depth-config "${DOMINANT_CONFIG}"
"${PYTHON}" "${ROOT}/build_gallery_missing_metric_snapshot.py" \
  --output-base "${OUTPUT_BASE}"
snapshot="$(cat "${OUTPUT_BASE}/latest")"
session="wan_s_full_metrics_interim_$(basename "${snapshot}")"
SESSION="${session}" GPU_MIN_FREE_MIB=30000 \
  bash "${ROOT}/run_s_full_metric_snapshot_tmux.sh" "${snapshot}"

while [[ ! -f "${snapshot}/run.complete" ]]; do
  if [[ -f "${snapshot}/run.failed" ]]; then
    touch "${FAILED}"
    echo "[interim-metrics] metric snapshot failed" >&2
    exit 1
  fi
  sleep 60
done

"${PYTHON}" "${ROOT}/build_head_role_dose_control_case_gallery.py" \
  --config "${GALLERY_CONFIG}" \
  --s-dominant-depth-config "${DOMINANT_CONFIG}"
"${PYTHON}" "${ROOT}/build_motion_n_analysis_status.py"
touch "${snapshot}/gallery.complete"
touch "${READY}"
echo "[interim-metrics] complete: ${snapshot}"
