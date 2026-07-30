#!/usr/bin/env bash
set -euo pipefail

# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_s_dominant_depth_postprocess_waiter.sh

ROOT="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit"
PYTHON="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
LOCAL_ROOT="/data/gaoya/agent-data/outputs/wan_dit_s_dominant_depth/seed851"
ANALYSIS="/data/gaoya/agent-data/outputs/wan_dit_s_motion_analysis"
FULL_METRIC_BASE="${ANALYSIS}/full_metric_snapshots"
GALLERY_CONFIG="${ROOT}/head_role_dose_control_pilot.json"
DOMINANT_CONFIG="${ROOT}/head_role_s_dominant_depth_experiment.json"
EXPECTED_TASKS=72
INTERIM_READY="${ANALYSIS}/interim_full_metrics.complete"
INTERIM_FAILED="${ANALYSIS}/interim_full_metrics.failed"

count_complete() {
  "${PYTHON}" - "${LOCAL_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
count = 0
for path in (root / "state").glob("*.json"):
    try:
        count += json.loads(path.read_text(encoding="utf-8")).get("status") == "complete"
    except json.JSONDecodeError:
        pass
print(count)
PY
}

echo "[dominant-post] waiting for validated local pull"
while [[ "$(count_complete)" -ne "${EXPECTED_TASKS}" ]]; do
  echo "[dominant-post] local_complete=$(count_complete)/${EXPECTED_TASKS}"
  sleep 60
done

echo "[dominant-post] waiting for interim full-metric audit"
while [[ ! -f "${INTERIM_READY}" ]]; do
  if [[ -f "${INTERIM_FAILED}" ]]; then
    echo "[dominant-post] interim full-metric audit failed" >&2
    exit 1
  fi
  sleep 60
done

"${PYTHON}" "${ROOT}/build_head_role_dose_control_case_gallery.py" \
  --config "${GALLERY_CONFIG}" \
  --s-dominant-depth-config "${DOMINANT_CONFIG}"

"${PYTHON}" "${ROOT}/build_s_motion_inventory.py" --output-root "${ANALYSIS}"
"${PYTHON}" "${ROOT}/build_gallery_missing_metric_snapshot.py" \
  --output-base "${FULL_METRIC_BASE}"
metric_snapshot="$(cat "${FULL_METRIC_BASE}/latest")"
metric_session="wan_s_full_metrics_$(basename "${metric_snapshot}")"
SESSION="${metric_session}" \
  bash "${ROOT}/run_s_full_metric_snapshot_tmux.sh" "${metric_snapshot}"

while [[ ! -f "${metric_snapshot}/run.complete" ]]; do
  if [[ -f "${metric_snapshot}/run.failed" ]]; then
    echo "[dominant-post] full metric snapshot failed" >&2
    exit 1
  fi
  sleep 60
done
while [[ ! -f "${ANALYSIS}/post_generation_motion.complete" ]]; do
  if [[ -f "${ANALYSIS}/post_generation_motion.failed" ]]; then
    echo "[dominant-post] Motion analysis failed" >&2
    exit 1
  fi
  sleep 60
done

"${PYTHON}" "${ROOT}/build_head_role_dose_control_case_gallery.py" \
  --config "${GALLERY_CONFIG}" \
  --s-dominant-depth-config "${DOMINANT_CONFIG}"
"${PYTHON}" "${ROOT}/build_motion_n_analysis_status.py"
"${PYTHON}" "${ROOT}/build_s_head_integrated_analysis.py"
"${PYTHON}" "${ROOT}/build_visualization_hub.py"
touch "${LOCAL_ROOT}/postprocess.complete"
echo "[dominant-post] metrics and galleries complete"
