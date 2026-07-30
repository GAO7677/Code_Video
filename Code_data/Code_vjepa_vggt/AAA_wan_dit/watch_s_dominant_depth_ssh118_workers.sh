#!/usr/bin/env bash
set -euo pipefail

# Run this script on SSH host 118.

ROOT="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit"
OUTPUT="/mnt/data/gaoya/agent-data/outputs/wan_dit_s_dominant_depth/seed851"
SESSION="wan_s_dominant_depth_ssh118"
EXPECTED=72

status_counts() {
  python3 - "${OUTPUT}" <<'PY'
import collections
import json
import sys
from pathlib import Path

counts = collections.Counter()
for path in (Path(sys.argv[1]) / "state").glob("*.json"):
    try:
        counts[json.loads(path.read_text(encoding="utf-8")).get("status", "invalid")] += 1
    except json.JSONDecodeError:
        counts["invalid"] += 1
print(
    counts.get("complete", 0),
    counts.get("running", 0),
    counts.get("failed", 0),
)
PY
}

while true; do
  read -r complete running failed < <(status_counts)
  workers="$(
    {
      pgrep -af "run_head_role_dose_control_pilot_worker.py.*head_role_s_dominant_depth_experiment_ssh118.json.*--gpu" \
        || true
    } | wc -l
  )"
  echo "[remote-watch] complete=${complete}/${EXPECTED} running=${running} failed=${failed} workers=${workers}"
  if [[ "${complete}" -eq "${EXPECTED}" ]]; then
    break
  fi
  if [[ "${workers}" -eq 0 ]]; then
    if tmux has-session -t "${SESSION}" 2>/dev/null; then
      while read -r window; do
        [[ "${window}" == remote_g* ]] || continue
        tmux kill-window -t "${SESSION}:${window}"
      done < <(tmux list-windows -t "${SESSION}" -F '#W')
    fi
    bash "${ROOT}/run_s_dominant_depth_ssh118_remote.sh"
  fi
  sleep 60
done

touch "${OUTPUT}/generation.complete"
echo "[remote-watch] all generation tasks complete"
