#!/usr/bin/env bash
set -euo pipefail

# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_fill_missing_baselines_test5_tmux.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-${SCRIPT_DIR}/config_fill_missing_baselines_test5.sh}"
# shellcheck source=/dev/null
source "${CONFIG}"
WORKER="${SCRIPT_DIR}/run_fill_missing_baselines_worker.sh"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi
mkdir -p "${BASELINE_RUN_ROOT}/inputs" "${BASELINE_RUN_ROOT}/logs" \
  "${BASELINE_RUN_ROOT}/state" "${BASELINE_RUN_ROOT}/validations"

python3 - "${HEAD_RUN_ROOT}/input_unique.txt" "${BASELINE_RUN_ROOT}/inputs" <<'PY'
import importlib.util
import sys
from pathlib import Path

source = Path(sys.argv[1])
output = Path(sys.argv[2])
script = Path(
    "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/"
    "AAA_wan_dit/serve_configured_head_ablation_gallery.py"
)
spec = importlib.util.spec_from_file_location("head_gallery", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
inputs = [Path(line).resolve() for line in source.read_text().splitlines() if line.strip()]
for model in module.MODEL_LABELS:
    missing = [
        path for path in inputs
        if module.baseline_path(model, path.stem) is None
    ]
    (output / f"{model}.txt").write_text(
        "".join(f"{path}\n" for path in missing), encoding="utf-8"
    )
    print(f"{model}_missing={len(missing)}")
PY

cp "${CONFIG}" "${BASELINE_RUN_ROOT}/config_snapshot.sh"
tmux new-session -d -s "${SESSION}" -n status \
  "while true; do date -u; find '${BASELINE_RUN_ROOT}/state' -maxdepth 1 -type f -printf '%f\n' | sort; sleep 30; done"
tmux new-window -t "${SESSION}" -n gpu5 \
  "bash '${WORKER}' '${CONFIG}' 5 baseline_g5 '${GPU5_MODELS}'; exec bash"
tmux new-window -t "${SESSION}" -n gpu6 \
  "bash '${WORKER}' '${CONFIG}' 6 baseline_g6 '${GPU6_MODELS}'; exec bash"
tmux select-window -t "${SESSION}:status"

echo "session=${SESSION}"
echo "run_root=${BASELINE_RUN_ROOT}"
echo "gpu5_models=${GPU5_MODELS}"
echo "gpu6_models=${GPU6_MODELS}"
