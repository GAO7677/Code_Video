#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT=/data/gaoya/agent-data/outputs/train_subset_val_loss_seed42
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
PROJECT_ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
SCRIPT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/xssc_loss_project/evaluate_train_subset_val_loss.py

while true; do
  env \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH="${PROJECT_ROOT}:${DIFFSYNTH_ROOT}" \
    "${PYTHON}" "${SCRIPT}" \
    --output-root "${OUTPUT_ROOT}" \
    --build-report-only
  complete="$(${PYTHON} - "${OUTPUT_ROOT}/rankings.json" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(int(payload["complete_entries"]) >= int(payload["total_entries"]))
PY
)"
  if [[ "${complete}" == "True" ]]; then
    break
  fi
  sleep 60
done
