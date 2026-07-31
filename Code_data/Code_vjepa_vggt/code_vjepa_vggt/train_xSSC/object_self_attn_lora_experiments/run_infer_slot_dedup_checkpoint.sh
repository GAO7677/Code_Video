#!/usr/bin/env bash
# Run:
# NUM_INFERENCE_STEPS=8 TEST_LIST=/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/run_infer_slot_dedup_checkpoint.sh CHECKPOINT_DIR GPU_ID [OUTPUT_ROOT]
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 CHECKPOINT_DIR GPU_ID [OUTPUT_ROOT]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR="$(realpath "$1")"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
search_dir="${CHECKPOINT_DIR}"
experiment_config=""
while [[ "${search_dir}" != "/" ]]; do
  candidate="${search_dir}/resolved_experiment_config.json"
  if [[ -s "${candidate}" ]]; then
    experiment_config="${candidate}"
    break
  fi
  search_dir="$(dirname "${search_dir}")"
done
if [[ -z "${experiment_config}" ]]; then
  echo "Could not find resolved_experiment_config.json above ${CHECKPOINT_DIR}" >&2
  exit 2
fi

dedup_mode="$(${PYTHON} - "${experiment_config}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], "r", encoding="utf-8"))
print(payload["resolved_config"].get("conditioning", {}).get("slot_dedup", {}).get("mode", "none"))
PY
)"
if [[ "${dedup_mode}" == "none" ]]; then
  echo "Checkpoint config does not enable slot de-duplication: ${experiment_config}" >&2
  exit 2
fi

export EXPERIMENT_CONFIG="${experiment_config}"
exec bash "${SCRIPT_DIR}/run_infer_from_experiment.sh" "$@"
