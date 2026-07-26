#!/usr/bin/env bash
set -uo pipefail

# Run:
# GPU_A=4 GPU_B=none bash run_grouped_role_head_ablation_case001460_all.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_BASE="${OUTPUT_BASE:-/data/gaoya/agent-data/outputs/wan_dit_grouped_role_head_ablation/case001460}"
BASELINE_ROOT=/data/gaoya/agent-data/outputs/wan_dit_ball_query_attention/case001460_frame08
GPU_A="${GPU_A:-4}"
GPU_B="${GPU_B:-5}"
RUN_ONE="${SCRIPT_DIR}/run_grouped_role_head_ablation_case001460_one.sh"
VERIFY="${SCRIPT_DIR}/verify_grouped_role_head_ablation.py"
GALLERY="${SCRIPT_DIR}/build_grouped_role_head_ablation_gallery.py"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
LOG_DIR="${OUTPUT_BASE}/logs"
STATE_DIR="${OUTPUT_BASE}/state"
mkdir -p "${LOG_DIR}" "${STATE_DIR}"

TASKS=(
  "wan_lora S" "wan_lora T" "wan_lora P" "wan_lora C" "wan_lora G"
  "xssc S" "xssc T" "xssc P" "xssc C" "xssc G"
  "physrvg S" "physrvg T" "physrvg P" "physrvg C" "physrvg G"
)

run_worker() {
  local gpu="$1" parity="$2" worker="$3"
  local failed=0
  for index in "${!TASKS[@]}"; do
    if (( parity >= 0 )); then
      (( index % 2 == parity )) || continue
    fi
    read -r model category <<< "${TASKS[index]}"
    local tag="self_attn_grouped_head_zero_category_${category,,}"
    local output_root="${OUTPUT_BASE}/${model}/${tag}"
    local log="${LOG_DIR}/${model}_${category}.log"
    local complete="${STATE_DIR}/${model}_${category}.complete"
    if [[ -s "${complete}" ]]; then
      echo "[${worker}] skip ${model}/${category}"
      continue
    fi
    echo "[${worker}] run ${model}/${category} on GPU ${gpu}"
    if OUTPUT_BASE="${OUTPUT_BASE}" \
      bash "${RUN_ONE}" "${model}" "${category}" "${gpu}" > "${log}" 2>&1 \
      && "${PYTHON}" "${VERIFY}" \
        --output-root "${output_root}" --model "${model}" \
        --category "${category}" >> "${log}" 2>&1; then
      printf 'model=%s\ncategory=%s\ngpu=%s\n' \
        "${model}" "${category}" "${gpu}" > "${complete}"
      echo "[${worker}] complete ${model}/${category}"
    else
      echo "[${worker}] FAILED ${model}/${category}; log=${log}" >&2
      failed=1
    fi
  done
  return "${failed}"
}

status=0
if [[ "${GPU_B}" == "none" ]]; then
  run_worker "${GPU_A}" -1 worker-a || status=1
else
  run_worker "${GPU_A}" 0 worker-a &
  pid_a=$!
  run_worker "${GPU_B}" 1 worker-b &
  pid_b=$!
  wait "${pid_a}" || status=1
  wait "${pid_b}" || status=1
fi

if (( status != 0 )); then
  echo "At least one grouped ablation failed; inspect ${LOG_DIR}" >&2
  exit "${status}"
fi

"${PYTHON}" "${GALLERY}" \
  --root "${OUTPUT_BASE}" \
  --baseline-root "${BASELINE_ROOT}" \
  --output "${OUTPUT_BASE}/_gallery"
date -u +%Y-%m-%dT%H:%M:%SZ > "${STATE_DIR}/all.complete"
echo "gallery=${OUTPUT_BASE}/_gallery/index.html"
