#!/usr/bin/env bash
set -euo pipefail

# MODEL=wan_lora GPU=3 bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/run_common_stc_all_heads_qk_model_worker.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:?set MODEL}"
GPU="${GPU:?set GPU}"
MAX_USED_MIB="${MAX_USED_MIB:-4096}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-5}"
ROOT=/data/gaoya/agent-data/outputs/wan_dit_common_stc_all_heads_qk_seed851
INPUT_LIST="${SCRIPT_DIR}/common22_public_head_ablation_case025.txt"
STATE_DIR="${ROOT}/state"
LOG="${ROOT}/logs/${MODEL}.log"

if [[ "${GPU}" == "4" ]]; then
  echo "GPU 4 is disabled by workspace policy" >&2
  exit 2
fi
case "${MODEL}" in
  wan_lora|xssc|physrvg) ;;
  *) echo "unsupported MODEL=${MODEL}" >&2; exit 2 ;;
esac

mkdir -p "${ROOT}/logs" "${STATE_DIR}"
rm -f "${STATE_DIR}/${MODEL}.failed"
: >"${LOG}"
for ((attempt=1; attempt<=MAX_ATTEMPTS; attempt++)); do
  while true; do
    used="$(nvidia-smi -i "${GPU}" --query-gpu=memory.used \
      --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
    if [[ "${used}" =~ ^[0-9]+$ ]] && (( used <= MAX_USED_MIB )); then
      echo "[common-stc-qk] ${MODEL} attempt ${attempt}/${MAX_ATTEMPTS} starts on GPU${GPU}; used=${used} MiB"
      break
    fi
    echo "[common-stc-qk] ${MODEL} waits for GPU${GPU}; used=${used:-unknown} MiB"
    sleep "${POLL_SECONDS}"
  done

  attempt_log="${ROOT}/logs/${MODEL}.attempt-${attempt}.log"
  set +e
  MODEL="${MODEL}" GPU="${GPU}" SEED=851 ROOT="${ROOT}" \
    OUTPUT_ROOT="${ROOT}/capture/${MODEL}" \
    SELECTION="${ROOT}/selection.json" INPUT_LIST="${INPUT_LIST}" \
    STEPS=5,15,25,35 OUTPUT_BINS=512 QUERY_CHUNK=64 \
    bash "${SCRIPT_DIR}/run_selected_qk_capture.sh" \
    2>&1 | tee "${attempt_log}"
  status="${PIPESTATUS[0]}"
  set -e
  cat "${attempt_log}" >>"${LOG}"
  if [[ "${status}" -eq 0 ]]; then
    touch "${STATE_DIR}/${MODEL}.complete"
    echo "[common-stc-qk] ${MODEL} complete"
    exit 0
  fi

  if grep -q "CUDA out of memory" "${attempt_log}"; then
    echo "[common-stc-qk] ${MODEL} hit a GPU allocation race; requeueing"
    sleep "${POLL_SECONDS}"
    continue
  fi
  printf '%s\n' "${status}" >"${STATE_DIR}/${MODEL}.failed"
  echo "[common-stc-qk] ${MODEL} failed with non-OOM status ${status}" >&2
  exit "${status}"
done

printf '%s\n' "${status:-1}" >"${STATE_DIR}/${MODEL}.failed"
echo "[common-stc-qk] ${MODEL} exhausted ${MAX_ATTEMPTS} attempts" >&2
exit "${status:-1}"
