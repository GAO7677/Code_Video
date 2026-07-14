#!/usr/bin/env bash
set -u

BASE=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box
PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
PROJ=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
INFER_SCRIPT="${BASE}/wan_stage1b_scheme_c_entity_caption_physical_v2v.py"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:?Set CHECKPOINT_ROOT to the training run checkpoints directory}"
INPUT_LIST="${INPUT_LIST:-/data/gaoya/agent-data/cache/stage1b_scheme_c_entity_caption_physical_fresh_val/physiq4_input_jsons.txt}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT under /data/gaoya/agent-data/outputs}"
MODEL_PREFIX="${MODEL_PREFIX:-stage1b_scheme_c_entity_caption_physical_fresh}"
GPU_UUID="${GPU_UUID:-GPU-99e4d61a-1169-14e0-d90c-364fdbe30065}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MIN_CHECKPOINT_AGE_SECONDS="${MIN_CHECKPOINT_AGE_SECONDS:-90}"
TMP_ROOT="${TMP_ROOT:-/data/gaoya/agent-data/cache/t/entity_physiq4_watch}"
STATE_ROOT="${STATE_ROOT:-${OUTPUT_ROOT}/_watch_state}"
WATCH_LOG="${STATE_ROOT}/watch.log"
LOCK_DIR="${STATE_ROOT}/watch.lock"

mkdir -p "${OUTPUT_ROOT}" "${TMP_ROOT}" "${STATE_ROOT}"
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "ERROR: watcher already running or stale lock exists: ${LOCK_DIR}" >&2
  exit 1
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${WATCH_LOG}"
}

checkpoint_is_ready() {
  local step_dir="$1"
  local checkpoint_file="${step_dir}/checkpoint.safetensors"
  local training_state="${step_dir}/training_state.pt"
  local now mtime age
  [ -s "${checkpoint_file}" ] && [ -s "${training_state}" ] || return 1
  now="$(date +%s)"
  mtime="$(stat -c '%Y' "${checkpoint_file}")"
  age=$((now - mtime))
  [ "${age}" -ge "${MIN_CHECKPOINT_AGE_SECONDS}" ]
}

gpu_is_available() {
  local memory_used
  memory_used="$(nvidia-smi -i "${GPU_UUID}" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')"
  [ -n "${memory_used}" ] && [ "${memory_used}" -lt 2048 ]
}

result_is_complete() {
  local result_path="$1"
  local expected_checkpoint="$2"
  [ -s "${result_path}" ] || return 1
  "${PYTHON_BIN}" - "${result_path}" "${expected_checkpoint}" "${INPUT_LIST}" <<'PY'
import json
import sys
from pathlib import Path

import cv2

result_path = Path(sys.argv[1])
expected_checkpoint = str(Path(sys.argv[2]).resolve())
expected_inputs = {
    str(Path(line.strip()).resolve())
    for line in Path(sys.argv[3]).read_text(encoding="utf-8").splitlines()
    if line.strip()
}
try:
    result = json.loads(result_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
entries = result.get("entries", [])
if not (
    str(Path(result.get("checkpoint_dir", "")).resolve()) == expected_checkpoint
    and int(result.get("num_total", -1)) == 4
    and int(result.get("num_success", -1)) == 4
    and int(result.get("num_failed", -1)) == 0
    and int(result.get("num_skipped", -1)) == 0
    and len(entries) == 4
    and {
        str(Path(entry.get("input_json", "")).resolve()) for entry in entries
    } == expected_inputs
):
    raise SystemExit(1)
output_videos = set()
for entry in entries:
    video_path = Path(entry.get("output_video", ""))
    if not video_path.is_file() or video_path.stat().st_size <= 0:
        raise SystemExit(1)
    output_videos.add(str(video_path.resolve()))
    capture = cv2.VideoCapture(str(video_path))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, _ = capture.read()
    capture.release()
    if not ok or frames != 49 or (width, height) != (896, 512):
        raise SystemExit(1)
    binding = entry.get("object_debug", {}).get("entity_id_binding", {})
    if not binding.get("enabled"):
        raise SystemExit(1)
    matched = binding.get("matched", [])
    unmatched = binding.get("unmatched", [])
    adapter_metrics = binding.get("adapter_metrics", {})
    matched_ids = [item.get("entity_id") for item in matched]
    if (
        not matched
        or unmatched
        or len(set(matched_ids)) != len(matched_ids)
        or float(adapter_metrics.get("train/entity_binding_active", 0.0)) != 1.0
        or int(adapter_metrics.get("train/entity_binding_valid_slot_count", -1))
        != len(matched)
        or int(adapter_metrics.get("train/entity_binding_matched_slot_count", -1))
        != len(matched)
        or int(adapter_metrics.get("train/entity_binding_id_collision_count", -1)) != 0
    ):
        raise SystemExit(1)
if len(output_videos) != 4:
    raise SystemExit(1)
print(result_path)
PY
}

run_checkpoint() {
  local step_dir="$1"
  local step_name output_dir result_path model_name tmp_dir run_log
  step_name="$(basename "${step_dir}")"
  output_dir="${OUTPUT_ROOT}/${step_name}"
  result_path="${output_dir}/results/result.json"
  model_name="${MODEL_PREFIX}_${step_name}"
  tmp_dir="${TMP_ROOT}/${step_name}"
  run_log="${STATE_ROOT}/${step_name}.log"

  if result_is_complete "${result_path}" "${step_dir}" >>"${WATCH_LOG}" 2>&1; then
    return 0
  fi
  if ! gpu_is_available; then
    log "defer ${step_name}: validation GPU ${GPU_UUID} is busy"
    return 0
  fi

  mkdir -p "${output_dir}" "${tmp_dir}"
  log "infer start ${step_name}: checkpoint=${step_dir} output=${output_dir}"
  if env \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH="${PROJ}:${DIFFSYNTH_ROOT}" \
    CUDA_VISIBLE_DEVICES="${GPU_UUID}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TMPDIR="${tmp_dir}" TMP="${tmp_dir}" TEMP="${tmp_dir}" \
    "${PYTHON_BIN}" "${INFER_SCRIPT}" \
      --weights-root "${step_dir}" \
      --input-json-list-path "${INPUT_LIST}" \
      --model-name "${model_name}" \
      --output-root "${output_dir}" \
      --step-output-dir-name results \
      --height 512 --width 896 \
      --context-frames 8 --num-frames 49 \
      --num-inference-steps 40 --cfg-scale 5.0 --seed 42 --fps 30 \
      --force >"${run_log}" 2>&1; then
    if result_is_complete "${result_path}" "${step_dir}" >>"${WATCH_LOG}" 2>&1; then
      log "infer success ${step_name}: 4/4 videos verified at ${output_dir}"
    else
      log "infer incomplete ${step_name}: process exited 0 but result verification failed"
    fi
  else
    status=$?
    log "infer failed ${step_name}: exit=${status}; retry in ${POLL_SECONDS}s; log=${run_log}"
  fi
}

if [ "$(sed '/^[[:space:]]*$/d' "${INPUT_LIST}" | wc -l)" -ne 4 ]; then
  echo "ERROR: INPUT_LIST must contain exactly four cases: ${INPUT_LIST}" >&2
  exit 1
fi
for input_json in $(sed '/^[[:space:]]*$/d' "${INPUT_LIST}"); do
  if [ ! -s "${input_json}" ]; then
    echo "ERROR: missing input JSON: ${input_json}" >&2
    exit 1
  fi
done

log "watcher start checkpoint_root=${CHECKPOINT_ROOT} gpu_uuid=${GPU_UUID} input_list=${INPUT_LIST}"
while true; do
  found=0
  while IFS= read -r step_dir; do
    [ -n "${step_dir}" ] || continue
    found=1
    if checkpoint_is_ready "${step_dir}"; then
      run_checkpoint "${step_dir}"
    fi
  done < <(find "${CHECKPOINT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'step-*' -print 2>/dev/null | sort -V)
  if [ "${found}" -eq 0 ]; then
    log "waiting: no checkpoints under ${CHECKPOINT_ROOT}"
  fi
  sleep "${POLL_SECONDS}"
done
