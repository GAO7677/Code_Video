#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-5}"
PROMPT_SEED_JSON="${PROMPT_SEED_JSON:-/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa2step/scripts/wan22_t2v_prompt_seed_list.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/AAA_test_video/0623/test/t2v_guidance_sweep}"
WAN22_REPO="${WAN22_REPO:-/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main}"
WAN22_CKPT="${WAN22_CKPT:-/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B}"
PYTHON_BIN="${PYTHON_BIN:-/data/gaoya/miniconda3/envs/wan/bin/python}"

SEED_OVERRIDE="${SEED_OVERRIDE:-20250623}"
GUIDANCE_LIST="${GUIDANCE_LIST:-3.0 5.0 7.0}"
STEP_COUNT="${STEP_COUNT:-50}"

HEIGHT="${HEIGHT:-704}"
WIDTH="${WIDTH:-1280}"
GUIDANCE_MODE="${GUIDANCE_MODE:-t2v}"

assert_file() {
  local path="$1"
  if [ ! -f "${path}" ]; then
    echo "[error] file not found: ${path}" >&2
    exit 1
  fi
}

assert_dir() {
  local path="$1"
  if [ ! -d "${path}" ]; then
    echo "[error] directory not found: ${path}" >&2
    exit 1
  fi
}

guidance_tag() {
  local guidance="$1"
  printf '%s\n' "${guidance}" | tr '.' 'p'
}

load_first_prompt_record() {
  "${PYTHON_BIN}" - <<'PY' "${PROMPT_SEED_JSON}"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
records = json.loads(path.read_text(encoding="utf-8"))
if not isinstance(records, list) or not records:
    raise SystemExit("Prompt/seed JSON must be a non-empty list.")
record = records[0]
print(f"{record['id']}\t{record['input_prompt']}")
PY
}

write_result_json() {
  local json_path="$1"
  local prompt_id="$2"
  local prompt="$3"
  local seed="$4"
  local steps="$5"
  local guidance="$6"
  local output_path="$7"

  "${PYTHON_BIN}" - <<'PY' "${json_path}" "${prompt_id}" "${prompt}" "${seed}" "${steps}" "${guidance}" "${output_path}" "${GUIDANCE_MODE}"
import json
import sys
from pathlib import Path

json_path = Path(sys.argv[1])
payload = {
    "id": sys.argv[2],
    "input_prompt": sys.argv[3],
    "seed": int(sys.argv[4]),
    "step": int(sys.argv[5]),
    "guidance_scale": float(sys.argv[6]),
    "output_path": sys.argv[7],
    "model": "wan22_base",
    "mode": sys.argv[8],
}
json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

main() {
  assert_file "${PROMPT_SEED_JSON}"
  assert_file "${PYTHON_BIN}"
  assert_dir "${WAN22_REPO}"
  assert_dir "${WAN22_CKPT}"

  mkdir -p "${OUTPUT_ROOT}/wan22_base"
  cp "${PROMPT_SEED_JSON}" "${OUTPUT_ROOT}/prompt_seed_list.json"

  local record
  record="$(load_first_prompt_record)"
  local prompt_id prompt
  prompt_id="$(printf '%s' "${record}" | cut -f1)"
  prompt="$(printf '%s' "${record}" | cut -f2-)"

  echo "[info] gpu_id=${GPU_ID}"
  echo "[info] output_root=${OUTPUT_ROOT}"
  echo "[info] prompt_id=${prompt_id}"
  echo "[info] seed_override=${SEED_OVERRIDE}"
  echo "[info] step_count=${STEP_COUNT}"
  echo "[info] guidance_list=${GUIDANCE_LIST}"

  for guidance in ${GUIDANCE_LIST}; do
    local gtag base_name output_path json_path
    gtag="$(guidance_tag "${guidance}")"
    base_name="${prompt_id}_wan22_base_step${STEP_COUNT}_guidance${gtag}_seed${SEED_OVERRIDE}"
    output_path="${OUTPUT_ROOT}/wan22_base/${base_name}.mp4"
    json_path="${OUTPUT_ROOT}/wan22_base/${base_name}.json"

    echo "[run] model=wan22_base step=${STEP_COUNT} guidance=${guidance} seed=${SEED_OVERRIDE} gpu=${GPU_ID}"
    echo "[run] output=${output_path}"

    (
      cd "${WAN22_REPO}"
      CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        "${PYTHON_BIN}" generate.py \
          --task ti2v-5B \
          --size "${WIDTH}*${HEIGHT}" \
          --ckpt_dir "${WAN22_CKPT}" \
          --offload_model True \
          --convert_model_dtype \
          --t5_cpu \
          --sample_steps "${STEP_COUNT}" \
          --sample_guide_scale "${guidance}" \
          --base_seed "${SEED_OVERRIDE}" \
          --prompt "${prompt}" \
          --save_file "${output_path}"
    )

    write_result_json "${json_path}" "${prompt_id}" "${prompt}" "${SEED_OVERRIDE}" "${STEP_COUNT}" "${guidance}" "${output_path}"
  done

  echo "[done] completed wan22 base step${STEP_COUNT} guidance seed sweep"
}

main "$@"
