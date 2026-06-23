#!/usr/bin/env bash
set -euo pipefail

# Batch t2v sweep for:
# - wan2.2 TI2V-5B in text-to-video mode
# - LoRA step-000500
# - LoRA step-001000
#
# Inputs come from a JSON list of dicts:
#   [{"id": "...", "input_prompt": "...", "seed": 20250622}, ...]
#
# Outputs:
#   /data/gaoya/AAA_test_video/tmp/0623wan22_t2v_base_lora_step_sweep/
#     wan22_base/
#     lora_step000500/
#     lora_step001000/

GPU_IDS="${GPU_IDS:-5 6 7}"
PROMPT_SEED_JSON="${PROMPT_SEED_JSON:-/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa2step/scripts/wan22_t2v_prompt_seed_list.json}"
STEPS_LIST="${STEPS_LIST:-5 15 25 50}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/AAA_test_video/0623/test/t2v}"

WAN22_REPO="${WAN22_REPO:-/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main}"
WAN22_CKPT="${WAN22_CKPT:-/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B}"

PYTHON_BIN="${PYTHON_BIN:-/data/gaoya/miniconda3/envs/wan/bin/python}"
BASE_PYTHONPATH="${BASE_PYTHONPATH:-/home/gaoya/Code_Video/Code_data/Code_train/train_0419:/home/gaoya/Code_Video/DiffSynth-Studio-main}"
T2V_LORA_SCRIPT="${T2V_LORA_SCRIPT:-/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/infer_t2v_lora.py}"

HEIGHT="${HEIGHT:-704}"
WIDTH="${WIDTH:-1280}"
NUM_FRAMES="${NUM_FRAMES:-121}"
FPS="${FPS:-24}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-5.0}"
MODE="${MODE:-t2v}"
TMUX_LAUNCH_NOTE="${TMUX_LAUNCH_NOTE:-}"

declare -A MODEL_DIRS=(
  ["wan22_base"]="wan22_base"
  ["lora_step000500"]="lora_step000500"
  ["lora_step001000"]="lora_step001000"
)

declare -A LORA_PATHS=(
  ["lora_step000500"]="/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors"
  ["lora_step001000"]="/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-001000/checkpoint.safetensors"
)

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

next_gpu_id() {
  local idx=$((GPU_CURSOR % GPU_COUNT))
  local gpu_id="${GPU_ID_ARRAY[${idx}]}"
  GPU_CURSOR=$((GPU_CURSOR + 1))
  printf '%s\n' "${gpu_id}"
}

iter_prompt_seed_records() {
  "${PYTHON_BIN}" - <<'PY' "${PROMPT_SEED_JSON}"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
records = json.loads(path.read_text(encoding="utf-8"))
if not isinstance(records, list):
    raise SystemExit("Prompt/seed JSON must be a list.")
for record in records:
    if not isinstance(record, dict):
        raise SystemExit("Each prompt/seed item must be a dict.")
    prompt_id = str(record["id"])
    prompt = str(record["input_prompt"])
    seed = int(record["seed"])
    print(f"{prompt_id}\t{prompt}\t{seed}")
PY
}

write_result_json() {
  local json_path="$1"
  local prompt_id="$2"
  local prompt="$3"
  local seed="$4"
  local steps="$5"
  local output_path="$6"
  local model_name="$7"
  local lora_path="$8"

  "${PYTHON_BIN}" - <<'PY' "${json_path}" "${prompt_id}" "${prompt}" "${seed}" "${steps}" "${GUIDANCE_SCALE}" "${output_path}" "${model_name}" "${MODE}" "${lora_path}"
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
    "model": sys.argv[8],
    "mode": sys.argv[9],
}
lora_path = sys.argv[10]
if lora_path:
    payload["lora_path"] = lora_path
json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

run_base_t2v() {
  local prompt_id="$1"
  local prompt="$2"
  local seed="$3"
  local steps="$4"
  local gpu_id="$5"
  local model_name="wan22_base"
  local model_dir="${OUTPUT_ROOT}/${MODEL_DIRS[${model_name}]}"
  local base_name="${model_name}_seed${seed}_step${steps}_guidance5p0"
  local output_path="${model_dir}/${base_name}.mp4"
  local json_path="${model_dir}/${base_name}.json"

  mkdir -p "${model_dir}"
  echo "[run] model=${model_name} id=${prompt_id} steps=${steps} seed=${seed} gpu=${gpu_id}"
  echo "[run] output=${output_path}"

  (
    cd "${WAN22_REPO}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
      "${PYTHON_BIN}" generate.py \
        --task ti2v-5B \
        --size "${WIDTH}*${HEIGHT}" \
        --ckpt_dir "${WAN22_CKPT}" \
        --offload_model True \
        --convert_model_dtype \
        --t5_cpu \
        --sample_steps "${steps}" \
        --sample_guide_scale "${GUIDANCE_SCALE}" \
        --base_seed "${seed}" \
        --prompt "${prompt}" \
        --save_file "${output_path}"
  )

  write_result_json "${json_path}" "${prompt_id}" "${prompt}" "${seed}" "${steps}" "${output_path}" "${model_name}" ""
}

run_lora_t2v() {
  local prompt_id="$1"
  local prompt="$2"
  local seed="$3"
  local steps="$4"
  local model_name="$5"
  local lora_path="$6"
  local gpu_id="$7"
  local model_dir="${OUTPUT_ROOT}/${MODEL_DIRS[${model_name}]}"
  local base_name="${model_name}_seed${seed}_step${steps}_guidance5p0"
  local output_path="${model_dir}/${base_name}.mp4"
  local json_path="${model_dir}/${base_name}.json"

  mkdir -p "${model_dir}"
  echo "[run] model=${model_name} id=${prompt_id} steps=${steps} seed=${seed} gpu=${gpu_id}"
  echo "[run] output=${output_path}"

  CUDA_VISIBLE_DEVICES="${gpu_id}" \
    PYTHONPATH="${BASE_PYTHONPATH}" \
    "${PYTHON_BIN}" "${T2V_LORA_SCRIPT}" \
      --wan_root "${WAN22_CKPT}" \
      --lora_path "${lora_path}" \
      --output_video_path "${output_path}" \
      --prompt "${prompt}" \
      --seed "${seed}" \
      --height "${HEIGHT}" \
      --width "${WIDTH}" \
      --num_frames "${NUM_FRAMES}" \
      --fps "${FPS}" \
      --num_inference_steps "${steps}" \
      --cfg_scale "${GUIDANCE_SCALE}" \
      --overwrite

  write_result_json "${json_path}" "${prompt_id}" "${prompt}" "${seed}" "${steps}" "${output_path}" "${model_name}" "${lora_path}"
}

main() {
  read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"
  GPU_COUNT="${#GPU_ID_ARRAY[@]}"
  GPU_CURSOR=0
  if [ "${GPU_COUNT}" -eq 0 ]; then
    echo "[error] no GPU ids configured" >&2
    exit 1
  fi

  assert_file "${PROMPT_SEED_JSON}"
  assert_dir "${WAN22_REPO}"
  assert_dir "${WAN22_CKPT}"
  assert_file "${PYTHON_BIN}"
  assert_file "${T2V_LORA_SCRIPT}"
  for lora_path in "${LORA_PATHS[@]}"; do
    assert_file "${lora_path}"
  done

  mkdir -p "${OUTPUT_ROOT}"
  cp "${PROMPT_SEED_JSON}" "${OUTPUT_ROOT}/prompt_seed_list.json"

  echo "[info] gpu_ids=${GPU_IDS}"
  echo "[info] prompt_seed_json=${PROMPT_SEED_JSON}"
  echo "[info] output_root=${OUTPUT_ROOT}"
  echo "[info] steps=${STEPS_LIST}"
  echo "[info] mode=${MODE}"
  if [ -n "${TMUX_LAUNCH_NOTE}" ]; then
    echo "[info] tmux_launch_note=${TMUX_LAUNCH_NOTE}"
  fi

  while IFS=$'\t' read -r prompt_id prompt seed; do
    for steps in ${STEPS_LIST}; do
      run_base_t2v "${prompt_id}" "${prompt}" "${seed}" "${steps}" "$(next_gpu_id)"
      for model_name in $(printf '%s\n' "${!LORA_PATHS[@]}" | sort); do
        run_lora_t2v "${prompt_id}" "${prompt}" "${seed}" "${steps}" "${model_name}" "${LORA_PATHS[${model_name}]}" "$(next_gpu_id)"
      done
    done
  done < <(iter_prompt_seed_records)

  echo "[done] completed wan2.2 base+lora t2v json-driven sweep"
}

main "$@"
