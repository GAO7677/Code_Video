#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${CONFIG:-${SCRIPT_DIR}/physrvg_verified_remote118_gpu67.env}"
# shellcheck source=/dev/null
source "${CONFIG}"

RUN_DIR="${REMOTE_OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_DIR}" "${REMOTE_EVAL_ROOT}" "${REMOTE_LOG_ROOT}"

generate_shard() {
  local physical_gpu="$1"
  local shard_index="$2"
  CUDA_VISIBLE_DEVICES="${physical_gpu}" "${REMOTE_PHYSRVG_PYTHON}" \
    "${REMOTE_ADAPTER_DIR}/generate_physrvg_verified.py" \
    --physrvg-root "${REMOTE_PHYSRVG_ROOT}" \
    --model-id "${REMOTE_MODEL_ID}" \
    --dit-checkpoint "${REMOTE_DIT_CHECKPOINT}" \
    --lora-checkpoint "${REMOTE_LORA_CHECKPOINT}" \
    --input-list "${REMOTE_INPUT_LIST}" \
    --output-root "${REMOTE_OUTPUT_ROOT}" \
    --run-name "${RUN_NAME}" \
    --device cuda:0 \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --condition-fps "${CONDITION_FPS}" \
    --condition-frames "${CONDITION_FRAMES}" \
    --model-context-frames "${MODEL_CONTEXT_FRAMES}" \
    --model-chunk-frames "${MODEL_CHUNK_FRAMES}" \
    --clean-prefix-frames "${CLEAN_PREFIX_FRAMES}" \
    --model-fps "${MODEL_FPS}" \
    --target-fps "${TARGET_FPS}" \
    --target-frames "${TARGET_FRAMES}" \
    --num-inference-steps "${NUM_INFERENCE_STEPS}" \
    --guidance-scale "${GUIDANCE_SCALE}" \
    --seed "${SEED}" \
    --shard-index "${shard_index}" \
    --num-shards 2
}

echo "[$(date -Is)] generation_start run=${RUN_NAME} gpu6=shard0 gpu7=shard1"
generate_shard 6 0 >"${REMOTE_LOG_ROOT}/${RUN_NAME}_gpu6.log" 2>&1 &
pid6=$!
generate_shard 7 1 >"${REMOTE_LOG_ROOT}/${RUN_NAME}_gpu7.log" 2>&1 &
pid7=$!

status=0
wait "${pid6}" || status=$?
wait "${pid7}" || status=$?
if [[ "${status}" -ne 0 ]]; then
  echo "[$(date -Is)] generation_failed status=${status}" >&2
  exit "${status}"
fi
echo "[$(date -Is)] generation_complete"

"${REMOTE_PHYSRVG_PYTHON}" - "${REMOTE_INPUT_LIST}" "${RUN_DIR}" <<'PY'
import json
import math
import sys
from pathlib import Path

import imageio.v2 as imageio

input_list = Path(sys.argv[1])
run_dir = Path(sys.argv[2])
declared = [Path(line.strip()) for line in input_list.read_text().splitlines() if line.strip()]
expected = set()
for path in declared:
    case_path = path if path.exists() else input_list.parent / "jsons" / path.name
    expected.add(json.loads(case_path.read_text())["generated_video_name"])
actual = {path.name for path in run_dir.glob("*.mp4")}
if len(expected) != 198 or actual != expected:
    raise RuntimeError(
        f"submission set mismatch: expected={len(expected)} actual={len(actual)} "
        f"missing={len(expected-actual)} extra={len(actual-expected)}"
    )
for index, name in enumerate(sorted(expected), start=1):
    with imageio.get_reader(str(run_dir / name), format="FFMPEG") as reader:
        fps = float(reader.get_meta_data()["fps"])
        frames = int(reader.count_frames())
    if frames != 120 or not math.isclose(fps, 24.0, abs_tol=0.01):
        raise RuntimeError(f"invalid submission video {name}: {frames} frames @ {fps} FPS")
    if index % 25 == 0 or index == 198:
        print(f"validated={index}/198", flush=True)
print("submission_validation=PASS")
PY

cd "${REMOTE_PHYSIQ_ROOT}"
"${REMOTE_EVAL_PYTHON}" physiq/run_physics_iq.py \
  --input_folders "${RUN_DIR}" \
  --output_folder "${REMOTE_EVAL_ROOT}" \
  --descriptions_file "${REMOTE_DESCRIPTIONS_FILE}" \
  --benchmark_base_folder "${REMOTE_BENCHMARK_BASE}"

RESULT_CSV="${REMOTE_EVAL_ROOT}/physics-IQ-benchmark-verified/results/${RUN_NAME}.csv"
"${REMOTE_EVAL_PYTHON}" physiq/aggregate_runs_from_csvs.py \
  "${RESULT_CSV}" --score-type verified
echo "[$(date -Is)] evaluation_complete result=${RESULT_CSV}"

