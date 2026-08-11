#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 CONFIG.env" >&2
  exit 2
fi
CONFIG="$(realpath "$1")"
# shellcheck source=/dev/null
source "$CONFIG"

die() {
  echo "[error] $*" >&2
  exit 1
}

for value in \
  "$REMOTE_PYTHON" \
  "$REMOTE_FFMPEG" \
  "$REMOTE_FFPROBE" \
  "$REMOTE_CHECKPOINT_DIR/checkpoint.safetensors" \
  "$REMOTE_EXPERIMENT_CONFIG" \
  "$REMOTE_PRETRAINED_LORA" \
  "$REMOTE_WAN_ROOT" \
  "$REMOTE_SHARED_INPUT_LIST" \
  "$REMOTE_DESCRIPTIONS_FILE" \
  "$REMOTE_EVAL_PYTHON"; do
  [[ -e "$value" ]] || die "required path is missing: $value"
done
[[ "$PROMPT_SETTING" == "bpp" ]] || die "P0 requires PROMPT_SETTING=bpp"
[[ "$INPUT_MODE" == "v2v" ]] || die "P0 requires INPUT_MODE=v2v"
[[ "$HEIGHT" == 512 && "$WIDTH" == 896 ]] || die "P0 requires 512x896"
[[ "$CONDITION_FRAMES" == 72 && "$CONDITION_FPS" == 24 ]] || \
  die "P0 requires a 72-frame, 24 FPS condition"
[[ "$MODEL_FRAMES" == 189 && "$MODEL_FPS" == 24 ]] || \
  die "P0 requires a 189-frame, 24 FPS raw output"
[[ "$SUBMISSION_PREFIX_FRAMES" == 69 && "$SUBMISSION_FRAMES" == 120 ]] || \
  die "P0 requires dropping 69 frames and submitting 120"
[[ "$NUM_INFERENCE_STEPS" == 40 && "$SEED" == 42 ]] || \
  die "P0 run_01 requires 40 steps and seed 42"

mkdir -p \
  "$REMOTE_LOG_DIR" \
  "$REMOTE_WORK_DIR" \
  "$REMOTE_RAW_RUN_DIR" \
  "$REMOTE_SUBMISSION_DIR" \
  "$REMOTE_EVAL_ROOT" \
  "$(dirname "$REMOTE_GPU_LOCK")"

if find "$REMOTE_RAW_RUN_DIR" "$REMOTE_SUBMISSION_DIR" -maxdepth 1 \
  -type f -name '*.mp4' -print -quit | grep -q .; then
  die "run output already exists; refusing to overwrite: $RUN_NAME"
fi

"$REMOTE_PYTHON" "$REMOTE_INPUT_PREPARER" \
  --source-list "$REMOTE_SHARED_INPUT_LIST" \
  --output-root "$REMOTE_MAPPED_INPUT_ROOT" \
  --descriptions-file "$REMOTE_DESCRIPTIONS_FILE" \
  --ffprobe "$REMOTE_FFPROBE" \
  --expected-cases "$EXPECTED_CASES"

SHARD_LIST_ROOT="$REMOTE_WORK_DIR/input_shards"
mkdir -p "$SHARD_LIST_ROOT"
"$REMOTE_PYTHON" - "$REMOTE_INPUT_LIST" "$SHARD_LIST_ROOT" <<'PY'
import sys
from pathlib import Path

source = Path(sys.argv[1])
output = Path(sys.argv[2])
paths = [line for line in source.read_text().splitlines() if line.strip()]
if len(paths) != 198:
    raise RuntimeError(f"expected 198 mapped inputs, found {len(paths)}")
for shard in range(2):
    selected = paths[shard::2]
    if len(selected) != 99:
        raise RuntimeError(f"shard {shard} has {len(selected)} cases")
    (output / f"shard_{shard}.txt").write_text(
        "".join(f"{path}\n" for path in selected)
    )
print("input_shards=PASS shard0=99 shard1=99")
PY

exec 9>"$REMOTE_GPU_LOCK"
flock 9
while :; do
  busy=()
  for gpu in "$GPU0" "$GPU1"; do
    pids="$(nvidia-smi -i "$gpu" --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')"
    [[ -z "$pids" ]] || busy+=("gpu${gpu}:$pids")
  done
  if ((${#busy[@]} == 0)); then
    echo "[$(date -Is)] GPUs $GPU0,$GPU1 are free; starting $RUN_NAME"
    break
  fi
  echo "[$(date -Is)] waiting for GPUs $GPU0,$GPU1 (${busy[*]})"
  sleep 60
done

INFER_SCRIPT="$REMOTE_RUNTIME_EXPERIMENT_ROOT/infer_xssc_object_self_attn_lora.py"
[[ -f "$INFER_SCRIPT" ]] || die "isolated inference entry is missing: $INFER_SCRIPT"

generate_shard() {
  local physical_gpu="$1"
  local shard_index="$2"
  local shard_root="$REMOTE_WORK_DIR/raw_shards/gpu${physical_gpu}"
  local shard_list="$SHARD_LIST_ROOT/shard_${shard_index}.txt"
  mkdir -p "$shard_root"
  local -a command=(
    "$REMOTE_PYTHON" "$INFER_SCRIPT"
    --weights-root "$REMOTE_CHECKPOINT_DIR"
    --input-json-list-path "$shard_list"
    --model-name "$MODEL_NAME"
    --output-root "$shard_root"
    --step-output-dir-name "$RUN_NAME"
    --shard-tag "${RUN_NAME}-shard${shard_index}"
    --wan-root "$REMOTE_WAN_ROOT"
    --lora-checkpoint "$REMOTE_PRETRAINED_LORA"
    --device cuda:0
    --aux-device cuda:0
    --inference-devices cuda:0,cuda:0
    --height "$HEIGHT"
    --width "$WIDTH"
    --num-frames "$MODEL_FRAMES"
    --context-frames "$CONDITION_FRAMES"
    --fps "$MODEL_FPS"
    --sampling-mode prefix
    --num-inference-steps "$NUM_INFERENCE_STEPS"
    --seed "$SEED"
    --negative-prompt "$NEGATIVE_PROMPT"
  )
  printf '[%s] gpu=%s shard=%s command:' "$(date -Is)" \
    "$physical_gpu" "$shard_index"
  printf ' %q' "${command[@]}"
  printf '\n'
  env \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH="$REMOTE_PROJECT_ROOT:$REMOTE_RUNTIME_TRAIN_ROOT:$REMOTE_RUNTIME_EXPERIMENT_ROOT:$REMOTE_PROJECT_ROOT/code_vjepa_vggt/train_xSSC:$REMOTE_DIFFSYNTH_ROOT" \
    CUDA_VISIBLE_DEVICES="$physical_gpu" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    EXPERIMENT_CONFIG="$REMOTE_EXPERIMENT_CONFIG" \
    "${command[@]}"
}

echo "[$(date -Is)] generation_start gpu${GPU0}=shard0 gpu${GPU1}=shard1"
generate_shard "$GPU0" 0 >"$REMOTE_LOG_DIR/gpu${GPU0}.log" 2>&1 &
pid0=$!
generate_shard "$GPU1" 1 >"$REMOTE_LOG_DIR/gpu${GPU1}.log" 2>&1 &
pid1=$!
status=0
wait "$pid0" || status=$?
wait "$pid1" || status=$?
((status == 0)) || die "generation failed with status $status"
echo "[$(date -Is)] generation_complete"

for gpu in "$GPU0" "$GPU1"; do
  shard_run="$REMOTE_WORK_DIR/raw_shards/gpu${gpu}/$RUN_NAME"
  [[ -d "$shard_run" ]] || die "missing raw shard folder: $shard_run"
  cp -a "$shard_run/." "$REMOTE_RAW_RUN_DIR/"
done

"$REMOTE_PYTHON" "$REMOTE_OUTPUT_PREPARER" \
  --raw-folder "$REMOTE_RAW_RUN_DIR" \
  --input-list "$REMOTE_INPUT_LIST" \
  --output-folder "$REMOTE_SUBMISSION_DIR"

"$REMOTE_PYTHON" - \
  "$REMOTE_INPUT_LIST" \
  "$REMOTE_RAW_RUN_DIR" \
  "$REMOTE_SUBMISSION_DIR" \
  "$REMOTE_FFPROBE" <<'PY'
import json
import math
import subprocess
import sys
from pathlib import Path

input_list, raw_dir, submission_dir, ffprobe = map(Path, sys.argv[1:])
declared = [Path(line) for line in input_list.read_text().splitlines() if line]
expected = {json.loads(path.read_text())["generated_video_name"] for path in declared}

def probe(path):
    payload = json.loads(subprocess.check_output([
        str(ffprobe), "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,nb_read_frames:format=duration",
        "-of", "json", str(path),
    ], text=True))
    stream = payload["streams"][0]
    n, d = stream["avg_frame_rate"].split("/", 1)
    return int(stream["nb_read_frames"]), float(n) / float(d), float(payload["format"]["duration"])

for folder, frames, duration in ((raw_dir, 189, 7.875), (submission_dir, 120, 5.0)):
    actual = {path.name for path in folder.glob("*.mp4")}
    if actual != expected:
        raise RuntimeError(
            f"{folder}: expected={len(expected)} actual={len(actual)} "
            f"missing={len(expected-actual)} extra={len(actual-expected)}"
        )
    for name in sorted(expected):
        got_frames, got_fps, got_duration = probe(folder / name)
        if got_frames != frames or not math.isclose(got_fps, 24.0, abs_tol=0.01):
            raise RuntimeError(f"invalid {folder / name}: {got_frames} @ {got_fps}")
        if not math.isclose(got_duration, duration, abs_tol=0.01):
            raise RuntimeError(f"invalid duration {folder / name}: {got_duration}")
print("raw_and_submission_validation=PASS cases=198 raw=189 submission=120 fps=24")
PY

STAGED_RUN="$REMOTE_WORK_DIR/eval_staged/$RUN_NAME"
mkdir -p "$STAGED_RUN"
for video in "$REMOTE_SUBMISSION_DIR"/*.mp4; do
  ln -sfn "$(readlink -f "$video")" "$STAGED_RUN/$(basename "$video")"
done

cd "$REMOTE_PHYSIQ_ROOT"
"$REMOTE_EVAL_PYTHON" physiq/run_physics_iq.py \
  --input_folders "$STAGED_RUN" \
  --output_folder "$REMOTE_EVAL_ROOT" \
  --descriptions_file "$REMOTE_DESCRIPTIONS_FILE" \
  --benchmark_base_folder "$REMOTE_BENCHMARK_BASE" \
  --n_process "$N_PROCESS"

RESULT_CSV="$REMOTE_EVAL_ROOT/physics-IQ-benchmark-verified/results/$RUN_NAME.csv"
[[ -s "$RESULT_CSV" ]] || die "official result CSV is missing: $RESULT_CSV"
"$REMOTE_EVAL_PYTHON" physiq/aggregate_runs_from_csvs.py \
  "$RESULT_CSV" \
  --score-type verified \
  --save-csv "$REMOTE_EVAL_ROOT/${RUN_NAME}_verified_summary.csv" \
  --model-name "$MODEL_NAME" \
  | tee "$REMOTE_LOG_DIR/official_score.log"
echo "[$(date -Is)] evaluation_complete result=$RESULT_CSV"
