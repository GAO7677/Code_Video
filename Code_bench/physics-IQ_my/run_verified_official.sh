#!/usr/bin/env bash
set -euo pipefail

REPO="/home/gaoya/Code_Video/Code_bench/physics-IQ-benchmark-main"
DATASET="/data/gaoya/dataset/Anates-Labs-Research-Physics-IQ-Verified"
WORK_BASE="/data/gaoya/agent-data/cache/physics-iq-verified"
DESCRIPTIONS_FILE="$REPO/descriptions/best_practice/descriptions_base.csv"
OUTPUT_FOLDER=""
N_PROCESS=0
KEEP_WORKDIR=0
INPUT_FOLDERS=()

usage() {
  cat <<'EOF'
Usage:
  run_verified_official.sh --output-folder DIR [options] RUN_FOLDER [RUN_FOLDER ...]

Required:
  --output-folder DIR       Parent directory for official CSV, JSON, and PDF outputs.
  RUN_FOLDER                Folder with exactly 198 generated MP4 files.

Options:
  --descriptions-file FILE  Prompt CSV used for generation.
                            Default: official best-practice base prompts.
  --n-process N             Official metric worker count. Default: 0 (serial).
  --work-base DIR           Temporary writable area for masks and staged inputs.
  --keep-workdir            Keep temporary masks and staged symlinks after exit.
  -h, --help                Show this help.

This wrapper intentionally runs Physics-IQ Verified. It never passes
--original_physics_iq to the official evaluator.
EOF
}

die() {
  printf 'Error: %s\n' "$*" >&2
  exit 1
}

while (($#)); do
  case "$1" in
    --output-folder)
      (($# >= 2)) || die "--output-folder requires a value"
      OUTPUT_FOLDER="$2"
      shift 2
      ;;
    --descriptions-file)
      (($# >= 2)) || die "--descriptions-file requires a value"
      DESCRIPTIONS_FILE="$2"
      shift 2
      ;;
    --n-process)
      (($# >= 2)) || die "--n-process requires a value"
      N_PROCESS="$2"
      shift 2
      ;;
    --work-base)
      (($# >= 2)) || die "--work-base requires a value"
      WORK_BASE="$2"
      shift 2
      ;;
    --keep-workdir)
      KEEP_WORKDIR=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      INPUT_FOLDERS+=("$@")
      break
      ;;
    -*)
      die "unknown option: $1"
      ;;
    *)
      INPUT_FOLDERS+=("$1")
      shift
      ;;
  esac
done

[[ -n "$OUTPUT_FOLDER" ]] || die "--output-folder is required"
((${#INPUT_FOLDERS[@]} > 0)) || die "at least one RUN_FOLDER is required"
[[ "$N_PROCESS" =~ ^[0-9]+$ ]] || die "--n-process must be a non-negative integer"
[[ -d "$REPO" ]] || die "official repository not found: $REPO"
[[ -d "$DATASET" ]] || die "verified dataset not found: $DATASET"
[[ -f "$DESCRIPTIONS_FILE" ]] || die "descriptions file not found: $DESCRIPTIONS_FILE"
command -v uv >/dev/null 2>&1 || die "uv is not installed or not on PATH"
command -v ffprobe >/dev/null 2>&1 || die "ffprobe is not installed or not on PATH (install ffmpeg)"

# The official metric code is CPU-only.  Limit native thread pools when using
# its ProcessPoolExecutor so N workers do not each create a full-CPU pool.
export OPENCV_FOR_THREADS_NUM="${OPENCV_FOR_THREADS_NUM:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export BLIS_NUM_THREADS="${BLIS_NUM_THREADS:-1}"
# No official evaluator component uses CUDA; keep all GPUs hidden by default.
export CUDA_VISIBLE_DEVICES="${PHYSIQ_CUDA_VISIBLE_DEVICES:-}"

# OpenCV's FFmpeg backend ignores the native pool limits above for video
# decoding and auto-creates a large decoder pool per process.  Add a runtime-
# only sitecustomize shim that opens captures with CAP_PROP_N_THREADS=1;
# official metric source files remain unchanged.
RUNTIME_SHIM_DIR="/data/gaoya/agent-data/cache/physics-iq-verified/runtime"
[[ -f "$RUNTIME_SHIM_DIR/sitecustomize.py" ]] || \
  die "official scorer runtime shim not found: $RUNTIME_SHIM_DIR/sitecustomize.py"
if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$RUNTIME_SHIM_DIR:$PYTHONPATH"
else
  export PYTHONPATH="$RUNTIME_SHIM_DIR"
fi

mkdir -p "$OUTPUT_FOLDER" "$WORK_BASE"
OUTPUT_FOLDER="$(cd "$OUTPUT_FOLDER" && pwd)"
DESCRIPTIONS_FILE="$(readlink -f "$DESCRIPTIONS_FILE")"
WORK_DIR="$(mktemp -d "$WORK_BASE/run.XXXXXXXX")"

cleanup() {
  if ((KEEP_WORKDIR)); then
    printf 'Kept temporary work directory: %s\n' "$WORK_DIR"
  else
    rm -rf -- "$WORK_DIR"
  fi
}
trap cleanup EXIT

# Build the exact directory name expected by the official evaluator while
# keeping the downloaded dataset read-only. Only generated masks are writable.
LAYOUT_PARENT="$WORK_DIR/layout"
BENCHMARK_ROOT="$LAYOUT_PARENT/physics-IQ-benchmark-verified"
mkdir -p \
  "$BENCHMARK_ROOT/split-videos/testing" \
  "$BENCHMARK_ROOT/video-masks/real"

ln -s "$DATASET/full-videos" "$BENCHMARK_ROOT/full-videos"
ln -s "$DATASET/switch-frames" "$BENCHMARK_ROOT/switch-frames"
ln -s "$DATASET/split-videos/conditioning" \
  "$BENCHMARK_ROOT/split-videos/conditioning"

for source_dir in "$DATASET/split-videos/testing/"*FPS; do
  [[ -d "$source_dir" ]] || continue
  ln -s "$source_dir" "$BENCHMARK_ROOT/split-videos/testing/$(basename "$source_dir")"
done

for source_dir in "$DATASET/video-masks/real/"*FPS; do
  [[ -d "$source_dir" ]] || continue
  ln -s "$source_dir" "$BENCHMARK_ROOT/video-masks/real/$(basename "$source_dir")"
done

# The official runner renames generated files in place. Stage symlinks so only
# temporary directory entries are renamed, never the user's source videos.
STAGED_ROOT="$WORK_DIR/generated"
mkdir -p "$STAGED_ROOT"
STAGED_INPUTS=()
declare -A SEEN_RUN_NAMES=()

for input_folder in "${INPUT_FOLDERS[@]}"; do
  [[ -d "$input_folder" ]] || die "run folder not found: $input_folder"
  input_folder="$(cd "$input_folder" && pwd)"
  run_name="$(basename "$input_folder")"
  [[ -z "${SEEN_RUN_NAMES[$run_name]:-}" ]] || \
    die "run folder basenames must be unique: $run_name"
  SEEN_RUN_NAMES[$run_name]=1

  shopt -s nullglob
  videos=("$input_folder"/*.mp4)
  shopt -u nullglob
  ((${#videos[@]} == 198)) || \
    die "$input_folder contains ${#videos[@]} MP4 files; official evaluation requires 198"

  staged_folder="$STAGED_ROOT/$run_name"
  mkdir -p "$staged_folder"
  for video in "${videos[@]}"; do
    ln -s "$(readlink -f "$video")" "$staged_folder/$(basename "$video")"
  done
  STAGED_INPUTS+=("$staged_folder")
done

CMD=(
  uv run physiq/run_physics_iq.py
  --input_folders "${STAGED_INPUTS[@]}"
  --output_folder "$OUTPUT_FOLDER"
  --descriptions_file "$DESCRIPTIONS_FILE"
  --benchmark_base_folder "$LAYOUT_PARENT"
  --n_process "$N_PROCESS"
)

printf 'Official command:'
printf ' %q' "${CMD[@]}"
printf '\n'

cd "$REPO"
"${CMD[@]}"

printf 'Official results: %s\n' \
  "$OUTPUT_FOLDER/physics-IQ-benchmark-verified/results"
