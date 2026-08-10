#!/usr/bin/env bash
set -euo pipefail




SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

WAN_PYTHON="${WAN_PYTHON:-/data/gaoya/miniconda3/envs/wan/bin/python}"
SAM_PYTHON="${SAM_PYTHON:-/data/gaoya/miniconda3/envs/physxnet_mpm_env/bin/python}"
VBENCH_PYTHON="${VBENCH_PYTHON:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
VBENCH_DRIVER="${VBENCH_DRIVER:-/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/bench.py}"
OUTPUT_BASE="${METRICS_OUTPUT_BASE:-/data/gaoya/agent-data/outputs/object_query_ablation_metrics}"
REGION_CACHE_BASE="${METRICS_REGION_CACHE_BASE:-/data/gaoya/agent-data/cache/wan22_ti2v_legacy_firstlatent_regions_704x1280}"
GPU="${GPU:-0}"
DRY_RUN=0
OVERWRITE=0
RUN_VBENCH=1
RUN_AGGREGATE=1
RUN_HEAD_SCOPE_BASELINE=0
HEAD_SCOPE_WATCH_SECONDS=0
HEAD_SCOPE_WORKERS=4

usage() {
  cat <<'EOF'
Usage:
  bash bench.sh RESULT_DIR [options]

RESULT_DIR may be either:
  1. seed_XXXXX/ containing video_similarity_top100.json; or
  2. its case directory containing one or more seed_XXXXX/ directories.

Options:
  --gpu ID          Physical GPU index (default: $GPU or 0; GPU 4 is forbidden)
  --output-base DIR Metric/cache output base (default: /data/gaoya/agent-data/outputs/object_query_ablation_metrics)
  --overwrite       Recompute cached tracks, masks, RAFT, perceptual data and overlays
  --skip-vbench     Do not run/complete the seven official VBench dimensions
  --no-aggregate    Do not rebuild the strict common-seed aggregate report
  --head-scope-baseline
                    Incrementally compute CPU baseline-effect metrics for the
                    generated M1/M2/M3 Top100/Bottom100/All-Heads videos
  --watch-seconds N With --head-scope-baseline, poll for new videos every N sec
  --workers N       CPU decode/metric workers for head-scope mode (default: 4)
  --dry-run         Validate and print every command without running model inference
  -h, --help        Show this help

Examples:
  bash bench.sh /path/to/CASE/seed_47326
  GPU=5 bash bench.sh /path/to/CASE
  bash bench.sh /path/to/temporal_tube_root --head-scope-baseline --watch-seconds 60
EOF
}

die() {
  echo "[bench:error] $*" >&2
  exit 2
}

quote_command() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
}

run() {
  quote_command "$@"
  if (( DRY_RUN == 0 )); then
    "$@"
  fi
}

[[ $# -gt 0 ]] || { usage >&2; exit 2; }
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  exit 0
fi
RESULT_INPUT="$1"
shift
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      [[ $# -ge 2 ]] || die "--gpu requires an ID"
      GPU="$2"
      shift 2
      ;;
    --output-base)
      [[ $# -ge 2 ]] || die "--output-base requires a directory"
      OUTPUT_BASE="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --skip-vbench)
      RUN_VBENCH=0
      shift
      ;;
    --no-aggregate)
      RUN_AGGREGATE=0
      shift
      ;;
    --head-scope-baseline)
      RUN_HEAD_SCOPE_BASELINE=1
      shift
      ;;
    --watch-seconds)
      [[ $# -ge 2 ]] || die "--watch-seconds requires a non-negative integer"
      HEAD_SCOPE_WATCH_SECONDS="$2"
      shift 2
      ;;
    --workers)
      [[ $# -ge 2 ]] || die "--workers requires a positive integer"
      HEAD_SCOPE_WORKERS="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

if (( RUN_HEAD_SCOPE_BASELINE == 1 )); then
  [[ "${HEAD_SCOPE_WATCH_SECONDS}" =~ ^[0-9]+$ ]] || die \
    "--watch-seconds must be a non-negative integer"
  [[ "${HEAD_SCOPE_WORKERS}" =~ ^[1-9][0-9]*$ ]] || die \
    "--workers must be a positive integer"
  [[ -x "${WAN_PYTHON}" ]] || die "missing Python executable: ${WAN_PYTHON}"
  RESULT_INPUT="$(realpath -e -- "${RESULT_INPUT}")"
  [[ -d "${RESULT_INPUT}" ]] || die "RESULT_DIR is not a directory: ${RESULT_INPUT}"
  HEAD_SCOPE_OUTPUT_BASE="${METRICS_HEAD_SCOPE_OUTPUT_BASE:-${OUTPUT_BASE}/head_scope_baseline_fast}"
  case "${HEAD_SCOPE_OUTPUT_BASE}" in
    /home/gaoya|/home/gaoya/*)
      die "large metric artifacts may not be stored under /home/gaoya"
      ;;
  esac
  declare -a head_scope_command=(
    "${WAN_PYTHON}"
    "${SCRIPT_DIR}/compute_head_scope_baseline_metrics.py"
    "${RESULT_INPUT}"
    --output-base "${HEAD_SCOPE_OUTPUT_BASE}"
    --workers "${HEAD_SCOPE_WORKERS}"
    --watch-seconds "${HEAD_SCOPE_WATCH_SECONDS}"
  )
  quote_command "${head_scope_command[@]}"
  if (( DRY_RUN == 0 )); then
    exec "${head_scope_command[@]}"
  fi
  exit 0
fi

[[ "${GPU}" =~ ^[0-9]+$ ]] || die "GPU must be one physical GPU index"
[[ "${GPU}" != "4" ]] || die "GPU 4 is forbidden by /home/gaoya/AGENTS.md"
command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
nvidia-smi -i "${GPU}" --query-gpu=index --format=csv,noheader \
  >/dev/null 2>&1 || die "GPU ${GPU} does not exist"
for executable in "${WAN_PYTHON}" "${SAM_PYTHON}" "${VBENCH_PYTHON}"; do
  [[ -x "${executable}" ]] || die "missing Python executable: ${executable}"
done
[[ -f "${VBENCH_DRIVER}" ]] || die "missing VBench driver: ${VBENCH_DRIVER}"

RESULT_INPUT="$(realpath -e -- "${RESULT_INPUT}")"
[[ -d "${RESULT_INPUT}" ]] || die "RESULT_DIR is not a directory: ${RESULT_INPUT}"
OUTPUT_BASE="$(realpath -m -- "${OUTPUT_BASE}")"
case "${OUTPUT_BASE}" in
  /home/gaoya|/home/gaoya/*)
    die "large metric artifacts may not be stored under /home/gaoya; use /data/gaoya/agent-data"
    ;;
esac

declare -a INVENTORIES=()
if [[ -f "${RESULT_INPUT}/video_similarity_top100.json" ]]; then
  INVENTORIES+=("${RESULT_INPUT}/video_similarity_top100.json")
else
  while IFS= read -r inventory; do
    INVENTORIES+=("${inventory}")
  done < <(
    find "${RESULT_INPUT}" -mindepth 2 -maxdepth 2 -type f \
      -path '*/seed_*/video_similarity_top100.json' -print | sort
  )
fi
[[ ${#INVENTORIES[@]} -gt 0 ]] || die \
  "no video_similarity_top100.json found in ${RESULT_INPUT} or its direct seed_* children"

export CUDA_VISIBLE_DEVICES="${GPU}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/Code_data/Code_try0526${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

declare -a COMPLETED_CASE_ROOTS=()

run_one_seed() {
  local inventory="$1"
  local result_dir
  result_dir="$(dirname -- "${inventory}")"
  local -a identity
  mapfile -t identity < <("${WAN_PYTHON}" - "${inventory}" <<'PY'
import json
import sys
from pathlib import Path

inventory_path = Path(sys.argv[1]).resolve()
payload = json.loads(inventory_path.read_text(encoding="utf-8"))
case = str(payload.get("case") or "")
seed = int(payload.get("seed", -1))
videos = payload.get("videos")
if not case or seed < 0:
    raise RuntimeError(f"invalid case/seed identity: {inventory_path}")
if not isinstance(videos, list) or len(videos) != 49:
    raise RuntimeError(f"expected baseline + 48 ablations, got {len(videos or [])}")
if videos[0].get("id") != "baseline":
    raise RuntimeError("first inventory record must be baseline")
ids = [str(row.get("id")) for row in videos]
if len(set(ids)) != 49:
    raise RuntimeError("inventory video IDs are not unique")
for row in videos:
    path = Path(str(row.get("path") or "")).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
baseline = Path(videos[0]["path"]).expanduser().resolve()
manifest_path = baseline.parent / "manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
input_json = Path(str(manifest.get("input_json") or "")).expanduser().resolve()
source_payload = json.loads(input_json.read_text(encoding="utf-8"))
source_video = Path(str(source_payload.get("source_video") or "")).expanduser().resolve()
if not source_video.is_file():
    raise FileNotFoundError(source_video)
source_root = source_video.parent
if not (source_root / "states.npz").is_file():
    raise FileNotFoundError(source_root / "states.npz")
print(case)
print(seed)
print(source_root)
print(input_json)
print(baseline)
PY
  )
  [[ ${#identity[@]} -eq 5 ]] || die "failed to resolve inventory identity: ${inventory}"
  local case_name="${identity[0]}"
  local seed="${identity[1]}"
  local source_root="${METRICS_SOURCE_ROOT:-${identity[2]}}"
  local input_json="${identity[3]}"
  local baseline_video="${identity[4]}"
  local region_cache="${METRICS_REGION_CACHE:-${REGION_CACHE_BASE}/${case_name}}"
  local output_root="${OUTPUT_BASE}/${case_name}/$(printf 'seed_%05d' "${seed}")"
  local case_output_root="${OUTPUT_BASE}/${case_name}"
  local raft_root="${result_dir}/raft_motion_top100_v1"
  local vbench_root="${output_root}/vbench"

  [[ -f "${result_dir}/frozen_baseline_tracks/tracks.npz" ]] || die \
    "missing frozen baseline tracks: ${result_dir}/frozen_baseline_tracks/tracks.npz"
  [[ -f "${region_cache}/regions.npz" ]] || die \
    "missing object region cache: ${region_cache}/regions.npz"
  [[ -f "${source_root}/source_video.mp4" ]] || die \
    "missing source video: ${source_root}/source_video.mp4"
  [[ -f "${source_root}/states.npz" ]] || die \
    "missing simulator states: ${source_root}/states.npz"

  echo
  echo "[bench:case] case=${case_name} seed=${seed} videos=49 gpu=${GPU}"
  echo "[bench:input] ${result_dir}"
  echo "[bench:baseline] ${baseline_video}"
  echo "[bench:source-json] ${input_json}"
  echo "[bench:output] ${output_root}"

  export OBJECT_QUERY_ABLATION_CASE="${case_name}"
  export OBJECT_QUERY_ABLATION_SEED="${seed}"
  export OBJECT_QUERY_ABLATION_RESULT_DIR="${result_dir}"
  export OBJECT_QUERY_ABLATION_INVENTORY="${inventory}"
  export OBJECT_QUERY_ABLATION_SOURCE_ROOT="${source_root}"
  export OBJECT_QUERY_ABLATION_REGION_CACHE="${region_cache}"
  export OBJECT_QUERY_ABLATION_OUTPUT_ROOT="${output_root}"

  if (( RUN_VBENCH == 1 )); then
    run "${WAN_PYTHON}" "${SCRIPT_DIR}/prepare_multiseed_vbench.py" \
      --result-dir "${result_dir}" --output-root "${vbench_root}"
    local metric
    local -a vbench_metrics=(
      vbench_subject_consistency
      vbench_background_consistency
      vbench_temporal_flickering
      vbench_motion_smoothness
      vbench_dynamic_degree
      vbench_aesthetic_quality
      vbench_imaging_quality
    )
    for metric in "${vbench_metrics[@]}"; do
      run env PYTHONNOUSERSITE=1 "${VBENCH_PYTHON}" "${VBENCH_DRIVER}" \
        --metric "${metric}" \
        --result-root "${vbench_root}/index" \
        --output-summary "${vbench_root}/eval_summary_${metric}.json" \
        --vbench-output-root "${vbench_root}/raw" \
        --vbench-device cuda
    done
  else
    echo "[bench:skip] official VBench completion"
  fi

  local -a overwrite_args=()
  local -a overlay_args=()
  if (( OVERWRITE == 1 )); then
    overwrite_args+=(--overwrite)
    overlay_args+=(--overwrite-overlays)
  fi

  run "${WAN_PYTHON}" "${SCRIPT_DIR}/extract_tracks.py" \
    --device cuda "${overwrite_args[@]}"
  run "${SAM_PYTHON}" "${SCRIPT_DIR}/extract_masks.py" \
    --device cuda "${overwrite_args[@]}"
  run "${WAN_PYTHON}" \
    "${REPO_ROOT}/AAA_my_test/analyze_legacy_ti2v_object_ablation_raft_motion.py" \
    --case "${case_name}" --seed "${seed}" --device cuda \
    --inventory "${inventory}" --output-root "${raft_root}" \
    --skip-flow-videos --flows-only "${overwrite_args[@]}"
  run "${WAN_PYTHON}" "${SCRIPT_DIR}/extract_source_raft.py" \
    --device cuda "${overwrite_args[@]}"
  run "${SAM_PYTHON}" "${SCRIPT_DIR}/compute_perceptual.py" \
    --device cuda "${overwrite_args[@]}"
  run "${WAN_PYTHON}" "${SCRIPT_DIR}/compute_metrics_and_overlays.py" \
    "${overlay_args[@]}"
  local -a validate_args=()
  if (( RUN_VBENCH == 1 )); then
    validate_args+=(--require-vbench)
  fi
  run "${WAN_PYTHON}" "${SCRIPT_DIR}/validate_outputs.py" \
    "${validate_args[@]}"

  COMPLETED_CASE_ROOTS+=("${case_output_root}")
  echo "[bench:seed-done] report=${output_root}/report.json"
}

cd "${REPO_ROOT}"
echo "[bench:start] inventories=${#INVENTORIES[@]} gpu=${GPU} dry_run=${DRY_RUN} overwrite=${OVERWRITE}"
for inventory in "${INVENTORIES[@]}"; do
  run_one_seed "${inventory}"
done

if (( RUN_AGGREGATE == 1 )); then
  mapfile -t unique_case_roots < <(printf '%s\n' "${COMPLETED_CASE_ROOTS[@]}" | sort -u)
  for case_root in "${unique_case_roots[@]}"; do
    declare -a seeds=()
    while IFS= read -r report; do
      seed_name="$(basename -- "$(dirname -- "${report}")")"
      seeds+=("$((10#${seed_name#seed_}))")
    done < <(find "${case_root}" -mindepth 2 -maxdepth 2 -type f \
      -path '*/seed_*/report.json' -print | sort)
    if [[ ${#seeds[@]} -eq 0 ]]; then
      if (( DRY_RUN == 1 )); then
        echo "[bench:aggregate] no pre-existing reports during dry-run; aggregation will use newly completed seed reports"
        continue
      fi
      die "no completed seed reports found under ${case_root}"
    fi
    representative="${seeds[0]}"
    for seed in "${seeds[@]}"; do
      if [[ "${seed}" == "47326" ]]; then
        representative=47326
        break
      fi
    done
    run "${WAN_PYTHON}" "${SCRIPT_DIR}/aggregate_reports.py" \
      --root "${case_root}" --seeds "${seeds[@]}" \
      --representative-seed "${representative}"
  done
else
  echo "[bench:skip] aggregate report"
fi

echo "[bench:done] all requested metric stages completed"
