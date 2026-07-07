#!/usr/bin/env bash
set -euo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODE="shard"
REPO_ID="PhysInOneP01/PhysInOneP01"
LOCAL_DIR=""
HF_ENDPOINT_VALUE="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_TOKEN_VALUE="${HF_TOKEN:-}"
MAX_WORKERS="${MAX_WORKERS:-8}"
DRY_RUN=0
FORCE_DOWNLOAD=0
INCLUDE_PATTERNS=()
EXCLUDE_PATTERNS=()

usage() {
  cat <<'EOF'
Usage:
  download_physinone_from_hf.sh [options]

Modes:
  --mode shard
      Download one full PhysInOne shard repo with `hf download`.
      Default repo: PhysInOneP01/PhysInOneP01

  --mode assets
      Download the official PhysInOne `assets/` folder from vLAR/PhysInOne.
      This follows the dataset card's recommended workflow:
        1) download assets/
        2) run assets/scripts/filter_cases.py
        3) run assets/scripts/download.py

Options:
  --repo-id REPO_ID
      Repo id used in shard mode, e.g. PhysInOneP01/PhysInOneP01.

  --local-dir PATH
      Target local directory. If omitted:
        shard mode -> /data/gaoya/dataset/vLAR-PhysInOne/<repo-id with slash replaced by dash>
        assets mode -> /data/gaoya/dataset/vLAR-PhysInOne/vLAR-PhysInOne-assets

  --include GLOB
      Optional include pattern passed through to `hf download`.
      Can be specified multiple times.

  --exclude GLOB
      Optional exclude pattern passed through to `hf download`.
      Can be specified multiple times.

  --hf-endpoint URL
      Hugging Face endpoint. Defaults to https://hf-mirror.com

  --hf-token TOKEN
      Optional Hugging Face token. If omitted, falls back to $HF_TOKEN.

  --max-workers N
      Passed to `hf download --max-workers`. Default: 8

  --dry-run
      Only print the resolved command.

  --force-download
      Force re-download even if cached.

Examples:
  # Download official assets/ folder for case filtering + selective download
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_physinone_no_gt_box/download_physinone_from_hf.sh \
    --mode assets

  # Download the full P01 shard into the path used by the current training scripts
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_physinone_no_gt_box/download_physinone_from_hf.sh \
    --mode shard \
    --repo-id PhysInOneP01/PhysInOneP01 \
    --local-dir /data/gaoya/dataset/vLAR-PhysInOne/PhysInOneP01-PhysInOneP01

  # Download only assets plus then manually filter/download cases
  python /data/gaoya/dataset/vLAR-PhysInOne/vLAR-PhysInOne-assets/assets/scripts/filter_cases.py \
    --split train \
    --activity_type double \
    --phenomena FrictionStop \
    --num 100 \
    --output selected_cases.json

  python /data/gaoya/dataset/vLAR-PhysInOne/vLAR-PhysInOne-assets/assets/scripts/download.py \
    --selection selected_cases.json \
    --output_dir /data/gaoya/dataset/vLAR-PhysInOne/selected_cases
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --repo-id)
      REPO_ID="$2"
      shift 2
      ;;
    --local-dir)
      LOCAL_DIR="$2"
      shift 2
      ;;
    --include)
      INCLUDE_PATTERNS+=("$2")
      shift 2
      ;;
    --exclude)
      EXCLUDE_PATTERNS+=("$2")
      shift 2
      ;;
    --hf-endpoint)
      HF_ENDPOINT_VALUE="$2"
      shift 2
      ;;
    --hf-token)
      HF_TOKEN_VALUE="$2"
      shift 2
      ;;
    --max-workers)
      MAX_WORKERS="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --force-download)
      FORCE_DOWNLOAD=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v hf >/dev/null 2>&1; then
  echo "Missing required command: hf" >&2
  echo "Install huggingface_hub CLI first, e.g. pip install -U huggingface_hub" >&2
  exit 127
fi

case "$MODE" in
  assets)
    DATASET_ID="vLAR/PhysInOne"
    if [[ -z "$LOCAL_DIR" ]]; then
      LOCAL_DIR="/data/gaoya/dataset/vLAR-PhysInOne/vLAR-PhysInOne-assets"
    fi
    INCLUDE_PATTERNS=("assets/*" "${INCLUDE_PATTERNS[@]}")
    ;;
  shard)
    DATASET_ID="$REPO_ID"
    if [[ -z "$LOCAL_DIR" ]]; then
      LOCAL_DIR="/data/gaoya/dataset/vLAR-PhysInOne/${REPO_ID//\//-}"
    fi
    ;;
  *)
    echo "Unsupported mode: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac

mkdir -p "$LOCAL_DIR"

CMD=(
  env
  HF_ENDPOINT="$HF_ENDPOINT_VALUE"
  hf download "$DATASET_ID"
  --repo-type dataset
  --local-dir "$LOCAL_DIR"
  --max-workers "$MAX_WORKERS"
)

if [[ -n "$HF_TOKEN_VALUE" ]]; then
  CMD+=(--token "$HF_TOKEN_VALUE")
fi

if [[ "$FORCE_DOWNLOAD" -eq 1 ]]; then
  CMD+=(--force-download)
fi

for pattern in "${INCLUDE_PATTERNS[@]}"; do
  [[ -n "$pattern" ]] && CMD+=(--include "$pattern")
done

for pattern in "${EXCLUDE_PATTERNS[@]}"; do
  [[ -n "$pattern" ]] && CMD+=(--exclude "$pattern")
done

printf 'Resolved command:\n'
printf '  %q' "${CMD[@]}"
printf '\n'

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

"${CMD[@]}"

cat <<EOF

Download finished.
mode:       $MODE
dataset id: $DATASET_ID
local dir:  $LOCAL_DIR

EOF

if [[ "$MODE" == "assets" ]]; then
  cat <<EOF
Official next steps from the PhysInOne dataset card:

1. Filter desired cases:
   python "$LOCAL_DIR/assets/scripts/filter_cases.py" \\
     --split train \\
     --activity_type double \\
     --phenomena FrictionStop \\
     --num 100 \\
     --output "$LOCAL_DIR/selected_cases.json"

2. Download selected cases:
   python "$LOCAL_DIR/assets/scripts/download.py" \\
     --selection "$LOCAL_DIR/selected_cases.json" \\
     --output_dir "/data/gaoya/dataset/vLAR-PhysInOne/selected_cases"

EOF
fi

if [[ "$MODE" == "shard" ]]; then
  cat <<EOF
This mode downloads a whole shard repo directly. That is useful for the current
train0705 PhysInOne adapter, which already points to a root like:

  /data/gaoya/dataset/vLAR-PhysInOne/PhysInOneP01-PhysInOneP01

EOF
fi
