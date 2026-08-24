#!/usr/bin/env bash
set -euo pipefail

HF_BIN="/home/gaoya/miniconda3/envs/wan-cu128/bin/hf"
PYTHON_BIN="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
TARGET="/data/gaoya/dataset/vLAR-PhysInOne"
FILTER_SCRIPT="$TARGET/assets/scripts/filter_cases.py"
ASSIGNMENT="$TARGET/assets/metadata/repo_assignment.txt"
SELECTION_DIR="$TARGET/selection"

mkdir -p "$TARGET" "$SELECTION_DIR"

# Read the existing token without printing it into the download log.
HF_TOKEN_VALUE="$(awk -F"'" '/HF_TOKEN=/{print $2; exit}' /home/gaoya/z-note/ssh.md)"
if [[ -z "$HF_TOKEN_VALUE" ]]; then
    echo "HF_TOKEN was not found in /home/gaoya/z-note/ssh.md" >&2
    exit 2
fi

download() {
    env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
        -u all_proxy -u ALL_PROXY \
        HF_ENDPOINT=https://hf-mirror.com \
        HF_TOKEN="$HF_TOKEN_VALUE" \
        "$HF_BIN" "$@"
}

echo "[$(date -u +%FT%TZ)] downloading filter script and assignment index"
download download vLAR/PhysInOne \
    assets/metadata/repo_assignment.txt \
    --repo-type dataset \
    --local-dir "$TARGET"
download download vLAR/PhysInOne \
    assets/scripts/filter_cases.py \
    --repo-type dataset \
    --local-dir "$TARGET"

phenomena=(
    MovingHitsFixed
    MovingHitsStationary
    MovingHitsMoving
    ObliqueProjectile
    VerticalFall
    RollDownSlope
    RollUpSlope
    FrictionStop
)

for phenomenon in "${phenomena[@]}"; do
    "$PYTHON_BIN" "$FILTER_SCRIPT" \
        --assignment_file "$ASSIGNMENT" \
        --phenomena "$phenomenon" \
        --match_mode contains \
        --output "$SELECTION_DIR/${phenomenon}.json" \
        --show_stats
done

"$PYTHON_BIN" - "$SELECTION_DIR" "$SELECTION_DIR/selected_zip_paths.txt" <<'PY'
import json
import sys
from pathlib import Path

selection_dir = Path(sys.argv[1])
output_path = Path(sys.argv[2])
paths = set()
for path in sorted(selection_dir.glob("*.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    paths.update(case["hf_zip_path"] for case in payload["cases"])
output_path.write_text("".join(f"{item}\n" for item in sorted(paths)), encoding="utf-8")
print(f"Selected unique zip files: {len(paths)}")
PY

export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN="$HF_TOKEN_VALUE"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

"$PYTHON_BIN" - "$SELECTION_DIR/selected_zip_paths.txt" "$TARGET" <<'PY'
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import hf_hub_download

list_path = Path(sys.argv[1])
target = Path(sys.argv[2])
files = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
print(f"Downloading {len(files)} selected zip files with 8 workers")

def fetch(filename: str) -> str:
    return hf_hub_download(
        repo_id="vLAR/PhysInOne",
        filename=filename,
        repo_type="dataset",
        local_dir=str(target),
        token=os.environ["HF_TOKEN"],
    )

completed = 0
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(fetch, filename): filename for filename in files}
    for future in as_completed(futures):
        filename = futures[future]
        future.result()
        completed += 1
        if completed % 25 == 0 or completed == len(files):
            print(f"Downloaded {completed}/{len(files)}: {filename}", flush=True)
PY

echo "[$(date -u +%FT%TZ)] filtered PhysInOne download complete"
echo "Target: $TARGET"
echo "Selection list: $SELECTION_DIR/selected_zip_paths.txt"
