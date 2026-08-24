#!/usr/bin/env bash
set -euo pipefail

HF_BIN="/home/gaoya/miniconda3/envs/wan-cu128/bin/hf"
PYTHON_BIN="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
TARGET="/data/gaoya/dataset/vLAR-PhysInOne"
FILTER_SCRIPT="$TARGET/assets/scripts/filter_cases.py"
ASSIGNMENT="$TARGET/assets/metadata/repo_assignment.txt"
REPO_MAP="$TARGET/assets/metadata/repo_map.json"
SELECTION_DIR="$TARGET/selection"
CENTER_SELECTION="$SELECTION_DIR/selected_front_center_camera_1700.json"
CENTER_OUTPUT="$TARGET/front_center_camera_1700"
CENTER_DOWNLOADER="$(dirname "$0")/download_physinone_center_camera.py"

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
download download vLAR/PhysInOne \
    assets/metadata/repo_map.json \
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

"$PYTHON_BIN" - "$SELECTION_DIR" "$CENTER_SELECTION" <<'PY'
import json
import random
import sys
from pathlib import Path

selection_dir = Path(sys.argv[1])
output_path = Path(sys.argv[2])
records = {}
for path in sorted(selection_dir.glob("*.json")):
    if path.name == output_path.name:
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    for case in payload.get("cases", []):
        records[(case["part_id"], case["hf_zip_path"])] = case

all_cases = sorted(
    records.values(),
    key=lambda item: (item["part_id"], item["hf_zip_path"]),
)
rng = random.Random(42)
selected = rng.sample(all_cases, 1700) if len(all_cases) > 1700 else all_cases
selected.sort(key=lambda item: (item["part_id"], item["hf_zip_path"]))
payload = {
    "filters": {
        "source": "union of the eight requested phenomenon selections",
        "deduplicated": True,
        "num": 1700,
        "seed": 42,
        "camera": "front-center camera: smallest optical-axis vertical component, center-offset tie-break",
    },
    "stats": {
        "available_unique_cases": len(all_cases),
        "selected_cases": len(selected),
    },
    "cases": selected,
}
output_path.write_text(
    json.dumps(payload, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
print(f"Available unique cases: {len(all_cases)}")
print(f"Selected center-camera cases: {len(selected)}")
print(f"Saved selection to: {output_path}")
PY

export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN="$HF_TOKEN_VALUE"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

"$PYTHON_BIN" "$CENTER_DOWNLOADER" \
    --selection "$CENTER_SELECTION" \
    --repo-map "$REPO_MAP" \
    --output-dir "$CENTER_OUTPUT" \
    --endpoint "$HF_ENDPOINT" \
    --workers 4

echo "[$(date -u +%FT%TZ)] filtered PhysInOne download complete"
echo "Target: $CENTER_OUTPUT"
echo "Selection: $CENTER_SELECTION"
