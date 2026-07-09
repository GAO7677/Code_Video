#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/physicIQ}"
PORT="${PORT:-8011}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROUPED_SCRIPT="${SCRIPT_DIR}/build_case_grouped_method_gallery.py"
ROOT_PORTAL_SCRIPT="${SCRIPT_DIR}/build_physiciq_compare_root_portal.py"
GLOBAL_PORTAL_SCRIPT="${SCRIPT_DIR}/build_physiciq_global_case_gallery.py"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found in PATH." >&2
  exit 1
fi

if [ ! -d "${ROOT}" ]; then
  echo "ERROR: ROOT directory does not exist: ${ROOT}" >&2
  exit 1
fi

echo "[portal] root = ${ROOT}"
echo "[portal] port = ${PORT}"

python3 - "${ROOT}" "${GROUPED_SCRIPT}" <<'PY'
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1]).resolve()
grouped_script = Path(sys.argv[2]).resolve()

leaf_dirs = []
for directory in sorted(root.rglob("*")):
    if not directory.is_dir():
        continue
    if directory.name.startswith("_"):
        continue
    if any(directory.glob("*_input_ctx*.jpg")):
        leaf_dirs.append(directory)

print(f"[portal] found {len(leaf_dirs)} result leaf dirs")
for directory in leaf_dirs:
    rel = directory.relative_to(root).as_posix()
    out_dir = directory / "_case_grouped_gallery"
    cmd = [
        "python3",
        str(grouped_script),
        "--result-root",
        str(directory),
        "--output-dir",
        str(out_dir),
        "--title",
        f"PhysicsIQ grouped compare: {rel}",
    ]
    subprocess.run(cmd, check=True)
    print(f"[portal] built grouped gallery: {out_dir / 'index.html'}")
PY

python3 "${ROOT_PORTAL_SCRIPT}" \
  --root "${ROOT}" \
  --output "${ROOT}/index.html" \
  --title "PhysicsIQ formal compare root portal"

echo "[portal] built root portal: ${ROOT}/index.html"

python3 "${GLOBAL_PORTAL_SCRIPT}" \
  --root "${ROOT}" \
  --output-dir "${ROOT}/_global_case_compare_gallery" \
  --title "PhysicsIQ global same-case compare across all method directories"

echo "[portal] built global gallery: ${ROOT}/_global_case_compare_gallery/index.html"
echo "[portal] starting local server in foreground"
echo "[portal] command: python3 -m http.server ${PORT} --bind 127.0.0.1"
echo "[portal] open: http://127.0.0.1:${PORT}/"
echo "[portal] global gallery: http://127.0.0.1:${PORT}/_global_case_compare_gallery/"

cd "${ROOT}"
exec python3 -m http.server "${PORT}" --bind 127.0.0.1
