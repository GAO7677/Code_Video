#!/usr/bin/env bash
set -euo pipefail
# /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/serve_test5_comparison_viewer.sh /data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/test_5 8095

ROOT="${1:-/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/test_5}"
PORT="${2:-8095}"
BIND="${BIND:-0.0.0.0}"

PYTHON="${PYTHON:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
BUILDER="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/build_test5_comparison_viewer.py"

if [[ ! -d "${ROOT}" ]]; then
  echo "ERROR: input root does not exist: ${ROOT}" >&2
  exit 2
fi
if [[ ! -f "${BUILDER}" ]]; then
  echo "ERROR: viewer builder does not exist: ${BUILDER}" >&2
  exit 2
fi

echo "[viewer] scanning ${ROOT}"
"${PYTHON}" "${BUILDER}" --root "${ROOT}"

URL="http://localhost:${PORT}/index.html"
echo "[viewer] page: ${ROOT}/index.html"
echo "[viewer] url: ${URL}"

if ss -ltn "sport = :${PORT}" | tail -n +2 | grep -q .; then
  echo "[viewer] port ${PORT} is already in use; page was rebuilt but no new server was started."
  echo "[viewer] if you want to restart manually, stop the existing process, then run:"
  echo "  ${PYTHON} -m http.server ${PORT} --bind ${BIND} --directory ${ROOT}"
  exit 0
fi

echo "[viewer] starting foreground server on ${BIND}:${PORT}"
exec "${PYTHON}" -m http.server "${PORT}" --bind "${BIND}" --directory "${ROOT}"
