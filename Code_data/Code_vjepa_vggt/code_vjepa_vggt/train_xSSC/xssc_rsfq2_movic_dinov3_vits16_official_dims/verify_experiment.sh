#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPECTED_DINOV3_COMMIT=6876159a11b4df116f30f667f8c9888617df0751

actual_commit="$(git -C "${ROOT}/third_party/dinov3" rev-parse HEAD)"
if [ "${actual_commit}" != "${EXPECTED_DINOV3_COMMIT}" ]; then
  echo "ERROR: DINOv3 commit mismatch: ${actual_commit}" >&2
  exit 1
fi
if [ -n "$(git -C "${ROOT}/third_party/dinov3" status --short)" ]; then
  echo "ERROR: vendored DINOv3 checkout is dirty" >&2
  exit 1
fi

cd "${ROOT}"
sha256sum --check EXPERIMENT_SHA256SUMS
echo "[ok] DINOv3 commit and experiment source hashes verified"
