#!/usr/bin/env bash
set -euo pipefail

# Queue the grouped backfill behind the current GPU2 regression.  The
# regression is a correctness gate: the grouped extraction must agree with
# the previous per-metric implementation before it is allowed to write the
# remaining metrics.
REG_ROOT="/data/gaoya/agent-data/outputs/physv_v2v_0819_rigidbench_strict_test70/logs/regression_gpu2_grouped"
BACKFILL="/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/run_test70_rigidbench_grouped_backfill_gpu2.sh"
PYTHON="${PYTHON:-/home/gaoya/miniconda3/envs/sam/bin/python}"

echo "[queue] started $(date -Is)"
echo "[queue] waiting for grouped regression workers on physical GPU2"
while pgrep -f '[c]ompare_test70_rigidbench_grouped.py' >/dev/null 2>&1; do
  date -Is
  sleep 30
done

echo "[queue] regression workers exited; validating reports"
REG_ROOT="$REG_ROOT" "$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["REG_ROOT"])
reports = root / "reports"
required = ("mask", "depth", "identity")
bad = []
for group in required:
    path = reports / f"gpu2_{group}.json"
    if not path.is_file():
        bad.append(f"missing:{path}")
        continue
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        bad.append(f"invalid:{path}:{exc}")
        continue
    if not payload.get("ok", False):
        bad.append(f"not_ok:{path}:{payload.get('mismatches')}")
if bad:
    raise SystemExit("Grouped regression gate failed: " + "; ".join(bad))
print("Grouped regression gate passed: mask/depth/identity")
PY

echo "[queue] starting grouped backfill $(date -Is)"
exec bash "$BACKFILL"
