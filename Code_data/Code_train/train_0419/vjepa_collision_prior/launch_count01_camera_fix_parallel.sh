#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPAIR_PY="${SCRIPT_DIR}/repair_count01_final_camera.py"
SCAN_PY="/data/gaoya/miniconda3/envs/vjepa2/bin/python"
WAN_PY="/data/gaoya/miniconda3/envs/wan/bin/python"
REPORT_ROOT="${SCRIPT_DIR}/count01_camera_fix_parallel"
DEVICES="${DEVICES:-0123}"
PROCS_PER_DEVICE="${PROCS_PER_DEVICE:-15}"

mkdir -p "${REPORT_ROOT}"

echo "[scan] refreshing failing list"
"${SCAN_PY}" "${REPAIR_PY}" --report-dir "${REPORT_ROOT}/summary_before" >/tmp/count01_camera_fix_scan.log
cat /tmp/count01_camera_fix_scan.log

FAIL_LIST="${REPORT_ROOT}/failing_samples.txt"
"${SCAN_PY}" - <<'PY' > "${FAIL_LIST}"
import json
from pathlib import Path
report = Path("/home/gaoya/Code_Video/Code_data/Code_train/train_0419/vjepa_collision_prior/count01_camera_fix_parallel/summary_before/scan_before.json")
data = json.loads(report.read_text(encoding="utf-8"))
for item in data["samples"]:
    if not item["last_safe_margin"]:
        print(item["sample_name"])
PY

TOTAL_FAILING="$(wc -l < "${FAIL_LIST}" | tr -d ' ')"
echo "[launch] total failing samples=${TOTAL_FAILING}"

for dev in $(echo "${DEVICES}" | grep -o .); do
  : > "${REPORT_ROOT}/device_${dev}.txt"
done

"${SCAN_PY}" - <<'PY'
from pathlib import Path
devices = list("0123")
root = Path("/home/gaoya/Code_Video/Code_data/Code_train/train_0419/vjepa_collision_prior/count01_camera_fix_parallel")
samples = [line.strip() for line in (root / "failing_samples.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
for idx, sample in enumerate(samples):
    dev = devices[idx % len(devices)]
    with (root / f"device_{dev}.txt").open("a", encoding="utf-8") as f:
        f.write(sample + "\n")
PY

for dev in $(echo "${DEVICES}" | grep -o .); do
  count="$(wc -l < "${REPORT_ROOT}/device_${dev}.txt" | tr -d ' ')"
  echo "[launch] device=${dev} assigned=${count}"
done

run_device_workers() {
  local dev="$1"
  local file="${REPORT_ROOT}/device_${dev}.txt"
  local device_root="${REPORT_ROOT}/workers/device_${dev}"
  mkdir -p "${device_root}"
  if [[ ! -s "${file}" ]]; then
    echo "[launch] device=${dev} no samples"
    return 0
  fi
  xargs -a "${file}" -I{} -P "${PROCS_PER_DEVICE}" bash -lc '
    set -euo pipefail
    dev="$1"
    sample="$2"
    root="$3"
    CUDA_VISIBLE_DEVICES="${dev}" '"${SCAN_PY}"' "'"${REPAIR_PY}"'" \
      --apply \
      --sample-name "${sample}" \
      --report-dir "${root}/${sample}" \
      > "${root}/${sample}.stdout.log" 2>&1
  ' _ "${dev}" "{}" "${device_root}"
}

for dev in $(echo "${DEVICES}" | grep -o .); do
  run_device_workers "${dev}" &
done

wait

echo "[scan] refreshing final summary"
"${SCAN_PY}" "${REPAIR_PY}" --report-dir "${REPORT_ROOT}/summary_after" >/tmp/count01_camera_fix_final.log
cat /tmp/count01_camera_fix_final.log
