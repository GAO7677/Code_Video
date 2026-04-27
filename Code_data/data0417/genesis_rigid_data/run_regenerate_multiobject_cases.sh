#!/usr/bin/env bash
# 批量重生成 version_1_genesis_rigid_data_all_cases 中现有的多物体 case。
# 默认等待单物体 count_01 重生成任务结束后再启动，避免争用 GPU。
set -euo pipefail

WAN_PY=/data/gaoya/miniconda3/envs/wan/bin/python
SCRIPT=/home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/try1_physxnet_articulation_mpm0417.py
OUT=/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases
MULTI_ROOT="$OUT/train/rigid/interaction_pair_plus_dynamic"
COUNT01_SCRIPT=/home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/run_regenerate_count01_static_cases.sh

LOG_DIR="$OUT/logs_regenerate_multiobject_cases"
PLAN_FILE="$LOG_DIR/regen_plan.tsv"
FAILED_LIST="$LOG_DIR/failed_jobs.txt"
DONE_LIST="$LOG_DIR/done_jobs.txt"
WORKER_LOG_DIR="$LOG_DIR/workers"

mkdir -p "$LOG_DIR" "$WORKER_LOG_DIR"
: > "$FAILED_LIST"
rm -f "$DONE_LIST"

python3 - <<'PY' > "$PLAN_FILE"
import json
import re
from collections import defaultdict
from pathlib import Path

multi_root = Path("/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/train/rigid/interaction_pair_plus_dynamic")
case_re = re.compile(r"__case(\d+)")
groups = defaultdict(set)

for bucket in ["count_02", "count_03_04"]:
    bucket_dir = multi_root / bucket
    if not bucket_dir.exists():
        continue
    for meta_path in bucket_dir.glob("*/metadata.json"):
        sample_name = meta_path.parent.name
        match = case_re.search(sample_name)
        if not match:
            continue
        object_id = sample_name.split("__case", 1)[0]
        case_idx = int(match.group(1))
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        num_objects = int(meta.get("num_objects", -1))
        if bucket == "count_02":
            target_count = 2
        else:
            target_count = num_objects
        key = (object_id, bucket, target_count)
        groups[key].add(case_idx)

for object_id, bucket, target_count in sorted(groups.keys()):
    cases = ",".join(str(idx) for idx in sorted(groups[(object_id, bucket, target_count)]))
    print(f"{object_id}\t{bucket}\t{target_count}\t{cases}")
PY

mapfile -t PLAN_ROWS < "$PLAN_FILE"
TOTAL="${#PLAN_ROWS[@]}"

GPU_IDS_CSV="${GPU_IDS:-0,1,2,3,5,6,7}"
PROCS_PER_GPU="${PROCS_PER_GPU:-2}"
WAIT_FOR_COUNT01="${WAIT_FOR_COUNT01:-1}"

IFS=',' read -r -a GPU_IDS_ARR <<< "$GPU_IDS_CSV"
SLOT_GPU_IDS=()
for gpu_id in "${GPU_IDS_ARR[@]}"; do
  gpu_trimmed="$(echo "$gpu_id" | xargs)"
  [[ -z "$gpu_trimmed" ]] && continue
  for ((rep=0; rep<PROCS_PER_GPU; rep++)); do
    SLOT_GPU_IDS+=("$gpu_trimmed")
  done
done
if [[ "${#SLOT_GPU_IDS[@]}" -eq 0 ]]; then
  SLOT_GPU_IDS=("0")
fi
TOTAL_SLOTS="${#SLOT_GPU_IDS[@]}"

case_index_to_name() {
  local case_idx="$1"
  case "$case_idx" in
    0) echo "case000_static_center" ;;
    1) echo "case001_static_left" ;;
    2) echo "case002_static_right" ;;
    3) echo "case003_static_highdrop" ;;
    5) echo "case005_entry_left" ;;
    6) echo "case006_entry_right" ;;
    7) echo "case007_entry_fast_center" ;;
    100) echo "case000_static_center_v2" ;;
    101) echo "case001_static_left_v2" ;;
    102) echo "case002_static_right_v2" ;;
    900) echo "case900_random_parabola" ;;
    901) echo "case901_high_drop" ;;
    *) return 1 ;;
  esac
}

wait_for_count01() {
  if [[ "$WAIT_FOR_COUNT01" != "1" ]]; then
    return 0
  fi
  echo "[$(date '+%F %T')] waiting for count_01 job: $COUNT01_SCRIPT" | tee -a "$WORKER_LOG_DIR/summary.log"
  while pgrep -af "$COUNT01_SCRIPT" >/dev/null 2>&1; do
    sleep 30
  done
  echo "[$(date '+%F %T')] count_01 job finished, starting multi-object regeneration" | tee -a "$WORKER_LOG_DIR/summary.log"
}

delete_target_cases() {
  local object_id="$1"
  local bucket="$2"
  local case_csv="$3"
  local sample_dir case_idx case_name
  IFS=',' read -r -a case_indices <<< "$case_csv"
  for case_idx in "${case_indices[@]}"; do
    [[ -z "$case_idx" ]] && continue
    case_name="$(case_index_to_name "$case_idx")" || return 1
    sample_dir="$MULTI_ROOT/${bucket}/${object_id}__${case_name}"
    if [[ -d "$sample_dir" ]]; then
      rm -rf "$sample_dir"
    fi
  done
}

process_plan_row() {
  local row="$1"
  local idx="$2"
  local gpu_id="$3"
  IFS=$'\t' read -r object_id bucket target_count case_csv <<< "$row"
  local safe_case_csv="${case_csv//,/--}"
  local log_path="$LOG_DIR/${object_id}__${bucket}__count${target_count}__${safe_case_csv}.log"

  echo "==> [${idx}/${TOTAL}] object_id=${object_id} bucket=${bucket} target_count=${target_count} gpu=${gpu_id} cases=${case_csv}"
  : > "$log_path"

  delete_target_cases "$object_id" "$bucket" "$case_csv"

  IFS=',' read -r -a case_indices <<< "$case_csv"

  set +e
  CUDA_VISIBLE_DEVICES="$gpu_id" "$WAN_PY" "$SCRIPT" \
    --physx_root /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet \
    --version version_1 \
    --object_id "$object_id" \
    --output_root "$OUT" \
    --run_genesis \
    --generate_all_count_motion_cases \
    --case_index_filter "${case_indices[@]}" \
    --rigid_count_filter "$target_count" \
    --prefer_existing_runtime_meshes \
    --dt 0.003 \
    --substeps 40 \
    --ball_posx 0.03 \
    --steps 12 \
    --fps 12 \
    --simulator_mode rigid \
    > "$log_path" 2>&1
  status=$?
  set -e

  local job_key="${object_id}\t${bucket}\t${target_count}\t${case_csv}"
  if [[ "$status" -ne 0 ]]; then
    echo -e "$job_key" >> "$FAILED_LIST"
    echo "FAILED object_id=${object_id} bucket=${bucket} target_count=${target_count} gpu=${gpu_id} rc=${status} cases=${case_csv} log=${log_path}" | tee -a "$WORKER_LOG_DIR/summary.log"
    return 0
  fi

  echo -e "$job_key" >> "$DONE_LIST"
  echo "DONE object_id=${object_id} bucket=${bucket} target_count=${target_count} gpu=${gpu_id} cases=${case_csv}" | tee -a "$WORKER_LOG_DIR/summary.log"
}

worker_main() {
  local slot_idx="$1"
  local gpu_id="$2"
  local worker_log="$WORKER_LOG_DIR/worker_${slot_idx}_gpu${gpu_id}.log"
  : > "$worker_log"
  echo "WORKER slot=${slot_idx} gpu=${gpu_id} start total=${TOTAL}" | tee -a "$worker_log"
  local idx
  for ((idx=slot_idx; idx<TOTAL; idx+=TOTAL_SLOTS)); do
    local row="${PLAN_ROWS[idx]}"
    echo "WORKER slot=${slot_idx} gpu=${gpu_id} pick plan_index=$((idx+1))" | tee -a "$worker_log"
    process_plan_row "$row" "$((idx+1))" "$gpu_id" 2>&1 | tee -a "$worker_log"
  done
  echo "WORKER slot=${slot_idx} gpu=${gpu_id} done" | tee -a "$worker_log"
}

echo "[$(date '+%F %T')] plan_rows=${TOTAL}" | tee -a "$WORKER_LOG_DIR/summary.log"
echo "[$(date '+%F %T')] gpu_ids=${GPU_IDS_CSV} procs_per_gpu=${PROCS_PER_GPU} total_slots=${TOTAL_SLOTS}" | tee -a "$WORKER_LOG_DIR/summary.log"
echo "[$(date '+%F %T')] log_dir=${LOG_DIR}" | tee -a "$WORKER_LOG_DIR/summary.log"

wait_for_count01

PIDS=()
for ((slot_idx=0; slot_idx<TOTAL_SLOTS; slot_idx++)); do
  worker_main "$slot_idx" "${SLOT_GPU_IDS[slot_idx]}" &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

echo "[$(date '+%F %T')] finished status=${status}" | tee -a "$WORKER_LOG_DIR/summary.log"
exit "$status"
