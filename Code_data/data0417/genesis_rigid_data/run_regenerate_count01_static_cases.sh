#!/usr/bin/env bash
# 批量重生成 version_1_genesis_rigid_data_all_cases 中 count_01 的静止 case000/001/002/003。
set -euo pipefail

WAN_PY=/data/gaoya/miniconda3/envs/wan/bin/python
SCRIPT=/home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/try1_physxnet_articulation_mpm0417.py
OUT=/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases
BASE="$OUT/train/rigid/single_object_preview/count_01"
LOG_DIR="$OUT/logs_regenerate_count01_static_cases"
IDS_FILE="$LOG_DIR/object_ids.txt"
FAILED_LIST="$LOG_DIR/failed_object_ids.txt"
DONE_LIST="$LOG_DIR/done_object_ids.txt"
WORKER_LOG_DIR="$LOG_DIR/workers"

CASE_INDICES=(0 1 2 3)

mkdir -p "$LOG_DIR" "$WORKER_LOG_DIR"
: > "$FAILED_LIST"
rm -f "$DONE_LIST"

python3 - <<'PY' > "$IDS_FILE"
from pathlib import Path
base = Path("/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/train/rigid/single_object_preview/count_01")
ids = sorted({p.name.split("__case", 1)[0] for p in base.glob("*__case*") if p.is_dir()})
for oid in ids:
    print(oid)
PY

mapfile -t OBJECT_IDS < "$IDS_FILE"
TOTAL="${#OBJECT_IDS[@]}"

# GPU 4 has shown repeated illegal-memory-access failures in Genesis, so keep it
# out of the default pool for reruns unless explicitly requested.
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3,5,6,7}"
PROCS_PER_GPU="${PROCS_PER_GPU:-2}"
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
    *) return 1 ;;
  esac
}

delete_target_cases() {
  local object_id="$1"
  local case_idx case_name sample_dir
  for case_idx in "${CASE_INDICES[@]}"; do
    case_name="$(case_index_to_name "$case_idx")" || return 1
    sample_dir="$BASE/${object_id}__${case_name}"
    if [[ -d "$sample_dir" ]]; then
      rm -rf "$sample_dir"
    fi
  done
}

process_object() {
  local object_id="$1"
  local idx="$2"
  local gpu_id="$3"
  local log_path="$LOG_DIR/${object_id}.log"

  echo "==> [${idx}/${TOTAL}] object_id=${object_id} gpu=${gpu_id} regenerate_count01_static"
  : > "$log_path"

  delete_target_cases "$object_id"

  set +e
  CUDA_VISIBLE_DEVICES="$gpu_id" "$WAN_PY" "$SCRIPT" \
    --physx_root /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet \
    --version version_1 \
    --object_id "$object_id" \
    --output_root "$OUT" \
    --run_genesis \
    --generate_all_count_motion_cases \
    --case_index_filter "${CASE_INDICES[@]}" \
    --rigid_count_filter 1 \
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

  if [[ "$status" -ne 0 ]]; then
    echo "FAILED object_id=${object_id} gpu=${gpu_id} rc=${status} log=${log_path}" | tee -a "$WORKER_LOG_DIR/summary.log"
    echo "$object_id" >> "$FAILED_LIST"
    return 0
  fi

  echo "$object_id" >> "$DONE_LIST"
  echo "DONE object_id=${object_id} gpu=${gpu_id}" | tee -a "$WORKER_LOG_DIR/summary.log"
}

worker_main() {
  local slot_idx="$1"
  local gpu_id="$2"
  local worker_log="$WORKER_LOG_DIR/worker_${slot_idx}_gpu${gpu_id}.log"
  : > "$worker_log"
  echo "WORKER slot=${slot_idx} gpu=${gpu_id} start total=${TOTAL}" | tee -a "$worker_log"
  local idx
  for ((idx=slot_idx; idx<TOTAL; idx+=TOTAL_SLOTS)); do
    local object_id="${OBJECT_IDS[idx]}"
    echo "WORKER slot=${slot_idx} gpu=${gpu_id} pick index=$((idx+1)) object_id=${object_id}" | tee -a "$worker_log"
    process_object "$object_id" "$((idx+1))" "$gpu_id" 2>&1 | tee -a "$worker_log"
  done
  echo "WORKER slot=${slot_idx} gpu=${gpu_id} done" | tee -a "$worker_log"
}

echo "[$(date '+%F %T')] count_01 objects=${TOTAL}" | tee -a "$WORKER_LOG_DIR/summary.log"
echo "[$(date '+%F %T')] gpu_ids=${GPU_IDS_CSV} procs_per_gpu=${PROCS_PER_GPU} total_slots=${TOTAL_SLOTS}" | tee -a "$WORKER_LOG_DIR/summary.log"
echo "[$(date '+%F %T')] log_dir=${LOG_DIR}" | tee -a "$WORKER_LOG_DIR/summary.log"

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
