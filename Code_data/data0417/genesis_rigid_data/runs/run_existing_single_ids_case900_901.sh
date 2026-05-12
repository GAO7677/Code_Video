#!/usr/bin/env bash
# 用途：为现有样本 id 回补 case900 与 case901。
# 该脚本用于为已有单物体 object_id 补生成或修复 case900/case901 样本；输入为 /data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases 中已有 id、PhysXNet 资产和过滤脚本，输出为同目录下更新后的样本及 logs_case900_901_existing_ids 日志。
set -u

WAN_PY=/data/gaoya/miniconda3/envs/wan/bin/python
SCRIPT=/home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/generators/try1_physxnet_articulation_mpm0417.py
FILTER_SCRIPT=/home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/repair/filter_single_object_motion_cases.py
OUT=/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases
BASE="$OUT/train/rigid/single_object_preview/count_01"
LOG_DIR="$OUT/logs_case900_901_existing_ids"
mkdir -p "$LOG_DIR"

IDS_FILE="$LOG_DIR/existing_ids.txt"
python3 - <<'PY' > "$IDS_FILE"
from pathlib import Path
base=Path('/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/train/rigid/single_object_preview')
ids=set()
for d in base.rglob('*__case*'):
    if d.is_dir():
        ids.add(d.name.split('__case',1)[0])
for oid in sorted(ids):
    print(oid)
PY

needs_generate() {
    local sample_dir="$1"
    if [ ! -d "$sample_dir" ]; then
        echo 0
        return
    fi
    if { [ ! -f "$sample_dir/meta.json" ] && [ ! -f "$sample_dir/metadata.json" ]; } || [ ! -f "$sample_dir/physics/rigid_kinematics.npz" ] || [ ! -f "$sample_dir/physics/anchor_targets.npz" ]; then
        echo 0
        return
    fi
    local qa_file="$sample_dir/qa_metrics.json"
    if [ -f "$qa_file" ]; then
        python3 - <<'PY' "$qa_file"
import json, sys
path=sys.argv[1]
obj=json.load(open(path,'r',encoding='utf-8'))
print('1' if bool(obj.get('valid', False)) else '0')
PY
        return
    fi
    "$WAN_PY" "$FILTER_SCRIPT" --root "$sample_dir" --report "$sample_dir/qa_scan_tmp.json" --write_metrics >/dev/null 2>&1
    if [ -f "$qa_file" ]; then
        python3 - <<'PY' "$qa_file"
import json, sys
path=sys.argv[1]
obj=json.load(open(path,'r',encoding='utf-8'))
print('1' if bool(obj.get('valid', False)) else '0')
PY
        rm -f "$sample_dir/qa_scan_tmp.json"
        return
    fi
    echo 0
}

total=$(wc -l < "$IDS_FILE" | tr -d ' ')
echo "[$(date '+%F %T')] Found $total existing ids under $BASE"
echo "[$(date '+%F %T')] Logs: $LOG_DIR"

idx=0
while read -r oid; do
    [ -z "$oid" ] && continue
    idx=$((idx+1))
    sample900="$BASE/${oid}__case900_random_parabola"
    sample901="$BASE/${oid}__case901_high_drop"
    keep900=$(needs_generate "$sample900")
    keep901=$(needs_generate "$sample901")
    if [ "$keep900" = "1" ] && [ "$keep901" = "1" ]; then
        echo "[$(date '+%F %T')] [$idx/$total] object_id=$oid skip_valid"
        continue
    fi
    echo "[$(date '+%F %T')] [$idx/$total] object_id=$oid start regen900=$([ "$keep900" = "1" ] && echo 0 || echo 1) regen901=$([ "$keep901" = "1" ] && echo 0 || echo 1)"
    if [ "$keep900" != "1" ]; then rm -rf "$sample900"; fi
    if [ "$keep901" != "1" ]; then rm -rf "$sample901"; fi
    "$WAN_PY" "$SCRIPT" \
      --physx_root /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet \
      --version version_1 \
      --object_id "$oid" \
      --output_root "$OUT" \
      --run_genesis \
      --num_random_cases 12 \
      --case_scene_mode diverse \
      --case_index_filter 900 901 \
      --prefer_existing_runtime_meshes \
      --dt 0.003 \
      --substeps 40 \
      --steps 49 \
      --fps 12 \
      --simulator_mode rigid \
      --rigid_target_object_count 1 \
      --physxnet_volume_threshold_m3 999999.0 \
      --motion_case_max_retries 8 \
      --disable_striker \
      > "$LOG_DIR/${oid}.log" 2>&1
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "[$(date '+%F %T')] [$idx/$total] object_id=$oid FAILED rc=$rc log=$LOG_DIR/${oid}.log"
    else
        echo "[$(date '+%F %T')] [$idx/$total] object_id=$oid done"
    fi
done < "$IDS_FILE"

echo "[$(date '+%F %T')] all done"
