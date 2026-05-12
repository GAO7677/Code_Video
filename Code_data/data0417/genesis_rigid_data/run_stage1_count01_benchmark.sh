#!/usr/bin/env bash
# 用途：生成 stage1 count_01 held-out benchmark。
# 该脚本用于生成 stage1 count_01 heldout benchmark 并构建子集；输入为 PhysXNet 数据和 /data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases 训练根目录，输出为 /data/gaoya/AAA_test_video/Dataset_physV/0417data_benchmark/stage1_count01_benchmark 下的 heldout 样本、ID 列表和 subset summary。
set -u

WAN_PY=/data/gaoya/miniconda3/envs/wan/bin/python
GEN_SCRIPT=/home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/generate_rigid_benchmark.py
TRAIN_ROOT=/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases
OUT=/data/gaoya/AAA_test_video/Dataset_physV/0417data_benchmark/stage1_count01_benchmark
LOG_DIR="$OUT/logs"
IDS_FILE="$OUT/heldout_ids.txt"
USED_FILE="$OUT/excluded_stage1_train_ids.txt"
mkdir -p "$LOG_DIR"

"$WAN_PY" "$GEN_SCRIPT" stage1_heldout \
  --physx_root /data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet \
  --version version_1 \
  --output_root "$OUT" \
  --stage1_train_root "$TRAIN_ROOT" \
  --heldout_seed 20260421 \
  --heldout_count 8 \
  --num_random_cases 12 \
  --case_index_filter 900 901 \
  --dt 0.003 \
  --substeps 40 \
  --steps 49 \
  --fps 12 \
  --rigid_target_object_count 1 \
  --physxnet_volume_threshold_m3 999999.0 \
  --motion_case_max_retries 8

echo "[$(date '+%F %T')] out=$OUT"
echo "[$(date '+%F %T')] excluded ids: $(tr '\n' ' ' < "$USED_FILE" 2>/dev/null)"
echo "[$(date '+%F %T')] heldout ids: $(tr '\n' ' ' < "$IDS_FILE" 2>/dev/null)"
echo "[$(date '+%F %T')] build subsets done"
cat "$OUT/preprocess_v1/stage1_subsets_v1/summary.json" 2>/dev/null || true
echo
