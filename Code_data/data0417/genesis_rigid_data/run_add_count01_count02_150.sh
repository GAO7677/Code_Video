#!/usr/bin/env bash
# 该脚本用于补生成 count_01 和 count_02 的 benchmark 样本；输入为 generate_rigid_benchmark.py、PhysXNet 默认配置和随机种子，输出为 /data/gaoya/AAA_test_video/Dataset_physV/0417data_benchmark 下的新样本及 run_add_count01_count02_150.log。
set -euo pipefail

PY=/data/gaoya/miniconda3/envs/wan/bin/python
SCRIPT=/home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/generate_rigid_benchmark.py
OUT=/data/gaoya/AAA_test_video/Dataset_physV/0417data_benchmark
LOG=/data/gaoya/AAA_test_video/Dataset_physV/0417data_benchmark/run_add_count01_count02_150.log

{
  echo "[start] $(date) count_01"
  $PY $SCRIPT physxnet_pool \
    --output_root $OUT \
    --num_objects 50 \
    --random_seed 20260423 \
    --cases_per_object 3 \
    --case_pool 5 6 7 \
    --rigid_target_object_count 1 \
    --overwrite

  echo "[start] $(date) count_02"
  $PY $SCRIPT physxnet_pool \
    --output_root $OUT \
    --num_objects 50 \
    --random_seed 20260424 \
    --cases_per_object 3 \
    --case_pool 0 1 2 5 6 7 \
    --rigid_target_object_count 2 \
    --overwrite

  echo "[done] $(date)"
} 2>&1 | tee -a "$LOG"
