#!/usr/bin/env bash
# 用途：运行当前 rigid 训练生成 smoke test 并检查输出。
# 该脚本用于跑当前可用的 Genesis rigid smoke test：生成少量 train 样本，并自动可视化第一个样本。
set -euo pipefail

ROOT=/home/gaoya/Code_Video/Code_data/data0417
PY=/data/gaoya/miniconda3/envs/wan/bin/python
OUT=/data/gaoya/AAA_test_video/Dataset_physV/0417data/physxnet_train_rigid_multi_smoke

cd "$ROOT"

"$PY" genesis_rigid_data/generators/generate_physxnet_train_rigid_multi.py \
  --output_root "$OUT" \
  --smoke \
  --samples_per_combo 1

SAMPLE_DIR="$("$PY" - <<'PY'
from pathlib import Path

root = Path("/data/gaoya/AAA_test_video/Dataset_physV/0417data/physxnet_train_rigid_multi_smoke")
metadata_paths = sorted(root.glob("train/rigid/*/*/*/metadata.json"))
if not metadata_paths:
    raise SystemExit("No generated sample found under smoke output root.")
print(metadata_paths[0].parent)
PY
)"

"$PY" genesis_rigid_data/inspect/visualize_sample.py \
  --sample_dir "$SAMPLE_DIR"

echo "Smoke sample: $SAMPLE_DIR"
echo "Visualization dir: $SAMPLE_DIR/visualizations"
