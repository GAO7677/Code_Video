#!/usr/bin/env bash
set -euo pipefail

source /home/gaoya/miniconda3/etc/profile.d/conda.sh
conda activate wan

TRAIN_ROOT=/home/gaoya/Code_Video/Code_data/Code_train/train_0419
STATE_ADAPTER_ROOT=${TRAIN_ROOT}/state_adapter

RAW_DATASET_ROOT=/data/gaoya/dataset/kubric_tfds_movi-d
PIPELINE_ROOT=/home/gaoya/movi_d_pipeline
PHYSICS_ROOT=${PIPELINE_ROOT}/movi_d_physics
WINDOWS_TRAIN_ROOT=${PIPELINE_ROOT}/oracle_windows_train
WINDOWS_TEST_ROOT=${PIPELINE_ROOT}/oracle_windows_test
PORTAL_INPUT_ROOT=${PIPELINE_ROOT}/portal_input
PORTAL_OUTPUT_ROOT=${PIPELINE_ROOT}/portal

COUNT_BUCKETS=$(
  python3 - <<'PY'
print(",".join(f"count_{idx:02d}" for idx in range(4, 21)))
PY
)

mkdir -p "${PIPELINE_ROOT}"

echo "[1/4] convert MOVI-D TFRecords -> physics samples"
python ${STATE_ADAPTER_ROOT}/prepare_movi_d_physics.py \
  --dataset_root ${RAW_DATASET_ROOT} \
  --out_root ${PHYSICS_ROOT} \
  --splits train,test \
  --skip_existing

echo "[2/4] build oracle windows for train split"
python ${STATE_ADAPTER_ROOT}/build_stage1_oracle_windows.py \
  --dataset_root ${PHYSICS_ROOT} \
  --source_split train \
  --out_root ${WINDOWS_TRAIN_ROOT} \
  --count_buckets ${COUNT_BUCKETS} \
  --context_len 8 \
  --future_lengths 5,9,13 \
  --contact_mode none \
  --future_main_visibility_threshold 0.5

echo "[3/4] build oracle windows for test split"
python ${STATE_ADAPTER_ROOT}/build_stage1_oracle_windows.py \
  --dataset_root ${PHYSICS_ROOT} \
  --source_split test \
  --out_root ${WINDOWS_TEST_ROOT} \
  --count_buckets ${COUNT_BUCKETS} \
  --context_len 8 \
  --future_lengths 5,9,13 \
  --contact_mode none \
  --future_main_visibility_threshold 0.5

echo "[4/4] build local portal from test windows"
mkdir -p "${PORTAL_INPUT_ROOT}/stage1a_precontact_strict"
mkdir -p "${PORTAL_INPUT_ROOT}/stage1b_simple_dynamics"
printf '{"accepted": [], "skipped": []}\n' > "${PORTAL_INPUT_ROOT}/stage1a_precontact_strict/manifest.json"
cp "${WINDOWS_TEST_ROOT}/manifest.json" "${PORTAL_INPUT_ROOT}/stage1b_simple_dynamics/manifest.json"

python ${STATE_ADAPTER_ROOT}/visualizations/visualize_stage1_subsets.py \
  --subset_root ${PORTAL_INPUT_ROOT} \
  --output_dir ${PORTAL_OUTPUT_ROOT} \
  --num_windows_per_subset 8

echo "DONE pipeline_root=${PIPELINE_ROOT}"
echo "PORTAL_URL=http://127.0.0.1:8150/movi_d_pipeline/portal/index.html"
