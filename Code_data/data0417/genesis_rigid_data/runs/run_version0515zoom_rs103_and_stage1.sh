#!/usr/bin/env bash
# 用途：基于 version0515zoom_genesis_rigid 当前已验证过的 object/case 组合，增量生成 rs103 样本，并在完成后切出 stage1 window。
set -euo pipefail

WAN_PY=/data/gaoya/miniconda3/envs/wan/bin/python
GEN_SCRIPT=/home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/generators/try1_physxnet_articulation_mpm0417.py
WINDOW_BUILD_PYC=/home/gaoya/Code_Video/Code_data/Code_train/train_0419/state_adapter/__pycache__/build_stage1_subsets.cpython-310.pyc
PHYSX_ROOT=/data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet
VERSION=version_1
OUT_ROOT=/data/gaoya/AAA_test_video/Dataset_physV/0417data/version0515zoom_genesis_rigid
SUMMARY_ROOT=/home/gaoya/Code_Video/Code_data/data0417/data_summary/version0515zoom_genesis_rigid
GPU_ID="${GPU_ID:-7}"
RS_INDEX="${RS_INDEX:-103}"
LOG_DIR="$OUT_ROOT/_logs"
TASKS_TXT="$LOG_DIR/version0515zoom_rs${RS_INDEX}_tasks.txt"
RUN_LOG="$LOG_DIR/version0515zoom_rs${RS_INDEX}_run.log"
WINDOW_OUT="$OUT_ROOT/preprocess_v1/stage1_subsets_v1"

mkdir -p "$LOG_DIR"

cat > "$TASKS_TXT" <<'EOF'
10027 case000_static_center 0
10027 case001_static_left 1
10027 case002_static_right 2
10032 case000_static_center 0
10032 case001_static_left 1
10032 case002_static_right 2
10033 case000_static_center 0
10033 case001_static_left 1
10033 case002_static_right 2
10034 case001_static_left 1
10034 case005_entry_left 5
10034 case006_entry_right 6
10034 case007_entry_fast_center 7
10035 case002_static_right 2
10035 case005_entry_left 5
10035 case006_entry_right 6
10035 case007_entry_fast_center 7
10036 case000_static_center 0
10036 case001_static_left 1
10036 case002_static_right 2
10037 case000_static_center 0
10037 case001_static_left 1
10037 case002_static_right 2
10037 case005_entry_left 5
10037 case006_entry_right 6
10037 case007_entry_fast_center 7
EOF

echo "[$(date '+%F %T')] start gpu=${GPU_ID} rs=${RS_INDEX}" | tee "$RUN_LOG"
echo "[$(date '+%F %T')] out_root=$OUT_ROOT" | tee -a "$RUN_LOG"
echo "[$(date '+%F %T')] tasks=$(wc -l < "$TASKS_TXT")" | tee -a "$RUN_LOG"

while read -r object_id case_name case_index; do
  [[ -n "${object_id:-}" ]] || continue
  sample_dir="$OUT_ROOT/train/rigid/interaction_pair_plus_dynamic/count_02/${object_id}__${case_name}__rs${RS_INDEX}"
  if [[ -f "$sample_dir/meta.json" || -f "$sample_dir/metadata.json" ]]; then
    echo "[$(date '+%F %T')] skip existing ${sample_dir}" | tee -a "$RUN_LOG"
    continue
  fi
  echo "[$(date '+%F %T')] generate object=${object_id} case=${case_name} case_index=${case_index}" | tee -a "$RUN_LOG"
  CUDA_VISIBLE_DEVICES="$GPU_ID" "$WAN_PY" "$GEN_SCRIPT" \
    --physx_root "$PHYSX_ROOT" \
    --version "$VERSION" \
    --object_id "$object_id" \
    --output_root "$OUT_ROOT" \
    --run_genesis \
    --generate_all_count_motion_cases \
    --rigid_count_filter 2 \
    --case_index_filter "$case_index" \
    --simple_case_resample_index "$RS_INDEX" \
    --prefer_existing_runtime_meshes \
    --dt 0.003 \
    --substeps 40 \
    --steps 49 \
    --fps 12 \
    --duration_sec 3.0 \
    --sampling_fps_mult 4.0 \
    --video_slowmo_prob 0.0 \
    --simulator_mode rigid \
    >> "$RUN_LOG" 2>&1
done < "$TASKS_TXT"

echo "[$(date '+%F %T')] rebuild stage1 windows" | tee -a "$RUN_LOG"
rm -rf "$WINDOW_OUT"
"$WAN_PY" "$WINDOW_BUILD_PYC" \
  --dataset_root "$OUT_ROOT" \
  --out_root "$WINDOW_OUT" \
  --sample_filter "__rs${RS_INDEX}" \
  --count_buckets count_02 \
  --max_source_samples 0 \
  --max_windows_per_subset 500 \
  --future_main_visibility_threshold 0.5 \
  >> "$RUN_LOG" 2>&1

echo "[$(date '+%F %T')] done" | tee -a "$RUN_LOG"
