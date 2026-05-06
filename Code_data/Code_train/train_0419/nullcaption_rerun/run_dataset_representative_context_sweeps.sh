#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=/data/gaoya/miniconda3/envs/wan/bin/python
SCRIPT_ROOT=/home/gaoya/Code_Video/Code_data/Code_train/train_0419
VACE_ROOT=/data/gaoya/ckpt/Wan-AI-Wan2.1-VACE-1.3B
BENCH_ROOT=/data/gaoya/AAA_test_video/Benchmark/stage0_V2V_nullcaption
WORK_ROOT=${BENCH_ROOT}/tools/dataset_representative_context_sweeps
META_ROOT=${WORK_ROOT}/meta
GEN_ROOT=${WORK_ROOT}/generated
RUNTIME_ROOT=${WORK_ROOT}/runtime

mkdir -p "${META_ROOT}" "${GEN_ROOT}" "${RUNTIME_ROOT}"

declare -a CASES=(
  "physics-iq-benchmark|0005_perspective-center_trimmed-ball-behind-rotating-paper|/data/gaoya/dataset/physics-iq-benchmark/mytest/0005_perspective-center_trimmed-ball-behind-rotating-paper/meta.json"
  "kubric_tfds_movi-d|movi_d_test_0005__video_668|/data/gaoya/dataset/kubric_tfds_movi-d/mytest/movi_d_test_0005__video_668/meta.json"
  "mvp-lab-OpenVidHD-0.4M-720p-48fps|rank0_1761115610.0727706_720x1280__00065__zy_NvBKW6O4_35_0to121|/data/gaoya/dataset/mvp-lab-OpenVidHD-0.4M-720p-48fps/mytest/rank0_1761115610.0727706_720x1280__00065__zy_NvBKW6O4_35_0to121/meta.json"
  "vLAR-PhysInOne|A__ObliqueProjectile_RollUpSlope_LinCarryInertia__bg158__LoFxHr_trajectory__CineCamera_0|/data/gaoya/dataset/vLAR-PhysInOne/mytest/A__ObliqueProjectile_RollUpSlope_LinCarryInertia__bg158__LoFxHr_trajectory__CineCamera_0/meta.json"
  "version_1_genesis_rigid_data_all_cases|genesis_heldout_0001__10005__case000_static_center_v2|/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/mytest/genesis_heldout_0001__10005__case000_static_center_v2/meta.json"
)

for CTX in 16 32 38; do
  LIST_PATH=${META_ROOT}/context_${CTX}f.txt
  : > "${LIST_PATH}"

  for ITEM in "${CASES[@]}"; do
    IFS='|' read -r DATASET SAMPLE META_PATH <<< "${ITEM}"
    CASE_DIR=${GEN_ROOT}/${DATASET}__${SAMPLE}/context_$(printf '%02d' "${CTX}")f
    mkdir -p "${CASE_DIR}"
    TMP_META=${META_ROOT}/${DATASET}__${SAMPLE}__context_$(printf '%02d' "${CTX}")f.json

    "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
src = Path("${META_PATH}")
dst = Path("${TMP_META}")
data = json.loads(src.read_text())
data["caption"] = data.get("caption", "")
dst.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
print(dst)
PY

    printf '%s\n' "${TMP_META}" >> "${LIST_PATH}"
  done

  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4} "${PYTHON_BIN}" "${SCRIPT_ROOT}/batch_eval_vace.py" \
    --vace_root "${VACE_ROOT}" \
    --meta_list_path "${LIST_PATH}" \
    --output_root "${GEN_ROOT}/batch_context_$(printf '%02d' "${CTX}")f" \
    --runtime_root "${RUNTIME_ROOT}/context_$(printf '%02d' "${CTX}")f" \
    --model_name "dataset_representative_ctx$(printf '%02d' "${CTX}")f" \
    --mode v2v_clipref \
    --device cuda:0 \
    --height 544 \
    --width 720 \
    --fps 16 \
    --num_frames 49 \
    --context_frames "${CTX}" \
    --num_inference_steps 50 \
    --cfg_scale 5.0 \
    --seed 42 \
    --overwrite
done

"${PYTHON_BIN}" "${SCRIPT_ROOT}/nullcaption_rerun/build_dataset_representative_context_sweeps.py"
