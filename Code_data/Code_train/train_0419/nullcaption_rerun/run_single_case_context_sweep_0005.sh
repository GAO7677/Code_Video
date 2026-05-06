#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=/data/gaoya/miniconda3/envs/wan/bin/python
SCRIPT_ROOT=/home/gaoya/Code_Video/Code_data/Code_train/train_0419
VACE_ROOT=/data/gaoya/ckpt/Wan-AI-Wan2.1-VACE-1.3B
CASE_META=/data/gaoya/dataset/physics-iq-benchmark/mytest/0005_perspective-center_trimmed-ball-behind-rotating-paper/meta.json
BENCH_ROOT=/data/gaoya/AAA_test_video/Benchmark/stage0_V2V_nullcaption
WORK_ROOT=${BENCH_ROOT}/tools/context_sweep_case0005
META_ROOT=${WORK_ROOT}/meta
GEN_ROOT=${WORK_ROOT}/generated
RUNTIME_ROOT=${WORK_ROOT}/runtime

mkdir -p "${META_ROOT}" "${GEN_ROOT}" "${RUNTIME_ROOT}"

for CTX in 16 32 38; do
  META_PATH=${META_ROOT}/context_${CTX}f.json
  META_LIST_PATH=${META_ROOT}/context_${CTX}f.txt
  OUTPUT_DIR=${GEN_ROOT}/context_${CTX}f
  RUNTIME_DIR=${RUNTIME_ROOT}/context_${CTX}f
  mkdir -p "${OUTPUT_DIR}" "${RUNTIME_DIR}"

  "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
src = Path("${CASE_META}")
dst = Path("${META_PATH}")
data = json.loads(src.read_text())
data["caption"] = data.get("caption", "")
dst.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
print(dst)
PY

  printf '%s\n' "${META_PATH}" > "${META_LIST_PATH}"

  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2} "${PYTHON_BIN}" "${SCRIPT_ROOT}/batch_eval_vace.py" \
    --vace_root "${VACE_ROOT}" \
    --meta_list_path "${META_LIST_PATH}" \
    --output_root "${OUTPUT_DIR}" \
    --runtime_root "${RUNTIME_DIR}" \
    --model_name "vace_ctx$(printf '%02d' "${CTX}")f_case0005" \
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

"${PYTHON_BIN}" "${SCRIPT_ROOT}/nullcaption_rerun/build_single_case_context_sweep_portal.py"
