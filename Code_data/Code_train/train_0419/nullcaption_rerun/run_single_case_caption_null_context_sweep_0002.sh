#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=/data/gaoya/miniconda3/envs/wan/bin/python
SCRIPT_ROOT=/home/gaoya/Code_Video/Code_data/Code_train/train_0419
VACE_ROOT=/data/gaoya/ckpt/Wan-AI-Wan2.1-VACE-1.3B
CASE_META=/data/gaoya/dataset/physics-iq-benchmark/mytest/0002_perspective-center_trimmed-ball-and-block-fall/meta.json
BENCH_ROOT=/data/gaoya/AAA_test_video/Benchmark/stage0_V2V_nullcaption
WORK_ROOT=${BENCH_ROOT}/tools/single_case_runs/0002_perspective-center_trimmed-ball-and-block-fall
META_ROOT=${WORK_ROOT}/meta
GEN_ROOT=${WORK_ROOT}/generated
RUNTIME_ROOT=${WORK_ROOT}/runtime
LOG_ROOT=${WORK_ROOT}/logs

mkdir -p "${META_ROOT}" "${GEN_ROOT}" "${RUNTIME_ROOT}" "${LOG_ROOT}"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

src = Path("${CASE_META}")
meta_root = Path("${META_ROOT}")
src_data = json.loads(src.read_text(encoding="utf-8"))

caption_data = dict(src_data)
caption_data["caption"] = str(src_data.get("caption") or "")
(meta_root / "caption.json").write_text(
    json.dumps(caption_data, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

null_data = dict(src_data)
null_data["caption"] = ""
if "description" in null_data:
    null_data["description"] = ""
(meta_root / "nullcaption.json").write_text(
    json.dumps(null_data, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

printf '%s\n' "${META_ROOT}/caption.json" > "${META_ROOT}/caption.txt"
printf '%s\n' "${META_ROOT}/nullcaption.json" > "${META_ROOT}/nullcaption.txt"

run_job() {
  local gpu=$1
  local variant=$2
  local ctx=$3
  local meta_list_path=$4
  local output_dir=${GEN_ROOT}/${variant}_context_$(printf '%02d' "${ctx}")f
  local runtime_dir=${RUNTIME_ROOT}/${variant}_context_$(printf '%02d' "${ctx}")f
  local log_path=${LOG_ROOT}/${variant}_context_$(printf '%02d' "${ctx}")f.log

  mkdir -p "${output_dir}" "${runtime_dir}"

  CUDA_VISIBLE_DEVICES=${gpu} "${PYTHON_BIN}" "${SCRIPT_ROOT}/batch_eval_vace.py" \
    --vace_root "${VACE_ROOT}" \
    --meta_list_path "${meta_list_path}" \
    --output_root "${output_dir}" \
    --runtime_root "${runtime_dir}" \
    --model_name "case0002_${variant}_ctx$(printf '%02d' "${ctx}")f" \
    --mode v2v_clipref \
    --device cuda:0 \
    --height 544 \
    --width 720 \
    --fps 16 \
    --num_frames 49 \
    --context_frames "${ctx}" \
    --num_inference_steps 50 \
    --cfg_scale 5.0 \
    --seed 42 \
    --overwrite \
    2>&1 | tee "${log_path}"
}

(
  run_job 0 caption 8 "${META_ROOT}/caption.txt"
  run_job 0 nullcaption 8 "${META_ROOT}/nullcaption.txt"
) &
PID0=$!

(
  run_job 1 caption 16 "${META_ROOT}/caption.txt"
  run_job 1 nullcaption 16 "${META_ROOT}/nullcaption.txt"
) &
PID1=$!

(
  run_job 2 caption 32 "${META_ROOT}/caption.txt"
  run_job 2 nullcaption 32 "${META_ROOT}/nullcaption.txt"
) &
PID2=$!

(
  run_job 3 caption 38 "${META_ROOT}/caption.txt"
  run_job 3 nullcaption 38 "${META_ROOT}/nullcaption.txt"
) &
PID3=$!

wait "${PID0}" "${PID1}" "${PID2}" "${PID3}"

cat > "${WORK_ROOT}/README.txt" <<EOF
case_meta=${CASE_META}
work_root=${WORK_ROOT}
variants=caption,nullcaption
context_lengths=8,16,32,38
generated_root=${GEN_ROOT}
runtime_root=${RUNTIME_ROOT}
log_root=${LOG_ROOT}
EOF

echo "done"
