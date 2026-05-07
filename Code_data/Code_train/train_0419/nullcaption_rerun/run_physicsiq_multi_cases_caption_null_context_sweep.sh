#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=/data/gaoya/miniconda3/envs/wan/bin/python
SCRIPT_ROOT=/home/gaoya/Code_Video/Code_data/Code_train/train_0419
VACE_ROOT=/data/gaoya/ckpt/Wan-AI-Wan2.1-VACE-1.3B
DATA_ROOT=/data/gaoya/dataset/physics-iq-benchmark/mytest
BENCH_ROOT=/data/gaoya/AAA_test_video/Benchmark/stage0_V2V_nullcaption
WORK_ROOT=${BENCH_ROOT}/tools/multi_case_runs/physicsiq_more_cases_ctx_sweep
META_ROOT=${WORK_ROOT}/meta
GEN_ROOT=${WORK_ROOT}/generated
RUNTIME_ROOT=${WORK_ROOT}/runtime
LOG_ROOT=${WORK_ROOT}/logs
JOB_ROOT=${WORK_ROOT}/jobs

CASES=(
  0008_perspective-center_trimmed-ball-hits-duck
  0011_perspective-center_trimmed-ball-hits-nothing
  0014_perspective-center_trimmed-ball-in-basket
  0017_perspective-center_trimmed-ball-in-sand
)
CONTEXTS=(8 16 32 38)
VARIANTS=(caption nullcaption)
GPUS=(0 1 2 3)

mkdir -p "${META_ROOT}" "${GEN_ROOT}" "${RUNTIME_ROOT}" "${LOG_ROOT}" "${JOB_ROOT}"

for case_name in "${CASES[@]}"; do
  case_root=${DATA_ROOT}/${case_name}
  case_meta=${case_root}/meta.json
  case_meta_root=${META_ROOT}/${case_name}
  mkdir -p "${case_meta_root}"

  "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

src = Path("${case_meta}")
dst_root = Path("${case_meta_root}")
data = json.loads(src.read_text(encoding="utf-8"))

caption_data = dict(data)
caption_data["caption"] = str(data.get("caption") or "")
(dst_root / "caption.json").write_text(
    json.dumps(caption_data, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

null_data = dict(data)
null_data["caption"] = ""
if "description" in null_data:
    null_data["description"] = ""
(dst_root / "nullcaption.json").write_text(
    json.dumps(null_data, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

  printf '%s\n' "${case_meta_root}/caption.json" > "${case_meta_root}/caption.txt"
  printf '%s\n' "${case_meta_root}/nullcaption.json" > "${case_meta_root}/nullcaption.txt"
done

JOB_FILE=${JOB_ROOT}/all_jobs.tsv
: > "${JOB_FILE}"

for case_name in "${CASES[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    meta_list=${META_ROOT}/${case_name}/${variant}.txt
    for ctx in "${CONTEXTS[@]}"; do
      output_dir=${GEN_ROOT}/${case_name}/${variant}_context_$(printf '%02d' "${ctx}")f
      runtime_dir=${RUNTIME_ROOT}/${case_name}/${variant}_context_$(printf '%02d' "${ctx}")f
      log_path=${LOG_ROOT}/${case_name}__${variant}_context_$(printf '%02d' "${ctx}")f.log
      model_name=physicsiq_${case_name}_${variant}_ctx$(printf '%02d' "${ctx}")f
      mkdir -p "${output_dir}" "${runtime_dir}"
      if ! find "${output_dir}" -maxdepth 1 -name '*.mp4' -print -quit | grep -q .; then
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "${case_name}" "${variant}" "${ctx}" "${meta_list}" "${output_dir}" "${runtime_dir}" "${log_path}" \
          >> "${JOB_FILE}"
      fi
    done
  done
done

split_jobs() {
  local gpu=$1
  local shard=$2
  local shard_count=$3
  local out=${JOB_ROOT}/jobs_gpu${gpu}.tsv
  : > "${out}"
  awk -F '\t' -v shard="${shard}" -v shard_count="${shard_count}" '((NR-1) % shard_count) == shard {print}' "${JOB_FILE}" > "${out}"
}

split_jobs "${GPUS[0]}" 0 ${#GPUS[@]}
split_jobs "${GPUS[1]}" 1 ${#GPUS[@]}
split_jobs "${GPUS[2]}" 2 ${#GPUS[@]}
split_jobs "${GPUS[3]}" 3 ${#GPUS[@]}

run_jobs_for_gpu() {
  local gpu=$1
  local job_file=${JOB_ROOT}/jobs_gpu${gpu}.tsv
  while IFS=$'\t' read -r case_name variant ctx meta_list output_dir runtime_dir log_path; do
    [ -n "${case_name}" ] || continue
    CUDA_VISIBLE_DEVICES=${gpu} "${PYTHON_BIN}" "${SCRIPT_ROOT}/batch_eval_vace.py" \
      --vace_root "${VACE_ROOT}" \
      --meta_list_path "${meta_list}" \
      --output_root "${output_dir}" \
      --runtime_root "${runtime_dir}" \
      --model_name "physicsiq_${case_name}_${variant}_ctx$(printf '%02d' "${ctx}")f" \
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
  done < "${job_file}"
}

(
  run_jobs_for_gpu "${GPUS[0]}"
) &
PID0=$!

(
  run_jobs_for_gpu "${GPUS[1]}"
) &
PID1=$!

(
  run_jobs_for_gpu "${GPUS[2]}"
) &
PID2=$!

(
  run_jobs_for_gpu "${GPUS[3]}"
) &
PID3=$!

wait "${PID0}" "${PID1}" "${PID2}" "${PID3}"

cat > "${WORK_ROOT}/README.txt" <<EOF
data_root=${DATA_ROOT}
cases=$(printf '%s,' "${CASES[@]}")
variants=$(printf '%s,' "${VARIANTS[@]}")
contexts=$(printf '%s,' "${CONTEXTS[@]}")
gpus=$(printf '%s,' "${GPUS[@]}")
generated_root=${GEN_ROOT}
runtime_root=${RUNTIME_ROOT}
log_root=${LOG_ROOT}
EOF

echo "done"
