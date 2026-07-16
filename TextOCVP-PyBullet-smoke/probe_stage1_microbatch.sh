#!/usr/bin/env bash
set -uo pipefail

PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
SCRIPT=/home/gaoya/Code_Video/TextOCVP-PyBullet-smoke/probe_stage1_microbatch.py
INDEX_ROOT=/data/gaoya/AAA_test_video/0623_savi/indices
OUTPUT_ROOT=/data/gaoya/AAA_test_video/0623_savi/outputs/memory_probe
mkdir -p "${OUTPUT_ROOT}"

for micro_batch in 64 32 16 8 4; do
  log="${OUTPUT_ROOT}/micro_global_batch_${micro_batch}.log"
  echo "[probe] micro_global_batch=${micro_batch}"
  if CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONNOUSERSITE=1 \
      "${PYTHON}" "${SCRIPT}" \
        --index-root "${INDEX_ROOT}" \
        --dataset-mode mixed \
        --micro-global-batch-size "${micro_batch}" \
        2>&1 | tee "${log}"; then
    echo "${micro_batch}" > "${OUTPUT_ROOT}/selected_micro_global_batch.txt"
    echo "[probe] selected=${micro_batch} accumulation=$((64 / micro_batch))"
    exit 0
  fi
  echo "[probe] failed=${micro_batch}"
done

echo "No tested micro batch fits on GPUs 0,1,2,3" >&2
exit 1
