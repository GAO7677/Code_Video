#!/usr/bin/env bash
set -euo pipefail

BASE=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box
PROJ=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
INFER_SCRIPT="${BASE}/wan_stage1b_scheme_c_entity_binding_intervention_v2v.py"
CHECKPOINT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_raw49f_scheme_c_entity_caption_physical_fresh_20260714T174707Z/checkpoints/step-003500
INPUT_LIST=/data/gaoya/agent-data/cache/stage1b_scheme_c_entity_caption_physical_fresh_val/crop_top60px_single_input_json.txt
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/agent-data/outputs/AAA_physv/entity_binding_intervention_crop_top60px_step3500_20260715}"
TMP_ROOT=/data/gaoya/agent-data/cache/t/entity_binding_intervention_crop_top60px_step3500
GPU_ID="${GPU_ID:-6}"

mkdir -p "${OUTPUT_ROOT}/logs" "${TMP_ROOT}"

run_one() {
  local mode="$1"
  local scale="$2"
  local tag="${mode}_gate_${scale/./p}x"
  local out="${OUTPUT_ROOT}/${tag}"
  echo "[matrix] start mode=${mode} gate_scale=${scale} output=${out}"
  env \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH="${PROJ}:${DIFFSYNTH_ROOT}" \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TMPDIR="${TMP_ROOT}" TMP="${TMP_ROOT}" TEMP="${TMP_ROOT}" \
    "${PYTHON_BIN}" "${INFER_SCRIPT}" \
      --weights-root "${CHECKPOINT}" \
      --input-json-list-path "${INPUT_LIST}" \
      --model-name "scheme_c_entity_step3500_crop_top60px_${tag}" \
      --output-root "${out}" \
      --step-output-dir-name results \
      --height 512 --width 896 \
      --context-frames 8 --num-frames 49 \
      --num-inference-steps 40 --cfg-scale 5.0 --seed 42 --fps 30 \
      --object-branch-residual-scale 1.0 \
      --entity-binding-map-mode "${mode}" \
      --entity-binding-gate-scale "${scale}" \
      --force \
      2>&1 | tee "${OUTPUT_ROOT}/logs/${tag}.log"
}

run_one correct 0.0
run_one disabled 1.0
run_one correct 1.0
run_one correct 2.0
run_one correct 4.0
run_one swapped 1.0
run_one swapped 2.0
run_one swapped 4.0

echo "[matrix] complete output=${OUTPUT_ROOT}"
