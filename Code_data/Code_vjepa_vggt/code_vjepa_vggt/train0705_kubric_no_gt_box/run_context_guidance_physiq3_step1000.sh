#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box
RUNNER="${ROOT}/run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh"
INFER_SCRIPT="${ROOT}/wan_stage1b_context_guidance_guard_v2v.py"

GPU_PAIR="${GPU_PAIR:-6,6}"
INPUT_LIST="${INPUT_LIST:-${ROOT}/context_guidance_physiq_cases.txt}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_raw49f_kubric_openvid_replay_sourceaware_fp32gate_fixedctx8_init3500_save500_keepall_20260713T090024Z/checkpoints/step-001000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/agent-data/outputs/AAA_physv/context_guidance_physiq3_step1000_20260713}"
MODES="${MODES:-text_video_baseline positive_text_off negative_text_off video_only low_text_cfg anti_duplicate_prompt adaptive_context_guard}"
CONTEXT_GUARD_CFG_SCALE="${CONTEXT_GUARD_CFG_SCALE:-2.5}"
CONTEXT_GUARD_SCORE_THRESHOLD="${CONTEXT_GUARD_SCORE_THRESHOLD:-0.20}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
DRY_RUN="${DRY_RUN:-0}"
MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB:-44000}"

if [ "${DRY_RUN}" != "1" ]; then
  primary_gpu="${GPU_PAIR%%,*}"
  free_gpu_mib="$(
    nvidia-smi --id="${primary_gpu}" --query-gpu=memory.free \
      --format=csv,noheader,nounits | head -n 1 | tr -d '[:space:]'
  )"
  if [ -z "${free_gpu_mib}" ] || [ "${free_gpu_mib}" -lt "${MIN_FREE_GPU_MIB}" ]; then
    echo "ERROR: GPU ${primary_gpu} free memory ${free_gpu_mib:-unknown} MiB; " \
      "at least ${MIN_FREE_GPU_MIB} MiB is required." >&2
    exit 1
  fi
fi

for mode in ${MODES}; do
  echo "[context-guidance-sweep] mode=${mode} gpu_pair=${GPU_PAIR}"
  if [ "${DRY_RUN}" = "1" ]; then
    echo "[context-guidance-sweep] dry_run=1 output=${OUTPUT_ROOT}/${mode}"
    continue
  fi
  GPU_PAIR="${GPU_PAIR}" \
  TEST_JSON_TXT="${INPUT_LIST}" \
  WEIGHTS_ROOT="${WEIGHTS_ROOT}" \
  METHOD_NAME="context_guidance_step1000_${mode}" \
  OUTPUT_ROOT="${OUTPUT_ROOT}/${mode}" \
  OUTPUT_FRAMES=49 \
  CTX=8 \
  NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS}" \
  CFG_SCALE=5.0 \
  SEED=42 \
  COMPACT_OBJECT_CONTEXT_SLOTS=1 \
  OBJECT_ADAPTER_MLP_RESIDUAL_MAX_RATIO=3.0 \
  OBJECT_BRANCH_RATIO_GUARD_MAX_RATIO=0.30 \
  OBJECT_BRANCH_RATIO_GUARD_MAX_BLOCK_ID=-1 \
  INFER_SCRIPT_OVERRIDE="${INFER_SCRIPT}" \
  CONDITION_MODE="${mode}" \
  CONTEXT_GUARD_CFG_SCALE="${CONTEXT_GUARD_CFG_SCALE}" \
  CONTEXT_GUARD_SCORE_THRESHOLD="${CONTEXT_GUARD_SCORE_THRESHOLD}" \
  bash "${RUNNER}"
done
