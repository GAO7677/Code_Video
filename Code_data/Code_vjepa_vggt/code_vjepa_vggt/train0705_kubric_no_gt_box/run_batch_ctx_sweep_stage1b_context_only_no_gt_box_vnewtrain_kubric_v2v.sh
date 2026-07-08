#!/usr/bin/env bash
# =============================================================================
# Run command example:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_batch_ctx_sweep_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v.sh
#
# Optional overrides:
# VISIBLE_GPU_IDS=5,6 \
# INFERENCE_DEVICES=cuda:0,cuda:1 \
# CONTEXT_FRAME_VALUES=1,2,3,4,6,8,9,12,16,20 \
# OUTPUT_ROOT=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0708_ctxn \
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_batch_ctx_sweep_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v.sh
#
# 说明:
# - 前台逐个 context 长度运行，不使用 nohup / 后台
# - 默认扫训练时常用的正整数 context 候选: 1,2,3,4,6,8,9,12,16,20
# - 结果根目录固定在 /data/gaoya/.../train0705_kubric_test5_compare_0708_ctxn
# - 每个 context 长度落到单独文件夹: <OUTPUT_ROOT>/ctxXX/
# =============================================================================
set -euo pipefail

PROJ=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
INFER_SCRIPT="${PROJ}/code_vjepa_vggt/train0705_kubric_no_gt_box/wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v.py"

VISIBLE_GPU_IDS="${VISIBLE_GPU_IDS:-5,6}"
INFERENCE_DEVICES="${INFERENCE_DEVICES:-cuda:0,cuda:1}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-001000}"
INPUT_JSON_LIST_PATH="${INPUT_JSON_LIST_PATH:-/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt}"
MODEL_NAME_PREFIX="${MODEL_NAME_PREFIX:-train_stage1b_kubric0708_step1000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0708_ctxn}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
OUTPUT_NUM_FRAMES="${OUTPUT_NUM_FRAMES:-49}"
CONTEXT_FRAME_VALUES="${CONTEXT_FRAME_VALUES:-1,2,3,4,6,8,9,12,16,20}"
FORCE="${FORCE:-0}"

if [[ "${VISIBLE_GPU_IDS}" == *"4"* ]]; then
  echo "ERROR: gpu4 故障, 禁止使用。当前 VISIBLE_GPU_IDS=${VISIBLE_GPU_IDS}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

IFS=',' read -r -a CTX_VALUES <<< "${CONTEXT_FRAME_VALUES}"
if [ "${#CTX_VALUES[@]}" -eq 0 ]; then
  echo "ERROR: CONTEXT_FRAME_VALUES 不能为空" >&2
  exit 1
fi

for RAW_CTX in "${CTX_VALUES[@]}"; do
  CTX="$(echo "${RAW_CTX}" | xargs)"
  if [ -z "${CTX}" ]; then
    continue
  fi
  if ! [[ "${CTX}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: 非法 context 长度: ${CTX}" >&2
    exit 1
  fi
  if [ "${CTX}" -le 0 ]; then
    echo "ERROR: 当前批量推理 sweep 只支持正整数 context 长度，收到 ${CTX}" >&2
    exit 1
  fi

  printf -v CTX_TAG "ctx%02d" "${CTX}"
  CTX_OUTPUT_ROOT="${OUTPUT_ROOT}/${CTX_TAG}"
  CTX_MODEL_NAME="${MODEL_NAME_PREFIX}_${CTX_TAG}"

  CMD=(
    env
    PYTHONPATH="${PROJ}:${DIFFSYNTH_ROOT}"
    CUDA_VISIBLE_DEVICES="${VISIBLE_GPU_IDS}"
    "${PYTHON_BIN}"
    "${INFER_SCRIPT}"
    --weights-root "${WEIGHTS_ROOT}"
    --input-json-list-path "${INPUT_JSON_LIST_PATH}"
    --model-name "${CTX_MODEL_NAME}"
    --output-root "${CTX_OUTPUT_ROOT}"
    --num-inference-steps "${NUM_INFERENCE_STEPS}"
    --output-num-frames "${OUTPUT_NUM_FRAMES}"
    --context-frames "${CTX}"
  )

  if [ -n "${INFERENCE_DEVICES}" ] && [ "${INFERENCE_DEVICES}" != "none" ]; then
    CMD+=(--inference-devices "${INFERENCE_DEVICES}")
  fi

  if [ "${FORCE}" = "1" ]; then
    CMD+=(--force)
  fi

  echo "[ctx-sweep] running ${CTX_TAG}"
  echo "[ctx-sweep] output=${CTX_OUTPUT_ROOT}"
  echo "[ctx-sweep] command: ${CMD[*]}"
  "${CMD[@]}"
done

echo "[ctx-sweep] done. outputs under ${OUTPUT_ROOT}"
