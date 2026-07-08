#!/usr/bin/env bash
# =============================================================================
# Run command example:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_batch_ctx_sweep_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v.sh
#
# Single-pair example:
# VISIBLE_GPU_IDS=6,7 \
# INFERENCE_DEVICES=cuda:0,cuda:1 \
# CONTEXT_FRAME_VALUES=1,4,8,12,16,20 \
# OUTPUT_ROOT=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0708_ctxn \
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_batch_ctx_sweep_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v.sh
#





# Multi-pair example:
# INFERENCE_GPU_PAIRS="3,7 5,7 " \
# INFERENCE_DEVICES=cuda:0,cuda:1 \
# CONTEXT_FRAME_VALUES=1,4,8,12,16,20 \
# OUTPUT_ROOT=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0708_ctxn \
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_batch_ctx_sweep_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v.sh
# =============================================================================
#
# 说明:
# - 默认前台运行
# - 如果提供 INFERENCE_GPU_PAIRS="6,7 7,6 3,5 5,3"，会在当前前台脚本内启动多个子进程并 wait
# - 多 pair 模式下，按 round-robin 将不同 ctx 值分配给不同 GPU pair
# - 单个 worker 如果触发 OOM，会直接退出对应 worker，不继续占用 GPU；其他 worker 继续跑
# - 每个 context 长度落到单独文件夹: <OUTPUT_ROOT>/ctxXX/
# - 禁止使用 gpu4
# =============================================================================
set -euo pipefail

PROJ=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
INFER_SCRIPT="${PROJ}/code_vjepa_vggt/train0705_kubric_no_gt_box/wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v.py"

VISIBLE_GPU_IDS="${VISIBLE_GPU_IDS:-5,6}"
INFERENCE_GPU_PAIRS="${INFERENCE_GPU_PAIRS:-}"
INFERENCE_DEVICES="${INFERENCE_DEVICES:-cuda:0,cuda:1}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-001000}"
INPUT_JSON_LIST_PATH="${INPUT_JSON_LIST_PATH:-/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt}"
MODEL_NAME_PREFIX="${MODEL_NAME_PREFIX:-train_stage1b_kubric0708_step1000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0708_ctxn}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
OUTPUT_NUM_FRAMES="${OUTPUT_NUM_FRAMES:-49}"
CONTEXT_FRAME_VALUES="${CONTEXT_FRAME_VALUES:-1,2,3,4,6,8,9,12,16,20}"
FORCE="${FORCE:-0}"

check_gpu_pair_has_faulty_gpu4() {
  local RAW_PAIR="$1"
  local CLEAN_PAIR
  local GPU_ID
  CLEAN_PAIR="$(echo "${RAW_PAIR}" | tr -d '[:space:]')"
  IFS=',' read -r -a PAIR_GPU_IDS <<< "${CLEAN_PAIR}"
  for GPU_ID in "${PAIR_GPU_IDS[@]}"; do
    if [ "${GPU_ID}" = "4" ]; then
      echo "ERROR: gpu4 故障, 禁止使用。当前 GPU pair=${RAW_PAIR}" >&2
      exit 1
    fi
  done
}

normalize_ctx_values() {
  local RAW_LIST="$1"
  local -n OUT_ARRAY="$2"
  local RAW_CTX
  local CTX
  OUT_ARRAY=()
  IFS=',' read -r -a RAW_CTX_VALUES <<< "${RAW_LIST}"
  if [ "${#RAW_CTX_VALUES[@]}" -eq 0 ]; then
    echo "ERROR: CONTEXT_FRAME_VALUES 不能为空" >&2
    exit 1
  fi
  for RAW_CTX in "${RAW_CTX_VALUES[@]}"; do
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
    OUT_ARRAY+=("${CTX}")
  done
  if [ "${#OUT_ARRAY[@]}" -eq 0 ]; then
    echo "ERROR: CONTEXT_FRAME_VALUES 解析后为空" >&2
    exit 1
  fi
}

run_ctx_values_for_pair() {
  local GPU_PAIR="$1"
  shift
  local CTX
  local CTX_TAG
  local CTX_OUTPUT_ROOT
  local CTX_MODEL_NAME
  local -a CMD

  check_gpu_pair_has_faulty_gpu4 "${GPU_PAIR}"
  for CTX in "$@"; do
    printf -v CTX_TAG "ctx%02d" "${CTX}"
    CTX_OUTPUT_ROOT="${OUTPUT_ROOT}/${CTX_TAG}"
    CTX_MODEL_NAME="${MODEL_NAME_PREFIX}_${CTX_TAG}"

    CMD=(
      env
      PYTHONPATH="${PROJ}:${DIFFSYNTH_ROOT}"
      CUDA_VISIBLE_DEVICES="${GPU_PAIR}"
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

    echo "[ctx-sweep] gpu_pair=${GPU_PAIR} running ${CTX_TAG}"
    echo "[ctx-sweep] gpu_pair=${GPU_PAIR} output=${CTX_OUTPUT_ROOT}"
    echo "[ctx-sweep] gpu_pair=${GPU_PAIR} command: ${CMD[*]}"
    "${CMD[@]}"
  done
}

mkdir -p "${OUTPUT_ROOT}"

declare -a CTX_VALUES=()
normalize_ctx_values "${CONTEXT_FRAME_VALUES}" CTX_VALUES

if [ -n "${INFERENCE_GPU_PAIRS}" ]; then
  read -r -a GPU_PAIRS <<< "${INFERENCE_GPU_PAIRS}"
  if [ "${#GPU_PAIRS[@]}" -eq 0 ]; then
    echo "ERROR: INFERENCE_GPU_PAIRS 解析后为空" >&2
    exit 1
  fi

  declare -a CHILD_PIDS=()
  declare -a ACTIVE_GPU_PAIRS=()

  for PAIR_INDEX in "${!GPU_PAIRS[@]}"; do
    GPU_PAIR="${GPU_PAIRS[PAIR_INDEX]}"
    declare -a ASSIGNED_CTX_VALUES=()
    for CTX_INDEX in "${!CTX_VALUES[@]}"; do
      if [ $((CTX_INDEX % ${#GPU_PAIRS[@]})) -eq "${PAIR_INDEX}" ]; then
        ASSIGNED_CTX_VALUES+=("${CTX_VALUES[CTX_INDEX]}")
      fi
    done
    if [ "${#ASSIGNED_CTX_VALUES[@]}" -eq 0 ]; then
      continue
    fi

    ACTIVE_GPU_PAIRS+=("${GPU_PAIR}")
    (
      run_ctx_values_for_pair "${GPU_PAIR}" "${ASSIGNED_CTX_VALUES[@]}"
    ) &
    CHILD_PIDS+=("$!")
  done

  if [ "${#CHILD_PIDS[@]}" -eq 0 ]; then
    echo "ERROR: 没有可运行的 GPU pair / ctx 任务" >&2
    exit 1
  fi

  echo "[ctx-sweep] running ${#CHILD_PIDS[@]} parallel worker(s)"
  echo "[ctx-sweep] gpu_pairs=${ACTIVE_GPU_PAIRS[*]}"

  declare -a FAILED_WORKERS=()
  for INDEX in "${!CHILD_PIDS[@]}"; do
    CHILD_PID="${CHILD_PIDS[INDEX]}"
    GPU_PAIR="${ACTIVE_GPU_PAIRS[INDEX]}"
    if wait "${CHILD_PID}"; then
      echo "[ctx-sweep] worker done gpu_pair=${GPU_PAIR}"
    else
      STATUS=$?
      echo "[ctx-sweep] worker failed gpu_pair=${GPU_PAIR} exit_code=${STATUS}" >&2
      FAILED_WORKERS+=("${GPU_PAIR}:${STATUS}")
    fi
  done

  if [ "${#FAILED_WORKERS[@]}" -gt 0 ]; then
    echo "[ctx-sweep] failed_workers=${FAILED_WORKERS[*]}" >&2
    exit 1
  fi

  echo "[ctx-sweep] done. outputs under ${OUTPUT_ROOT}"
  exit 0
fi

check_gpu_pair_has_faulty_gpu4 "${VISIBLE_GPU_IDS}"
run_ctx_values_for_pair "${VISIBLE_GPU_IDS}" "${CTX_VALUES[@]}"

echo "[ctx-sweep] done. outputs under ${OUTPUT_ROOT}"
