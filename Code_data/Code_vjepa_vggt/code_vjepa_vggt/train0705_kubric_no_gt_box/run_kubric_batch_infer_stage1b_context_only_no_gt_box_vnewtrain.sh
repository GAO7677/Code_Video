#!/usr/bin/env bash
# =============================================================================
# Unified launcher for Kubric batch inference.
# Preferred business-facing inputs:
# - GPU pair: GPU_PAIR=6,7
# - multiple GPU pairs: GPU_PAIRS="6,7 7,6 3,5 5,3"
# - test json txt: TEST_JSON_TXT=/data/.../test_5.txt
# - weights: WEIGHTS_ROOT=/data/.../step-001000
# - method name: METHOD_NAME=train_stage1b_kubric0708_step1000
# - output root: OUTPUT_ROOT=/data/.../train0705_kubric_test5_compare_0708
# - output frames: OUTPUT_FRAMES=49
# - ctx: CTX=8
# - multiple ctx counts: CTX=1,4,8,12,16,20
#
# Direct one-run example:
# GPU_PAIR=6,7 \
# TEST_JSON_TXT=/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
# WEIGHTS_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-001000 \
# METHOD_NAME=train_stage1b_kubric0708_step1000 \
# OUTPUT_ROOT=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0708 \
# OUTPUT_FRAMES=49 \
# CTX=8 \
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh
#
# Sweep on one GPU pair:
# GPU_PAIR=6,7 \
# TEST_JSON_TXT=/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
# WEIGHTS_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-001000 \
# METHOD_NAME=train_stage1b_kubric0708_step1000 \
# OUTPUT_ROOT=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0708_ctxn \
# OUTPUT_FRAMES=49 \
# CTX=1,4,8,12,16,20 \
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh
#
# Sweep on multiple GPU pairs:
# GPU_PAIRS="6,7 7,6 3,5 5,3" \
# TEST_JSON_TXT=/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
# WEIGHTS_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-001000 \
# METHOD_NAME=train_stage1b_kubric0708_step1000 \
# OUTPUT_ROOT=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0708_ctxn \
# OUTPUT_FRAMES=49 \
# CTX=1,4,8,12,16,20 \
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh
# =============================================================================
set -euo pipefail

PROJ=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
INFER_SCRIPT="${PROJ}/code_vjepa_vggt/train0705_kubric_no_gt_box/wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v.py"

RUN_MODE="${RUN_MODE:-}"
GPU_PAIR="${GPU_PAIR:-}"
GPU_PAIRS="${GPU_PAIRS:-}"
TEST_JSON_TXT="${TEST_JSON_TXT:-}"
METHOD_NAME="${METHOD_NAME:-}"
OUTPUT_FRAMES="${OUTPUT_FRAMES:-}"
CTX="${CTX:-}"
CTX_NUM="${CTX_NUM:-}"
CTX_NUMS="${CTX_NUMS:-}"
USER_VISIBLE_GPU_IDS="${VISIBLE_GPU_IDS:-}"
USER_INFERENCE_GPU_PAIRS="${INFERENCE_GPU_PAIRS:-}"
USER_CONTEXT_FRAMES="${CONTEXT_FRAMES:-}"
USER_CONTEXT_FRAME_VALUES="${CONTEXT_FRAME_VALUES:-}"

VISIBLE_GPU_IDS="${VISIBLE_GPU_IDS:-${GPU_PAIR:-5,6}}"
INFERENCE_GPU_PAIRS="${INFERENCE_GPU_PAIRS:-${GPU_PAIRS:-}}"
INFERENCE_DEVICES="${INFERENCE_DEVICES:-cuda:0,cuda:1}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-001000}"
INPUT_JSON_LIST_PATH="${INPUT_JSON_LIST_PATH:-${TEST_JSON_TXT:-/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt}}"
MODEL_NAME="${MODEL_NAME:-${METHOD_NAME:-train_stage1b_kubric0708_step1000}}"
MODEL_NAME_PREFIX="${MODEL_NAME_PREFIX:-${MODEL_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0708}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
OUTPUT_NUM_FRAMES="${OUTPUT_NUM_FRAMES:-${OUTPUT_FRAMES:-49}}"
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-896}"
INPUT_COVER_CROP_WIDTH="${INPUT_COVER_CROP_WIDTH:-896}"
INPUT_COVER_CROP_HEIGHT="${INPUT_COVER_CROP_HEIGHT:-512}"
CONTEXT_FRAMES="${CONTEXT_FRAMES:-${CTX_NUM:-${CTX:-8}}}"
CONTEXT_FRAME_VALUES="${CONTEXT_FRAME_VALUES:-${CTX_NUMS:-${CTX:-1,2,3,4,6,8,9,12,16,20}}}"
SAMPLING_MODE="${SAMPLING_MODE:-prefix}"
CFG_SCALE="${CFG_SCALE:-5.0}"
SEED="${SEED:-42}"
FPS="${FPS:-30}"
LIMIT="${LIMIT:-}"
FORCE="${FORCE:-0}"
OVERWRITE="${OVERWRITE:-0}"

infer_run_mode() {
  if [ -n "${RUN_MODE}" ]; then
    echo "${RUN_MODE}"
    return
  fi
  if [ -n "${GPU_PAIRS}" ] || [ -n "${USER_INFERENCE_GPU_PAIRS}" ]; then
    echo "sweep"
    return
  fi
  if [ -n "${CTX}" ]; then
    if [[ "${CTX}" == *","* ]]; then
      echo "sweep"
    else
      echo "direct"
    fi
    return
  fi
  if [ -n "${CTX_NUMS}" ] || [ -n "${USER_CONTEXT_FRAME_VALUES}" ]; then
    echo "sweep"
    return
  fi
  if [ -n "${CTX_NUM}" ] || [ -n "${USER_CONTEXT_FRAMES}" ]; then
    echo "direct"
    return
  fi
  if [ -n "${GPU_PAIR}" ] || [ -n "${USER_VISIBLE_GPU_IDS}" ]; then
    echo "direct"
    return
  fi
  echo "direct"
}

check_gpu_pair_has_faulty_gpu4() {
  local raw_pair="$1"
  local clean_pair
  local gpu_id
  clean_pair="$(echo "${raw_pair}" | tr -d '[:space:]')"
  IFS=',' read -r -a pair_gpu_ids <<< "${clean_pair}"
  for gpu_id in "${pair_gpu_ids[@]}"; do
    if [ "${gpu_id}" = "4" ]; then
      echo "ERROR: gpu4 故障, 禁止使用。当前 GPU pair=${raw_pair}" >&2
      exit 1
    fi
  done
}

prepare_launch_layout() {
  local raw_pair="$1"
  local requested_inference_devices="$2"
  local -n out_visible_gpu_ids="$3"
  local -n out_inference_devices="$4"
  local clean_pair
  local gpu_id
  local -a pair_gpu_ids=()
  local -a unique_gpu_ids=()

  clean_pair="$(echo "${raw_pair}" | tr -d '[:space:]')"
  IFS=',' read -r -a pair_gpu_ids <<< "${clean_pair}"
  for gpu_id in "${pair_gpu_ids[@]}"; do
    if [ -z "${gpu_id}" ]; then
      continue
    fi
    if [ "${#unique_gpu_ids[@]}" -eq 0 ] || [ "${unique_gpu_ids[-1]}" != "${gpu_id}" ]; then
      unique_gpu_ids+=("${gpu_id}")
    fi
  done

  if [ "${#unique_gpu_ids[@]}" -eq 0 ]; then
    echo "ERROR: GPU pair 解析后为空: ${raw_pair}" >&2
    exit 1
  fi

  out_visible_gpu_ids="$(IFS=,; echo "${unique_gpu_ids[*]}")"
  if [ "${#unique_gpu_ids[@]}" -lt 2 ]; then
    out_inference_devices="none"
  else
    out_inference_devices="${requested_inference_devices}"
  fi
}

normalize_ctx_values() {
  local raw_list="$1"
  local -n out_array="$2"
  local raw_ctx
  local ctx
  out_array=()
  IFS=',' read -r -a raw_ctx_values <<< "${raw_list}"
  if [ "${#raw_ctx_values[@]}" -eq 0 ]; then
    echo "ERROR: CONTEXT_FRAME_VALUES 不能为空" >&2
    exit 1
  fi
  for raw_ctx in "${raw_ctx_values[@]}"; do
    ctx="$(echo "${raw_ctx}" | xargs)"
    if [ -z "${ctx}" ]; then
      continue
    fi
    if ! [[ "${ctx}" =~ ^[0-9]+$ ]]; then
      echo "ERROR: 非法 context 长度: ${ctx}" >&2
      exit 1
    fi
    if [ "${ctx}" -le 0 ]; then
      echo "ERROR: context 长度必须为正整数，收到 ${ctx}" >&2
      exit 1
    fi
    out_array+=("${ctx}")
  done
  if [ "${#out_array[@]}" -eq 0 ]; then
    echo "ERROR: CONTEXT_FRAME_VALUES 解析后为空" >&2
    exit 1
  fi
}

run_one_inference() {
  local gpu_pair="$1"
  local context_frames="$2"
  local run_output_root="$3"
  local run_model_name="$4"
  local step_output_dir_name="${5:-}"
  local launch_visible_gpu_ids
  local launch_inference_devices
  local -a cmd

  check_gpu_pair_has_faulty_gpu4 "${gpu_pair}"
  prepare_launch_layout "${gpu_pair}" "${INFERENCE_DEVICES}" launch_visible_gpu_ids launch_inference_devices
  cmd=(
    env
    PYTHONNOUSERSITE=1
    PYTHONPATH="${PROJ}:${DIFFSYNTH_ROOT}"
    CUDA_VISIBLE_DEVICES="${launch_visible_gpu_ids}"
    "${PYTHON_BIN}"
    "${INFER_SCRIPT}"
    --weights-root "${WEIGHTS_ROOT}"
    --input-json-list-path "${INPUT_JSON_LIST_PATH}"
    --model-name "${run_model_name}"
    --output-root "${run_output_root}"
    --height "${HEIGHT}"
    --width "${WIDTH}"
    --input-cover-crop-width "${INPUT_COVER_CROP_WIDTH}"
    --input-cover-crop-height "${INPUT_COVER_CROP_HEIGHT}"
    --context-frames "${context_frames}"
    --sampling-mode "${SAMPLING_MODE}"
    --num-inference-steps "${NUM_INFERENCE_STEPS}"
    --cfg-scale "${CFG_SCALE}"
    --seed "${SEED}"
    --fps "${FPS}"
    --output-num-frames "${OUTPUT_NUM_FRAMES}"
  )

  if [ -n "${launch_inference_devices}" ] && [ "${launch_inference_devices}" != "none" ]; then
    cmd+=(--inference-devices "${launch_inference_devices}")
  fi
  if [ -n "${step_output_dir_name}" ]; then
    cmd+=(--step-output-dir-name "${step_output_dir_name}")
  fi
  if [ -n "${LIMIT}" ]; then
    cmd+=(--limit "${LIMIT}")
  fi
  if [ "${FORCE}" = "1" ]; then
    cmd+=(--force)
  fi
  if [ "${OVERWRITE}" = "1" ]; then
    cmd+=(--overwrite)
  fi

  echo "[kubric-batch] gpu_pair=${gpu_pair} context_frames=${context_frames}"
  echo "[kubric-batch] cuda_visible_devices=${launch_visible_gpu_ids} inference_devices=${launch_inference_devices}"
  echo "[kubric-batch] output=${run_output_root}"
  echo "[kubric-batch] model_name=${run_model_name}"
  echo "[kubric-batch] command: ${cmd[*]}"
  "${cmd[@]}"
}

run_sweep_for_pair() {
  local gpu_pair="$1"
  shift
  local ctx
  local ctx_tag
  local ctx_output_root
  local ctx_model_name

  for ctx in "$@"; do
    printf -v ctx_tag "ctx%02d" "${ctx}"
    ctx_output_root="${OUTPUT_ROOT}/${ctx_tag}"
    ctx_model_name="${MODEL_NAME_PREFIX}_${ctx_tag}"
    run_one_inference "${gpu_pair}" "${ctx}" "${ctx_output_root}" "${ctx_model_name}"
  done
}

run_direct_mode() {
  if [ -n "${INFERENCE_GPU_PAIRS}" ]; then
    echo "ERROR: RUN_MODE=direct 时不要设置 INFERENCE_GPU_PAIRS" >&2
    exit 1
  fi
  run_one_inference "${VISIBLE_GPU_IDS}" "${CONTEXT_FRAMES}" "${OUTPUT_ROOT}" "${MODEL_NAME}" "__METHOD_NAME__"
}

run_sweep_mode() {
  mkdir -p "${OUTPUT_ROOT}"
  declare -a ctx_values=()
  normalize_ctx_values "${CONTEXT_FRAME_VALUES}" ctx_values

  if [ -n "${INFERENCE_GPU_PAIRS}" ]; then
    read -r -a gpu_pairs <<< "${INFERENCE_GPU_PAIRS}"
    if [ "${#gpu_pairs[@]}" -eq 0 ]; then
      echo "ERROR: INFERENCE_GPU_PAIRS 解析后为空" >&2
      exit 1
    fi

    declare -a child_pids=()
    declare -a active_gpu_pairs=()
    local pair_index
    local ctx_index
    local gpu_pair
    for pair_index in "${!gpu_pairs[@]}"; do
      gpu_pair="${gpu_pairs[pair_index]}"
      declare -a assigned_ctx_values=()
      for ctx_index in "${!ctx_values[@]}"; do
        if [ $((ctx_index % ${#gpu_pairs[@]})) -eq "${pair_index}" ]; then
          assigned_ctx_values+=("${ctx_values[ctx_index]}")
        fi
      done
      if [ "${#assigned_ctx_values[@]}" -eq 0 ]; then
        continue
      fi

      active_gpu_pairs+=("${gpu_pair}")
      (
        run_sweep_for_pair "${gpu_pair}" "${assigned_ctx_values[@]}"
      ) &
      child_pids+=("$!")
    done

    if [ "${#child_pids[@]}" -eq 0 ]; then
      echo "ERROR: 没有可运行的 GPU pair / ctx 任务" >&2
      exit 1
    fi

    echo "[kubric-batch] running ${#child_pids[@]} sweep worker(s)"
    echo "[kubric-batch] gpu_pairs=${active_gpu_pairs[*]}"

    declare -a failed_workers=()
    local child_pid
    local status
    for pair_index in "${!child_pids[@]}"; do
      child_pid="${child_pids[pair_index]}"
      gpu_pair="${active_gpu_pairs[pair_index]}"
      if wait "${child_pid}"; then
        echo "[kubric-batch] worker done gpu_pair=${gpu_pair}"
      else
        status=$?
        echo "[kubric-batch] worker failed gpu_pair=${gpu_pair} exit_code=${status}" >&2
        failed_workers+=("${gpu_pair}:${status}")
      fi
    done

    if [ "${#failed_workers[@]}" -gt 0 ]; then
      echo "[kubric-batch] failed_workers=${failed_workers[*]}" >&2
      exit 1
    fi
    echo "[kubric-batch] sweep done. outputs under ${OUTPUT_ROOT}"
    return
  fi

  run_sweep_for_pair "${VISIBLE_GPU_IDS}" "${ctx_values[@]}"
  echo "[kubric-batch] sweep done. outputs under ${OUTPUT_ROOT}"
}

RUN_MODE="$(infer_run_mode)"

case "${RUN_MODE}" in
  direct)
    run_direct_mode
    ;;
  sweep)
    run_sweep_mode
    ;;
  *)
    echo "ERROR: RUN_MODE 只支持 direct 或 sweep，收到 ${RUN_MODE}" >&2
    exit 1
    ;;
esac
