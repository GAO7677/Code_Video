#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box
INFER_SH="${SCRIPT_DIR}/run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh"
BENCH_SH="${SCRIPT_DIR}/bench.sh"
CHECKPOINT_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_raw49f_kubric_openvid_replay_sourceaware_fp32gate_fixedctx8_init3500_save500_keepall_20260713T090024Z/checkpoints
INPUT_LIST=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt
OUTPUT_ROOT=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/physicIQ/train_stage1b_mixdataset
EVAL_ALL="${SCRIPT_DIR}/AAAeval.txt"
EVAL_PHYSIQ="${SCRIPT_DIR}/AAAevalphysiq.txt"
WATCH_SESSION=watch_replay_sourceaware_physiq3_gpu6_20260713
WATCH_SCRIPT="${SCRIPT_DIR}/watch_replay_sourceaware_physiq3_checkpoints.sh"
RUN_LOG_ROOT="${OUTPUT_ROOT}/_sweep_logs"

mkdir -p "${OUTPUT_ROOT}" "${RUN_LOG_ROOT}"

watcher_was_present=0
if tmux has-session -t "${WATCH_SESSION}" 2>/dev/null; then
  watcher_was_present=1
  tmux send-keys -t "${WATCH_SESSION}" C-c
  sleep 5
fi

resume_watcher() {
  if [[ "${watcher_was_present}" == "1" ]]; then
    if tmux has-session -t "${WATCH_SESSION}" 2>/dev/null; then
      tmux send-keys -t "${WATCH_SESSION}" "bash ${WATCH_SCRIPT}" Enter
    else
      tmux new-session -d -s "${WATCH_SESSION}" "bash ${WATCH_SCRIPT}"
    fi
  fi
}
trap resume_watcher EXIT

append_unique() {
  local line="$1"
  local file="$2"
  touch "${file}"
  if ! grep -Fqx -- "${line}" "${file}"; then
    printf '%s\n' "${line}" >> "${file}"
  fi
}

run_one() {
  local step="$1"
  local scale="$2"
  local scale_tag="${scale/./p}"
  local checkpoint="${CHECKPOINT_ROOT}/${step}"
  local combo_root="${OUTPUT_ROOT}/${step}/object_residual_${scale_tag}x"
  local method="mixdataset_${step}_object_residual_${scale_tag}x"
  local log="${RUN_LOG_ROOT}/${step}_object_residual_${scale_tag}x.log"
  local leaf

  if [[ ! -s "${checkpoint}/checkpoint.safetensors" ]]; then
    echo "[sweep] missing checkpoint: ${checkpoint}" >&2
    return 1
  fi

  mkdir -p "${combo_root}"
  echo "[sweep] start step=${step} scale=${scale} output=${combo_root}"
  GPU_PAIR="6,6" \
  TEST_JSON_TXT="${INPUT_LIST}" \
  WEIGHTS_ROOT="${checkpoint}" \
  METHOD_NAME="${method}" \
  OUTPUT_ROOT="${combo_root}" \
  OUTPUT_FRAMES=49 \
  CTX=8 \
  NUM_INFERENCE_STEPS=40 \
  CFG_SCALE=5.0 \
  SEED=42 \
  COMPACT_OBJECT_CONTEXT_SLOTS=1 \
  OBJECT_ADAPTER_MLP_RESIDUAL_MAX_RATIO=3.0 \
  OBJECT_BRANCH_RESIDUAL_SCALE="${scale}" \
  OBJECT_BRANCH_RATIO_GUARD_MAX_RATIO=0.20 \
  OBJECT_BRANCH_RATIO_GUARD_MAX_BLOCK_ID=-1 \
  bash "${INFER_SH}" 2>&1 | tee -a "${log}"

  leaf="$(find "${combo_root}" -mindepth 1 -maxdepth 2 -type f -name result.json -printf '%h\n' | sort -u | head -n 1)"
  if [[ -z "${leaf}" ]]; then
    echo "[sweep] no result leaf found under ${combo_root}" >&2
    return 1
  fi
  append_unique "${leaf}" "${EVAL_ALL}"
  append_unique "${leaf}" "${EVAL_PHYSIQ}"
  echo "[sweep] completed step=${step} scale=${scale} leaf=${leaf}"
}

for step in step-001000 step-001500 step-002000; do
  for scale in 1.0 1.5 2.0; do
    run_one "${step}" "${scale}"
  done
done

echo "[sweep] all inference groups completed; starting metrics"
CUDA_VISIBLE_DEVICES=6 \
BENCH_CUDA_VISIBLE_DEVICES=6 \
bash "${BENCH_SH}" "${EVAL_PHYSIQ}" 2>&1 | tee -a "${RUN_LOG_ROOT}/bench_AAAevalphysiq.log"

echo "[sweep] inference and metrics completed"
