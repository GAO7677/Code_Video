#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box
SWEEP_SESSION=mixdataset_physiq_objres_sweep_gpu6_20260713
WATCH_SESSION=watch_replay_sourceaware_physiq3_gpu6_20260713
WATCH_SCRIPT="${SCRIPT_DIR}/watch_replay_sourceaware_physiq3_checkpoints.sh"
EVAL_PHYSIQ="${SCRIPT_DIR}/AAAevalphysiq.txt"
BENCH_SH="${SCRIPT_DIR}/bench.sh"
LOG=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/physicIQ/train_stage1b_mixdataset/_sweep_logs/bench_AAAevalphysiq.log
TARGET='/step-002000/object_residual_2p0x/'

echo "[handoff] waiting for final eval entry: ${TARGET}"
until grep -Fq "${TARGET}" "${EVAL_PHYSIQ}"; do
  sleep 10
done

# Give run_one enough time to finish its final log and return to the outer loop.
sleep 5
if tmux has-session -t "${SWEEP_SESSION}" 2>/dev/null; then
  tmux kill-session -t "${SWEEP_SESSION}"
fi

echo "[handoff] starting benchmark on physical GPU 0"
CUDA_VISIBLE_DEVICES=0 \
BENCH_CUDA_VISIBLE_DEVICES=0 \
bash "${BENCH_SH}" "${EVAL_PHYSIQ}" 2>&1 | tee -a "${LOG}"

echo "[handoff] benchmark completed; restoring GPU 6 checkpoint watcher"
if ! tmux has-session -t "${WATCH_SESSION}" 2>/dev/null; then
  tmux new-session -d -s "${WATCH_SESSION}" "bash ${WATCH_SCRIPT}"
fi
