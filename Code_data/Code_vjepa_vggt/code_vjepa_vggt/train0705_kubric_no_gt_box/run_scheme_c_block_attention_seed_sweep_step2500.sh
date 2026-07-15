#!/usr/bin/env bash
set -euo pipefail

BASE=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box
RUNNER="${BASE}/run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh"
INFER_SCRIPT="${BASE}/visualize_scheme_c_text_noun_attention_x0_v2v.py"
CHECKPOINT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_raw49f_scheme_c_entity_caption_physical_fresh_20260714T174707Z/checkpoints/step-002500
CASE_LIST=/data/gaoya/agent-data/cache/stage1b_scheme_c_entity_mapping_20260715/smoke_ball_block.txt
OUTPUT_ROOT=/data/gaoya/agent-data/outputs/AAA_physv/scheme_c_block_attention_seed_sweep_step2500_residual1p0x_20260715

run_worker() {
  local gpu="$1"
  local seed="$2"
  local seed_tag
  seed_tag="$(printf '%06d' "${seed}")"
  local output_dir="${OUTPUT_ROOT}/seed_${seed_tag}"
  local log="${OUTPUT_ROOT}/_logs/seed_${seed_tag}_gpu${gpu}.log"
  local method="scheme_c_block_attention_step2500_residual1p0x_seed${seed_tag}"

  mkdir -p "${output_dir}" "${OUTPUT_ROOT}/_logs"
  printf '%s\n' \
    "checkpoint=${CHECKPOINT}" \
    "input_json_list=${CASE_LIST}" \
    "seed=${seed}" \
    "object_branch_residual_scale=1.0" \
    "cfg_scale=5.0" \
    "num_inference_steps=40" \
    "capture_progress_indices=0,10,20,30,39" \
    "save_all_attention_maps=1" \
    "shared_context_future_scale=1" \
    > "${output_dir}/seed_sweep_config.txt"

  GPU_PAIR="${gpu},${gpu}" \
  AUTO_SPLIT_INPUT=0 \
  TEST_JSON_TXT="${CASE_LIST}" \
  WEIGHTS_ROOT="${CHECKPOINT}" \
  METHOD_NAME="${method}" \
  OUTPUT_ROOT="${output_dir}" \
  OUTPUT_FRAMES=49 \
  CTX=8 \
  NUM_INFERENCE_STEPS=40 \
  CFG_SCALE=5.0 \
  SEED="${seed}" \
  FORCE=1 \
  COMPACT_OBJECT_CONTEXT_SLOTS=1 \
  OBJECT_ADAPTER_MLP_RESIDUAL_MAX_RATIO=3.0 \
  OBJECT_BRANCH_RESIDUAL_SCALE=1.0 \
  OBJECT_BRANCH_RATIO_GUARD_MAX_RATIO=0.30 \
  OBJECT_BRANCH_RATIO_GUARD_MAX_BLOCK_ID=-1 \
  INFER_SCRIPT_OVERRIDE="${INFER_SCRIPT}" \
  EXTRA_INFER_ARGS="--attention-capture-progress-indices auto5 --attention-query-chunk 256 --attention-save-all-maps 1 --attention-shared-context-future-scale 1" \
  bash "${RUNNER}" 2>&1 | tee -a "${log}"

  touch "${output_dir}/worker_complete.txt"
}

if [[ "${1:-}" == "--worker" ]]; then
  run_worker "$2" "$3"
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}/_logs"

# GPU4 is disabled by the batch runner. Keep the two GPU3 seeds serial so the
# launcher remains reproducible without oversubscribing a 48 GiB device.
session="scheme_c_block_attn_seed000007_then000123_gpu3_20260715"
if tmux has-session -t "${session}" 2>/dev/null; then
  echo "session already exists: ${session}" >&2
else
  tmux new-session -d -s "${session}" \
    "bash '${BASH_SOURCE[0]}' --worker 3 7 && bash '${BASH_SOURCE[0]}' --worker 3 123"
  echo "started ${session}: gpu=3 seeds=7,123 (serial)"
fi

session="scheme_c_block_attn_seed002026_gpu5_20260715"
if tmux has-session -t "${session}" 2>/dev/null; then
  echo "session already exists: ${session}" >&2
else
  tmux new-session -d -s "${session}" \
    "bash '${BASH_SOURCE[0]}' --worker 5 2026"
  echo "started ${session}: gpu=5 seed=2026"
fi
