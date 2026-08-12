#!/usr/bin/env bash
set -euo pipefail

# Finish the latest3350 Stage-3 metric backfill after video generation ends.
# The four disjoint head scopes run concurrently (two workers on each GPU),
# write isolated caches, and are consolidated into the canonical reports.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
EXPERIMENT_ROOT="/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1"
RESULT_ROOT="${EXPERIMENT_ROOT}/stage3_discovery_videos"
OUTPUT_BASE="${EXPERIMENT_ROOT}/stage3_metrics"
SCRATCH_BASE="${EXPERIMENT_ROOT}/stage3_metrics_parallel_scratch"
LOG_DIR="${EXPERIMENT_ROOT}/logs/stage3_metrics_post_gpu23"
PLAN_JSON="${EXPERIMENT_ROOT}/stage3_metrics_post_gpu23_plan.json"

WAN_PYTHON="/data/gaoya/miniconda3/envs/wan/bin/python"
SAM_PYTHON="/data/gaoya/miniconda3/envs/physxnet_mpm_env/bin/python"
FILL_MISSING="${SCRIPT_DIR}/fill_missing_metrics.py"
FAST_SCRIPT="${SCRIPT_DIR}/compute_head_scope_baseline_metrics.py"
TRAJECTORY_SCRIPT="${SCRIPT_DIR}/compute_head_scope_trajectory_metrics.py"
SURVIVAL_SCRIPT="${SCRIPT_DIR}/compute_head_scope_object_survival_metrics.py"

SCOPES=(top100 bottom100 random100_layer_matched_draw0 all720)
GPUS=(2 2 3 3)

mkdir -p "${SCRATCH_BASE}" "${LOG_DIR}"
cd "${REPO_ROOT}"

echo "[wait-generation] oqif_stage3_gpu2 / oqif_stage3_gpu3"
while tmux has-session -t oqif_stage3_gpu2 2>/dev/null \
  || tmux has-session -t oqif_stage3_gpu3 2>/dev/null; do
  sleep 30
done

video_count="$(find "${RESULT_ROOT}" -type f -name '*.mp4' | wc -l)"
echo "[generation-ended] videos=${video_count}"
if (( video_count < 1188 )); then
  echo "[error] generation session ended before the expected 1188 videos" >&2
  exit 1
fi

"${WAN_PYTHON}" "${FILL_MISSING}" "${RESULT_ROOT}" \
  --gpu 2 \
  --output-base "${OUTPUT_BASE}" \
  --stages fast,trajectory,survival \
  --plan-only > "${PLAN_JSON}"

mapfile -t missing_seed_dirs < <(
  "${WAN_PYTHON}" - "${PLAN_JSON}" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
for row in payload["head_scope"]:
    missing = row["missing"]
    if missing["trajectory"] or missing["survival"]:
        print(row["seed_dir"])
PY
)

echo "[plan] incomplete_case_seeds=${#missing_seed_dirs[@]}"

# Fast pixel/ROI metrics are CPU-heavy and already have internal worker support.
"${WAN_PYTHON}" "${FAST_SCRIPT}" "${RESULT_ROOT}" \
  --output-base "${OUTPUT_BASE}/head_scope_baseline_fast" \
  --workers 12 \
  > "${LOG_DIR}/fast.log" 2>&1 &
fast_pid=$!
echo "[fast-start] pid=${fast_pid} workers=12"

wait_workers() {
  local stage="$1"
  shift
  local failed=0
  local pid
  for pid in "$@"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if (( failed != 0 )); then
    echo "[error] one or more ${stage} workers failed" >&2
    exit 1
  fi
}

copy_tree_no_clobber() {
  local source="$1"
  local destination="$2"
  [[ -d "${source}" ]] || return 0
  mkdir -p "${destination}"
  cp -aln "${source}/." "${destination}/"
}

for seed_dir in "${missing_seed_dirs[@]}"; do
  case_name="$(basename -- "$(dirname -- "${seed_dir}")")"
  seed_name="$(basename -- "${seed_dir}")"
  unit_key="${case_name}__${seed_name}"
  canonical_root="${OUTPUT_BASE}/head_scope_trajectory/${case_name}/${seed_name}"
  echo "[unit-start] ${case_name}/${seed_name}"

  trajectory_pids=()
  for index in "${!SCOPES[@]}"; do
    scope="${SCOPES[$index]}"
    gpu="${GPUS[$index]}"
    worker_base="${SCRATCH_BASE}/${unit_key}/${scope}/head_scope_trajectory"
    log_path="${LOG_DIR}/${unit_key}__${scope}__trajectory.log"
    env CUDA_VISIBLE_DEVICES="${gpu}" \
      "${WAN_PYTHON}" "${TRAJECTORY_SCRIPT}" "${seed_dir}" \
      --output-base "${worker_base}" \
      --device cuda:0 \
      --head-scopes "${scope}" \
      > "${log_path}" 2>&1 &
    trajectory_pids+=("$!")
    echo "[trajectory-start] gpu=${gpu} scope=${scope} pid=$!"
  done
  wait_workers trajectory "${trajectory_pids[@]}"

  for scope in "${SCOPES[@]}"; do
    worker_root="${SCRATCH_BASE}/${unit_key}/${scope}/head_scope_trajectory/${case_name}/${seed_name}"
    copy_tree_no_clobber "${worker_root}/tracks" "${canonical_root}/tracks"
    copy_tree_no_clobber "${worker_root}/overlays" "${canonical_root}/overlays"
  done

  # Rebuild one complete trajectory report from the now-complete canonical cache.
  env CUDA_VISIBLE_DEVICES=2 \
    "${WAN_PYTHON}" "${TRAJECTORY_SCRIPT}" "${seed_dir}" \
    --output-base "${OUTPUT_BASE}/head_scope_trajectory" \
    --device cuda:0 \
    > "${LOG_DIR}/${unit_key}__trajectory_consolidate.log" 2>&1
  echo "[trajectory-complete] ${case_name}/${seed_name}"

  survival_pids=()
  for index in "${!SCOPES[@]}"; do
    scope="${SCOPES[$index]}"
    gpu="${GPUS[$index]}"
    worker_base="${SCRATCH_BASE}/${unit_key}/${scope}/head_scope_trajectory"
    log_path="${LOG_DIR}/${unit_key}__${scope}__survival.log"
    env CUDA_VISIBLE_DEVICES="${gpu}" \
      PYTHONPATH="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/Code_data/Code_try0526${PYTHONPATH:+:${PYTHONPATH}}" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "${SAM_PYTHON}" "${SURVIVAL_SCRIPT}" "${seed_dir}" \
      --output-base "${worker_base}" \
      --device cuda:0 \
      --head-scopes "${scope}" \
      > "${log_path}" 2>&1 &
    survival_pids+=("$!")
    echo "[survival-start] gpu=${gpu} scope=${scope} pid=$!"
  done
  wait_workers survival "${survival_pids[@]}"

  for scope in "${SCOPES[@]}"; do
    worker_root="${SCRATCH_BASE}/${unit_key}/${scope}/head_scope_trajectory/${case_name}/${seed_name}"
    copy_tree_no_clobber \
      "${worker_root}/object_survival" \
      "${canonical_root}/object_survival"
  done

  # Rebuild one complete survival report from the canonical track/mask/feature caches.
  env CUDA_VISIBLE_DEVICES=2 \
    PYTHONPATH="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/Code_data/Code_try0526${PYTHONPATH:+:${PYTHONPATH}}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${SAM_PYTHON}" "${SURVIVAL_SCRIPT}" "${seed_dir}" \
    --output-base "${OUTPUT_BASE}/head_scope_trajectory" \
    --device cuda:0 \
    > "${LOG_DIR}/${unit_key}__survival_consolidate.log" 2>&1
  echo "[unit-complete] ${case_name}/${seed_name}"
done

wait_workers fast "${fast_pid}"

# Final idempotent repair/audit pass: normally all three stages are already complete.
bash "${SCRIPT_DIR}/bench_missing.sh" "${RESULT_ROOT}" \
  --gpu 2 \
  --output-base "${OUTPUT_BASE}" \
  --stages fast,trajectory,survival \
  > "${LOG_DIR}/final_audit.log" 2>&1

echo "[all-done] GPU2/GPU3 parallel metric backfill complete"
