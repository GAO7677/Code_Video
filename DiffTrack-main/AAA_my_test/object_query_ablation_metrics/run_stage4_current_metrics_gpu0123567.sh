#!/usr/bin/env bash
set -euo pipefail

# Backfill the currently generated Stage-4 videos on every permitted GPU.
# Work is split by case-seed and head scope into isolated caches, then merged
# and consolidated into the canonical Stage-4 trajectory/survival reports.

REPO_ROOT="/home/gaoya/Code_Video/DiffTrack-main"
SCRIPT_DIR="${REPO_ROOT}/AAA_my_test/object_query_ablation_metrics"
EXPERIMENT_ROOT="/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1"
RESULT_ROOT="${EXPERIMENT_ROOT}/stage4_temporal_v1"
OUTPUT_BASE="${EXPERIMENT_ROOT}/stage4_metrics"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
SCRATCH_ROOT="${EXPERIMENT_ROOT}/stage4_metrics_parallel_scratch/${RUN_ID}"
LOG_ROOT="${EXPERIMENT_ROOT}/logs/stage4_metrics_gpu0123567/${RUN_ID}"
TASK_FILE="${SCRATCH_ROOT}/tasks.tsv"
SEED_FILE="${SCRATCH_ROOT}/seeds.tsv"

WAN_PYTHON="/data/gaoya/miniconda3/envs/wan/bin/python"
SAM_PYTHON="/data/gaoya/miniconda3/envs/physxnet_mpm_env/bin/python"
TRAJECTORY_SCRIPT="${SCRIPT_DIR}/compute_head_scope_trajectory_metrics.py"
SURVIVAL_SCRIPT="${SCRIPT_DIR}/compute_head_scope_object_survival_metrics.py"
GPUS=(0 1 2 3 5 6 7)

mkdir -p "${SCRATCH_ROOT}" "${LOG_ROOT}" "${OUTPUT_BASE}/head_scope_trajectory"
cd "${REPO_ROOT}"

"${WAN_PYTHON}" - "${RESULT_ROOT}" "${TASK_FILE}" "${SEED_FILE}" <<'PY'
import sys
from pathlib import Path

from AAA_my_test.object_query_ablation_metrics.compute_head_scope_baseline_metrics import (
    collect_candidates,
    discover_seed_dirs,
)

root, task_path, seed_path = map(Path, sys.argv[1:])
allowed = {"top100", "bottom100", "random100_layer_matched_draw0", "all720"}
tasks = []
seeds = []
for seed_dir in discover_seed_dirs(root):
    rows = collect_candidates(seed_dir, allowed)
    if not rows:
        continue
    case = str(rows[0]["case"])
    seed = int(rows[0]["seed"])
    seed_name = f"seed_{seed:05d}"
    scopes = sorted({str(row["head_scope"]) for row in rows})
    seeds.append((str(seed_dir), case, seed_name))
    for scope in scopes:
        tasks.append((len(tasks), str(seed_dir), case, seed_name, scope))

task_path.write_text(
    "".join("\t".join(map(str, row)) + "\n" for row in tasks), encoding="utf-8"
)
seed_path.write_text(
    "".join("\t".join(map(str, row)) + "\n" for row in seeds), encoding="utf-8"
)
print(f"[snapshot] case_seeds={len(seeds)} scope_tasks={len(tasks)}")
PY

task_count="$(wc -l < "${TASK_FILE}")"
seed_count="$(wc -l < "${SEED_FILE}")"
if (( task_count == 0 )); then
  echo "[done] no generated Stage-4 videos need GPU metrics"
  exit 0
fi
echo "[plan] run_id=${RUN_ID} case_seeds=${seed_count} scope_tasks=${task_count} gpus=${GPUS[*]}"

run_trajectory_worker() {
  local worker_index="$1"
  local gpu="$2"
  local failures=0
  while IFS=$'\t' read -r task_index seed_dir case seed_name scope; do
    (( task_index % ${#GPUS[@]} == worker_index )) || continue
    local unit="${case}__${seed_name}__${scope}"
    local worker_base="${SCRATCH_ROOT}/${unit}/head_scope_trajectory"
    local log_path="${LOG_ROOT}/gpu${gpu}__${unit}__trajectory.log"
    echo "[trajectory-start] gpu=${gpu} task=${task_index}/${task_count} ${case}/${seed_name}/${scope}"
    if env CUDA_VISIBLE_DEVICES="${gpu}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "${WAN_PYTHON}" -u "${TRAJECTORY_SCRIPT}" "${seed_dir}" \
        --output-base "${worker_base}" \
        --device cuda:0 \
        --head-scopes "${scope}" \
        > "${log_path}" 2>&1; then
      echo "[trajectory-done] gpu=${gpu} ${case}/${seed_name}/${scope}"
    else
      echo "[trajectory-failed] gpu=${gpu} ${case}/${seed_name}/${scope}; see ${log_path}" >&2
      failures=$((failures + 1))
    fi
  done < "${TASK_FILE}"
  return "${failures}"
}

pids=()
for worker_index in "${!GPUS[@]}"; do
  run_trajectory_worker "${worker_index}" "${GPUS[$worker_index]}" \
    > "${LOG_ROOT}/gpu${GPUS[$worker_index]}__trajectory_worker.log" 2>&1 &
  pids+=("$!")
done

trajectory_failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    trajectory_failed=1
  fi
done
if (( trajectory_failed != 0 )); then
  echo "[error] at least one trajectory scope task failed; refusing partial consolidation" >&2
  exit 1
fi

copy_tree_no_clobber() {
  local source="$1"
  local destination="$2"
  [[ -d "${source}" ]] || return 0
  mkdir -p "${destination}"
  cp -aln "${source}/." "${destination}/"
}

while IFS=$'\t' read -r _task_index _seed_dir case seed_name scope; do
  unit="${case}__${seed_name}__${scope}"
  worker_root="${SCRATCH_ROOT}/${unit}/head_scope_trajectory/${case}/${seed_name}"
  canonical_root="${OUTPUT_BASE}/head_scope_trajectory/${case}/${seed_name}"
  copy_tree_no_clobber "${worker_root}/tracks" "${canonical_root}/tracks"
  copy_tree_no_clobber "${worker_root}/overlays" "${canonical_root}/overlays"
done < "${TASK_FILE}"

# Rebuild canonical trajectory reports from the merged caches. Any video that
# appeared after the snapshot is safely picked up here on GPU0.
while IFS=$'\t' read -r seed_dir case seed_name; do
  echo "[trajectory-consolidate] gpu=0 ${case}/${seed_name}"
  env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${WAN_PYTHON}" -u "${TRAJECTORY_SCRIPT}" "${seed_dir}" \
      --output-base "${OUTPUT_BASE}/head_scope_trajectory" \
      --device cuda:0 \
      > "${LOG_ROOT}/${case}__${seed_name}__trajectory_consolidate.log" 2>&1
done < "${SEED_FILE}"

run_survival_worker() {
  local worker_index="$1"
  local gpu="$2"
  local failures=0
  while IFS=$'\t' read -r task_index seed_dir case seed_name scope; do
    (( task_index % ${#GPUS[@]} == worker_index )) || continue
    local unit="${case}__${seed_name}__${scope}"
    local worker_base="${SCRATCH_ROOT}/${unit}/head_scope_trajectory"
    local log_path="${LOG_ROOT}/gpu${gpu}__${unit}__survival.log"
    echo "[survival-start] gpu=${gpu} task=${task_index}/${task_count} ${case}/${seed_name}/${scope}"
    if env CUDA_VISIBLE_DEVICES="${gpu}" \
      PYTHONPATH="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/Code_data/Code_try0526${PYTHONPATH:+:${PYTHONPATH}}" \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "${SAM_PYTHON}" -u "${SURVIVAL_SCRIPT}" "${seed_dir}" \
        --output-base "${worker_base}" \
        --device cuda:0 \
        --batch-size 2 \
        --head-scopes "${scope}" \
        > "${log_path}" 2>&1; then
      echo "[survival-done] gpu=${gpu} ${case}/${seed_name}/${scope}"
    else
      echo "[survival-failed] gpu=${gpu} ${case}/${seed_name}/${scope}; see ${log_path}" >&2
      failures=$((failures + 1))
    fi
  done < "${TASK_FILE}"
  return "${failures}"
}

pids=()
for worker_index in "${!GPUS[@]}"; do
  run_survival_worker "${worker_index}" "${GPUS[$worker_index]}" \
    > "${LOG_ROOT}/gpu${GPUS[$worker_index]}__survival_worker.log" 2>&1 &
  pids+=("$!")
done

survival_failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    survival_failed=1
  fi
done
if (( survival_failed != 0 )); then
  echo "[error] at least one survival scope task failed; refusing partial consolidation" >&2
  exit 1
fi

while IFS=$'\t' read -r _task_index _seed_dir case seed_name scope; do
  unit="${case}__${seed_name}__${scope}"
  worker_root="${SCRATCH_ROOT}/${unit}/head_scope_trajectory/${case}/${seed_name}"
  canonical_root="${OUTPUT_BASE}/head_scope_trajectory/${case}/${seed_name}"
  copy_tree_no_clobber "${worker_root}/object_survival" "${canonical_root}/object_survival"
done < "${TASK_FILE}"

# Rebuild canonical survival reports from merged SAM2/DINO caches.
while IFS=$'\t' read -r seed_dir case seed_name; do
  echo "[survival-consolidate] gpu=1 ${case}/${seed_name}"
  env CUDA_VISIBLE_DEVICES=1 \
    PYTHONPATH="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/Code_data/Code_try0526${PYTHONPATH:+:${PYTHONPATH}}" \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${SAM_PYTHON}" -u "${SURVIVAL_SCRIPT}" "${seed_dir}" \
      --output-base "${OUTPUT_BASE}/head_scope_trajectory" \
      --device cuda:0 \
      --batch-size 2 \
      > "${LOG_ROOT}/${case}__${seed_name}__survival_consolidate.log" 2>&1
done < "${SEED_FILE}"

touch "${LOG_ROOT}/COMPLETE"
echo "[all-done] run_id=${RUN_ID} current Stage-4 trajectory/survival metrics complete"
