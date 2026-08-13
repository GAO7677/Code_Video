#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${ROOT}")"
PYTHON_BIN="${PYTHON_BIN:-/home/gaoya/miniconda3/envs/wan-cu128/bin/python}"
DEVICE="${DEVICE:?Set DEVICE to an available CUDA device; cuda:4 is prohibited}"
STAGE_CONFIG="${STAGE_CONFIG:-${ROOT}/configs/stage1_movic.yaml}"
SLOT_STATS="${SLOT_STATS:-/data/gaoya/agent-data/cache/xssc_stage1_causal_state/train_slot_stats.pt}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/data/gaoya/agent-data/checkpoints/xssc_stage1_causal_state}"
RESULTS_ROOT="${RESULTS_ROOT:-/data/gaoya/agent-data/outputs/xssc_stage1_causal_state/evaluations}"
WANDB_PROJECT="${WANDB_PROJECT:-}"

if [[ "${DEVICE}" == "cuda:4" ]]; then
  echo "ERROR: GPU 4 is prohibited by workspace policy" >&2
  exit 2
fi
if [[ ! -f "${SLOT_STATS}" ]]; then
  echo "ERROR: slot statistics are missing: ${SLOT_STATS}" >&2
  exit 2
fi

representations=(dyn dyn_static full)
histories=(1 2 4)
contexts=(individual set)
seeds=(42 43 44)
mappings=(prefix boundary)

cd "${PROJECT_ROOT}"
for representation in "${representations[@]}"; do
  for mapping in "${mappings[@]}"; do
    probe_dir="${CHECKPOINT_ROOT}/gt_probes/${representation}/${mapping}/seed_42"
    "${PYTHON_BIN}" -m stage1_causal_state_probe.train_gt_probes \
      --stage-config "${STAGE_CONFIG}" \
      --slot-stats "${SLOT_STATS}" \
      --representation "${representation}" \
      --mapping "${mapping}" \
      --seed 42 \
      --device "${DEVICE}" \
      --output-dir "${probe_dir}"
    "${PYTHON_BIN}" -m stage1_causal_state_probe.evaluate_gt_probes \
      --stage-config "${STAGE_CONFIG}" \
      --probe "${probe_dir}/best.pt" \
      --split test \
      --mapping "${mapping}" \
      --device "${DEVICE}" \
      --output-dir "${RESULTS_ROOT}/probe_ceiling/${mapping}/${representation}"
  done

  for history in "${histories[@]}"; do
    for context in "${contexts[@]}"; do
      for seed in "${seeds[@]}"; do
        predictor_dir="${CHECKPOINT_ROOT}/predictors/${representation}/h${history}_${context}/seed_${seed}"
        wandb_args=()
        if [[ -n "${WANDB_PROJECT}" ]]; then
          wandb_args+=(--wandb-project "${WANDB_PROJECT}")
        fi
        "${PYTHON_BIN}" -m stage1_causal_state_probe.train_state_predictor \
          --stage-config "${STAGE_CONFIG}" \
          --stats "${SLOT_STATS}" \
          --representation "${representation}" \
          --history "${history}" \
          --context "${context}" \
          --seed "${seed}" \
          --device "${DEVICE}" \
          --output-dir "${predictor_dir}" \
          "${wandb_args[@]}"

        for mapping in "${mappings[@]}"; do
          probe="${CHECKPOINT_ROOT}/gt_probes/${representation}/${mapping}/seed_42/best.pt"
          result_dir="${RESULTS_ROOT}/${mapping}/${representation}/h${history}_${context}/seed_${seed}"
          "${PYTHON_BIN}" -m stage1_causal_state_probe.evaluate_stage1 \
            --stage-config "${STAGE_CONFIG}" \
            --predictor "${predictor_dir}/best.pt" \
            --probe "${probe}" \
            --split test \
            --mapping "${mapping}" \
            --device "${DEVICE}" \
            --output-dir "${result_dir}"
        done
      done
    done
  done
done
