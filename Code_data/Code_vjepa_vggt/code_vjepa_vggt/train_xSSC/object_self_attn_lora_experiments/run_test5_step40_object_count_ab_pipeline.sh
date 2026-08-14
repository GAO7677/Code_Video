#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
CONFIG="${ROOT}/test5_step40_object_count_ab_config.json"
PREPARE="${ROOT}/prepare_test5_step40_object_count_ab.py"
PUBLISH="${ROOT}/publish_test5_step40_object_count_ab.py"
WATCHER="${ROOT}/xssc_lora_checkpoint_watch.py"
OUTPUT_ROOT=/data/gaoya/agent-data/outputs/test5_step40_object_count_ab
STATE_ROOT="${OUTPUT_ROOT}/watch/state"
EXPECTED_GROUPS=36
EXPECTED_CPU_METRICS=144
EXPECTED_GPU_METRICS=360

mkdir -p "${OUTPUT_ROOT}" "${STATE_ROOT}/checkpoints" "${STATE_ROOT}/metrics"

if [[ ! -s "${CONFIG}" ]]; then
  "${PYTHON}" "${PREPARE}" --output "${CONFIG}"
fi

count_groups() {
  find "${STATE_ROOT}/checkpoints" -type f -name 'step-000500.json' | wc -l
}

count_metrics() {
  local total=0
  local metric
  local found
  for metric in "$@"; do
    found="$(find "${STATE_ROOT}/metrics" -type f -name "${metric}.json" | wc -l)"
    total=$((total + found))
  done
  printf '%s\n' "${total}"
}

publish_stage() {
  "${PYTHON}" "${PUBLISH}" --stage "$1"
}

"${PYTHON}" "${WATCHER}" --config "${CONFIG}" --mode refresh
publish_stage inference

while [[ "$(count_groups)" -lt "${EXPECTED_GROUPS}" ]]; do
  before="$(count_groups)"
  printf '[pipeline] inference groups %s/%s\n' "${before}" "${EXPECTED_GROUPS}"
  "${PYTHON}" "${WATCHER}" --config "${CONFIG}" --mode inference --once --gpus 7
  after="$(count_groups)"
  publish_stage inference
  if [[ "${after}" -le "${before}" ]]; then
    sleep 10
  fi
done

CPU_METRICS=(
  physics_iq_with_context
  physics_iq_without_context
  pmf_with_context
  pmf_without_context
)
publish_stage cpu_metrics
while [[ "$(count_metrics "${CPU_METRICS[@]}")" -lt "${EXPECTED_CPU_METRICS}" ]]; do
  before="$(count_metrics "${CPU_METRICS[@]}")"
  printf '[pipeline] CPU metrics %s/%s\n' "${before}" "${EXPECTED_CPU_METRICS}"
  "${PYTHON}" "${WATCHER}" --config "${CONFIG}" --mode metrics --kind cpu --once
  after="$(count_metrics "${CPU_METRICS[@]}")"
  publish_stage cpu_metrics
  if [[ "${after}" -le "${before}" ]]; then
    sleep 10
  fi
done

GPU_METRICS=(
  wmreward
  vbench_subject_consistency
  vbench_background_consistency
  vbench_temporal_flickering
  vbench_motion_smoothness
  vbench_dynamic_degree
  vbench_aesthetic_quality
  vbench_imaging_quality
  videophy2
  cosmos_reason1
)
publish_stage gpu_metrics
while [[ "$(count_metrics "${GPU_METRICS[@]}")" -lt "${EXPECTED_GPU_METRICS}" ]]; do
  before="$(count_metrics "${GPU_METRICS[@]}")"
  printf '[pipeline] GPU metrics %s/%s\n' "${before}" "${EXPECTED_GPU_METRICS}"
  "${PYTHON}" "${WATCHER}" --config "${CONFIG}" --mode metrics --kind gpu --once --gpus 7
  after="$(count_metrics "${GPU_METRICS[@]}")"
  publish_stage gpu_metrics
  if [[ "${after}" -le "${before}" ]]; then
    sleep 10
  fi
done

"${PYTHON}" "${WATCHER}" --config "${CONFIG}" --mode refresh
publish_stage complete
printf '[pipeline] complete\n'
