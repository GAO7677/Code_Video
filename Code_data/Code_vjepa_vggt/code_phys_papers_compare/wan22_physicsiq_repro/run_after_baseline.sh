#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/gaoya/data/AAA_test_video/0623/test/physicsiq/wan22_i2v_physicsiq_original_repro
CODE=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_phys_papers_compare
WAN_PY=/home/gaoya/data/miniconda3/envs/wan22-physicsiq/bin/python
EVAL_PY=/home/gaoya/data/miniconda3/envs/vjepa2/bin/python
MANIFEST=${ROOT}/inputs/input_manifest.jsonl
BASELINE_RUN=wan22-ti2v5b-op-conditioning-last-frame-run_01_24fps
BASELINE_SUBMISSION=${ROOT}/baseline/submission_5s/${BASELINE_RUN}
BASELINE_EVAL=${ROOT}/official_eval/baseline
BON_RUN=wan22-ti2v5b-wmreward-bon16-op-switch-frame-run_01_24fps
BON_MANIFEST=${ROOT}/bon16/inputs/input_manifest.jsonl
BON_SUBMISSION=${ROOT}/bon16/submission_5s/${BON_RUN}
BON_EVAL=${ROOT}/official_eval/bon16

export PYTHONUNBUFFERED=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

count_mp4() { find "$1" -maxdepth 1 -type f -name '*.mp4' | wc -l; }

wait_for_gpu7() {
  while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 7)" ]]; do
    echo "[pipeline] GPU 7 busy; waiting 60 seconds"
    sleep 60
  done
}

echo "[pipeline] baseline validation"
[[ "$(count_mp4 "${ROOT}/baseline/raw")" -eq 198 ]]

echo "[pipeline] prepare baseline submission"
"${WAN_PY}" "${CODE}/wan22_physicsiq_repro/prepare_submission.py" \
  --manifest "${MANIFEST}" --input-root "${ROOT}/baseline/raw" \
  --output-root "${BASELINE_SUBMISSION}" --fps 24 --duration 5
[[ "$(count_mp4 "${BASELINE_SUBMISSION}")" -eq 198 ]]

echo "[pipeline] evaluate baseline"
mkdir -p "${BASELINE_EVAL}"
cd "${CODE}/physics-IQ-benchmark-main"
"${EVAL_PY}" physiq/run_physics_iq.py \
  --input_folders "${BASELINE_SUBMISSION}" \
  --output_folder "${BASELINE_EVAL}" \
  --descriptions_file "${CODE}/physics-IQ-benchmark-main/descriptions/descriptions_original.csv" \
  --benchmark_base_folder /home/gaoya/data/dataset
BASELINE_METRICS="${BASELINE_EVAL}/physics-IQ-benchmark-verified/results/${BASELINE_RUN}_metrics.json"
"${WAN_PY}" "${CODE}/wan22_physicsiq_repro/update_metrics_table.py" \
  --table-root /home/gaoya/data/AAA_test_video/0623/test/physicsiq \
  --metrics-file "${BASELINE_METRICS}" --method Wan2.2-TI2V-5B --run "${BASELINE_RUN}" \
  --prompt op/original --image-source conditioning-last-frame-png \
  --result-dir "${BASELINE_SUBMISSION}" --candidates 1 --output-frames 120 --fps 24

echo "[pipeline] prepare official switch-frame BoN inputs"
mkdir -p "$(dirname "${BON_MANIFEST}")"
"${WAN_PY}" "${CODE}/wan22_physicsiq_repro/prepare_physicsiq_original_i2v.py" \
  --dataset-root /home/gaoya/data/dataset/physics-iq-benchmark \
  --descriptions-file "${CODE}/physics-IQ-benchmark-main/descriptions/descriptions_original.csv" \
  --output-root "${ROOT}/bon16/inputs" --image-source switch-frame --ids $(seq 1 198)

echo "[pipeline] generate BoN=16 candidates"
wait_for_gpu7
"${WAN_PY}" "${CODE}/wan22_physicsiq_repro/run_wan22_i2v_bon.py" \
  --wan-repo "${CODE}/Wan2.2" \
  --checkpoint-dir /home/gaoya/data/ckpt/Wan-AI-Wan2.2-TI2V-5B \
  --manifest "${BON_MANIFEST}" --output-root "${ROOT}/bon16/candidates" \
  --device 7 --candidates 16 --base-seed 42000000 \
  --frame-num 121 --sample-steps 50 --sample-shift 5 --guide-scale 5

echo "[pipeline] score and select BoN candidates"
CUDA_VISIBLE_DEVICES=7 VJEPA_CHECKPOINT_DIR="${CODE}/WMReward/checkpoints" \
  "${EVAL_PY}" "${CODE}/wan22_physicsiq_repro/score_wmreward_bon.py" \
  --wmreward-repo "${CODE}/WMReward" --manifest "${BON_MANIFEST}" \
  --candidates-root "${ROOT}/bon16/candidates" --rewards-root "${ROOT}/bon16/rewards" \
  --selected-root "${ROOT}/bon16/selected" --candidates 16 --model vitg384

echo "[pipeline] prepare and evaluate BoN submission"
"${WAN_PY}" "${CODE}/wan22_physicsiq_repro/prepare_submission.py" \
  --manifest "${BON_MANIFEST}" --input-root "${ROOT}/bon16/selected" \
  --output-root "${BON_SUBMISSION}" --fps 24 --duration 5
mkdir -p "${BON_EVAL}"
cd "${CODE}/physics-IQ-benchmark-main"
"${EVAL_PY}" physiq/run_physics_iq.py \
  --input_folders "${BON_SUBMISSION}" --output_folder "${BON_EVAL}" \
  --descriptions_file "${CODE}/physics-IQ-benchmark-main/descriptions/descriptions_original.csv" \
  --benchmark_base_folder /home/gaoya/data/dataset
BON_METRICS="${BON_EVAL}/physics-IQ-benchmark-verified/results/${BON_RUN}_metrics.json"
"${WAN_PY}" "${CODE}/wan22_physicsiq_repro/update_metrics_table.py" \
  --table-root /home/gaoya/data/AAA_test_video/0623/test/physicsiq \
  --metrics-file "${BON_METRICS}" --method 'Wan2.2-TI2V-5B + WMReward' --run "${BON_RUN}" \
  --prompt op/original --image-source official-switch-frame-jpg \
  --result-dir "${BON_SUBMISSION}" --candidates 16 --output-frames 120 --fps 24

echo "[pipeline] all stages completed"
