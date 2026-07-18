#!/usr/bin/env bash
set -u

repo_root="/home/gaoya/Code_Video/DiffTrack-main"
dataset_root="/data/gaoya/AAA_test_video/Dataset_physV/0718ToyDataset"
track_dir="/data/gaoya/agent-data/outputs/difftrack_0718toy/tracks"
output_dir="/data/gaoya/agent-data/outputs/difftrack_0718toy/cogvideox_2b"
model_path="/data/gaoya/agent-data/weights/CogVideoX-2b-modelscope"
python_bin="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"

mkdir -p "${output_dir}/logs"
export HF_HOME="/data/gaoya/agent-data/cache/huggingface"
export PYTHONPATH="${repo_root}/diffusers/src:${repo_root}"

pids=()
for sample_index in 0 1 2 3; do
  gpu_index="${sample_index}"
  log_path="${output_dir}/logs/sample_${sample_index}.log"
  (
    cd "${repo_root}" || exit 1
    exec "${python_bin}" AAA_my_test/analyze_real_toy.py \
      --dataset-root "${dataset_root}" \
      --track-dir "${track_dir}" \
      --output-dir "${output_dir}" \
      --model-path "${model_path}" \
      --model cogvideox_t2v_2b \
      --device "cuda:${gpu_index}" \
      --start "${sample_index}" \
      --end "$((sample_index + 1))" \
      --matching-accuracy \
      --conf-attn-score
  ) >"${log_path}" 2>&1 &
  pids+=("$!")
  echo "Started sample ${sample_index} on GPU ${gpu_index}, PID ${pids[-1]}, log ${log_path}"
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "Sample ${index} completed"
  else
    echo "Sample ${index} failed; inspect ${output_dir}/logs/sample_${index}.log" >&2
    status=1
  fi
done

exit "${status}"
