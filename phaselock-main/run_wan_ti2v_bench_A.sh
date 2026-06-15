#!/usr/bin/env bash
set -euo pipefail

GPU_ID=2
SEED=42
SAMPLE_STEPS=20
FEW_STEPS=2
FRAME_NUM=121
SIZE="1280*704"
GROUP_NAME="A"
BENCHMARK_NAME="PDI-Bench"
PHASELOCK_METHOD_NAME="phaselock-wan22-5B-TI2V"
BASELINE_METHOD_NAME="wan22-5B-TI2V"

REPO_ROOT="/home/gaoya/Code_Video/phaselock-main"
WAN_ENV_PYTHON="/data/gaoya/miniconda3/envs/wan/bin/python"
WAN_CKPT_DIR="/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B"
PHASELOCK_SCRIPT="${REPO_ROOT}/code/scripts/wan_ti2v_phaselock.py"
BASELINE_SCRIPT="${REPO_ROOT}/Wan2.2-main/generate.py"

BENCH_ROOT="/data/gaoya/AAA_test_video/Output_try0526/ABD_test/A"
MANIFEST_PATH="${BENCH_ROOT}/_meta/source_manifest.json"
OUTPUT_ROOT="/data/gaoya/AAA_test_video/0529/phaselock/test/A"

mkdir -p "${OUTPUT_ROOT}"

if [ ! -f "${MANIFEST_PATH}" ]; then
  echo "Error: manifest not found: ${MANIFEST_PATH}" >&2
  exit 1
fi

if [ $(((FRAME_NUM - 1) % 4)) -ne 0 ]; then
  echo "Error: FRAME_NUM must satisfy 4n+1 for Wan TI2V, got ${FRAME_NUM}." >&2
  echo "Examples of valid values: 49, 81, 121." >&2
  exit 1
fi

echo "Bench root: ${BENCH_ROOT}"
echo "Manifest: ${MANIFEST_PATH}"
echo "Output root: ${OUTPUT_ROOT}"

"${WAN_ENV_PYTHON}" - <<'PY' \
  "${MANIFEST_PATH}" \
  "${OUTPUT_ROOT}" \
  "${WAN_ENV_PYTHON}" \
  "${PHASELOCK_SCRIPT}" \
  "${BASELINE_SCRIPT}" \
  "${WAN_CKPT_DIR}" \
  "${GPU_ID}" \
  "${SEED}" \
  "${SAMPLE_STEPS}" \
  "${FEW_STEPS}" \
  "${FRAME_NUM}" \
  "${SIZE}" \
  "${GROUP_NAME}" \
  "${BENCHMARK_NAME}" \
  "${PHASELOCK_METHOD_NAME}" \
  "${BASELINE_METHOD_NAME}"
import json
import os
import subprocess
import sys
from pathlib import Path

(
    manifest_path,
    output_root,
    python_bin,
    phaselock_script,
    baseline_script,
    ckpt_dir,
    gpu_id,
    seed,
    sample_steps,
    few_steps,
    frame_num,
    size,
    group_name,
    benchmark_name,
    phaselock_method_name,
    baseline_method_name,
) = sys.argv[1:]

with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

output_root = Path(output_root)
output_root.mkdir(parents=True, exist_ok=True)
phaselock_dir = output_root / phaselock_method_name
baseline_dir = output_root / baseline_method_name
phaselock_dir.mkdir(parents=True, exist_ok=True)
baseline_dir.mkdir(parents=True, exist_ok=True)


def write_case_json(
    json_path: Path,
    *,
    method_name: str,
    case_key: str,
    category: str,
    clip_name: str,
    input_prompt: str,
    input_image: str,
    source_video: str,
    output_video: Path,
    seed_value: str,
    sample_steps_value: str,
    frame_num_value: str,
    size_value: str,
    few_steps_value: str | None = None,
):
    payload = {
        "group": group_name,
        "benchmark": benchmark_name,
        "method_name": method_name,
        "case_key": case_key,
        "category": category,
        "clip_name": clip_name,
        "input_prompt": input_prompt,
        "input_image": input_image,
        "source_video": source_video,
        "output_video": str(output_video),
        "original_json": str(json_path),
        "run_config": {
            "seed": int(seed_value),
            "sample_steps": int(sample_steps_value),
            "frame_num": int(frame_num_value),
            "size": size_value,
        },
    }
    if few_steps_value is not None:
        payload["run_config"]["few_steps"] = int(few_steps_value)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

for idx, item in enumerate(manifest):
    category = item["category"]
    caption = item["caption"]
    first_frame = item["first_frame"]
    source_video = item["source_video"]

    case_key = f"{idx:03d}_{category}_{caption}"
    phaselock_output = phaselock_dir / f"{case_key}.mp4"
    baseline_output = baseline_dir / f"{case_key}.mp4"
    phaselock_json = phaselock_dir / f"{case_key}.json"
    baseline_json = baseline_dir / f"{case_key}.json"

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[{idx + 1}/{len(manifest)}] Running PhaseLock for {case_key}")
    phaselock_cmd = [
        python_bin,
        phaselock_script,
        "--ckpt_dir",
        ckpt_dir,
        "--size",
        size,
        "--image",
        first_frame,
        "--prompt",
        caption,
        "--output",
        str(phaselock_output),
        "--few_steps",
        few_steps,
        "--full_steps",
        sample_steps,
        "--frame_num",
        frame_num,
        "--seed",
        seed,
        "--offload_model",
        "--t5_cpu",
        "--convert_model_dtype",
        "--device_id",
        "0",
    ]
    subprocess.run(phaselock_cmd, check=True, env=env)
    write_case_json(
        phaselock_json,
        method_name=phaselock_method_name,
        case_key=case_key,
        category=category,
        clip_name=caption,
        input_prompt=caption,
        input_image=first_frame,
        source_video=source_video,
        output_video=phaselock_output,
        seed_value=seed,
        sample_steps_value=sample_steps,
        frame_num_value=frame_num,
        size_value=size,
        few_steps_value=few_steps,
    )

    print(f"[{idx + 1}/{len(manifest)}] Running baseline for {case_key}")
    baseline_cmd = [
        python_bin,
        baseline_script,
        "--task",
        "ti2v-5B",
        "--size",
        size,
        "--ckpt_dir",
        ckpt_dir,
        "--offload_model",
        "True",
        "--convert_model_dtype",
        "--t5_cpu",
        "--image",
        first_frame,
        "--prompt",
        caption,
        "--sample_steps",
        sample_steps,
        "--frame_num",
        frame_num,
        "--base_seed",
        seed,
        "--save_file",
        str(baseline_output),
    ]
    subprocess.run(baseline_cmd, check=True, env=env)
    write_case_json(
        baseline_json,
        method_name=baseline_method_name,
        case_key=case_key,
        category=category,
        clip_name=caption,
        input_prompt=caption,
        input_image=first_frame,
        source_video=source_video,
        output_video=baseline_output,
        seed_value=seed,
        sample_steps_value=sample_steps,
        frame_num_value=frame_num,
        size_value=size,
    )

print("Finished all cases.")
PY
