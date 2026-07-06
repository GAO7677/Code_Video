#!/usr/bin/env bash
set -euo pipefail
# CUDA_VISIBLE_DEVICES=7 /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/run_formal_compare_morpheus_physiciq_wan_base_two_loras_ti2v_t2v_gpu023.sh


# Formal batch run for:
# - morpheus_real_world
# - physicIQ
#
# Modes:
# - ti2v
# - t2v
#
# Methods:
# - wan_base
# - openvid_lora_step10000
# - openvid_0613pybullet_lora_step000500
#
# Output layout:
# - /data/gaoya/AAA_test_video/0623/test/ti2v/train_stage1b_diffsynth_native0705_ctx1dupjepa_0705_{dataset}/{method}
# - /data/gaoya/AAA_test_video/0623/test/t2v/train_stage1b_diffsynth_native0705_ctx1dupjepa_0705_{dataset}/{method}
#
# Execution policy:
# - Use only GPU 0 / 1 / 2
# - Before each launch, detect which candidate GPU is genuinely idle and use that one
# - For each (dataset, mode), run up to 3 methods in parallel across the idle GPUs
# - After each (dataset, mode) finishes, run bench.sh on that mode root
# - Keep per-case logs beside outputs
# - Keep orchestration logs under /data/gaoya/agent-data
#
# Full run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/run_formal_compare_morpheus_physiciq_wan_base_two_loras_ti2v_t2v_gpu023.sh
#
# Optional filters:
# TARGET_DATASETS="physicIQ" TARGET_MODES="ti2v" OVERWRITE=1 \
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/run_formal_compare_morpheus_physiciq_wan_base_two_loras_ti2v_t2v_gpu023.sh

PY=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
REPO=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
TRAIN0419=/home/gaoya/Code_Video/Code_data/Code_train/train_0419
ENTRY="${REPO}/code_vjepa_vggt/AAAinfer/wan_base_two_loras_ti2v_t2v.py"
BENCH_SCRIPT="${REPO}/code_vjepa_vggt/AAAinfer/bench.sh"

LIST_MORPHEUS=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_morpheus_real_world.txt
LIST_PHYSICIQ=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt

RESULT_BASE_TI2V=/data/gaoya/AAA_test_video/0623/test/ti2v
RESULT_BASE_T2V=/data/gaoya/AAA_test_video/0623/test/t2v
LOG_BASE=/data/gaoya/agent-data/outputs/train0705_wan_base_two_loras_formal_logs_20260706
mkdir -p "${RESULT_BASE_TI2V}" "${RESULT_BASE_T2V}" "${LOG_BASE}"

TARGET_DATASETS="${TARGET_DATASETS:-morpheus_real_world physicIQ}"
TARGET_MODES="${TARGET_MODES:-ti2v t2v}"
OVERWRITE="${OVERWRITE:-0}"
GPU_POOL=(0 1 2)
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"
GPU_MAX_MEMORY_MB="${GPU_MAX_MEMORY_MB:-1024}"
GPU_MAX_UTILIZATION="${GPU_MAX_UTILIZATION:-10}"

FAILED_JOBS=()
declare -A GPU_PID=()
declare -A GPU_LABEL=()

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found" >&2
  exit 127
fi

dataset_list_path() {
  local dataset_tag="$1"
  case "${dataset_tag}" in
    morpheus_real_world) echo "${LIST_MORPHEUS}" ;;
    physicIQ) echo "${LIST_PHYSICIQ}" ;;
    *)
      echo "unknown dataset_tag: ${dataset_tag}" >&2
      return 1
      ;;
  esac
}

mode_result_root() {
  local mode="$1"
  local dataset_tag="$2"
  case "${mode}" in
    ti2v) echo "${RESULT_BASE_TI2V}/train_stage1b_diffsynth_native0705_ctx1dupjepa_0705_${dataset_tag}" ;;
    t2v) echo "${RESULT_BASE_T2V}/train_stage1b_diffsynth_native0705_ctx1dupjepa_0705_${dataset_tag}" ;;
    *)
      echo "unknown mode: ${mode}" >&2
      return 1
      ;;
  esac
}

gpu_is_available() {
  local gpu="$1"
  local row
  row="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits -i "${gpu}" 2>/dev/null | head -n 1 || true)"
  if [[ -z "${row}" ]]; then
    return 1
  fi

  local index
  local memory_used
  local utilization
  IFS=',' read -r index memory_used utilization <<< "${row}"
  memory_used="$(echo "${memory_used}" | xargs)"
  utilization="$(echo "${utilization}" | xargs)"

  local app_count
  app_count="$(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i "${gpu}" 2>/dev/null \
      | sed '/^[[:space:]]*$/d' \
      | wc -l
  )"

  if (( app_count > 0 )); then
    return 1
  fi
  if (( memory_used > GPU_MAX_MEMORY_MB )); then
    return 1
  fi
  if (( utilization > GPU_MAX_UTILIZATION )); then
    return 1
  fi
  return 0
}

describe_gpu() {
  local gpu="$1"
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits -i "${gpu}" 2>/dev/null \
    | head -n 1 \
    | sed 's/^[[:space:]]*//'
}

check_freed_gpus() {
  local gpu
  for gpu in "${GPU_POOL[@]}"; do
    local pid="${GPU_PID[$gpu]:-}"
    if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
      unset 'GPU_PID[$gpu]'
      unset 'GPU_LABEL[$gpu]'
    fi
  done
}

wait_for_free_gpu() {
  while :; do
    check_freed_gpus

    local gpu
    for gpu in "${GPU_POOL[@]}"; do
      if [[ -n "${GPU_PID[$gpu]:-}" ]]; then
        continue
      fi
      if gpu_is_available "${gpu}"; then
        echo "${gpu}"
        return 0
      fi
    done

    echo "[gpu_wait] no idle gpu in pool ${GPU_POOL[*]}; poll=${GPU_POLL_SECONDS}s thresholds memory<=${GPU_MAX_MEMORY_MB}MB util<=${GPU_MAX_UTILIZATION}%" >&2
    for gpu in "${GPU_POOL[@]}"; do
      local owner="${GPU_LABEL[$gpu]:-(external_or_busy)}"
      echo "[gpu_wait] gpu=${gpu} owner=${owner} status=$(describe_gpu "${gpu}")" >&2
    done
    sleep "${GPU_POLL_SECONDS}"
  done
}

launch_method_job() {
  local mode="$1"
  local dataset_tag="$2"
  local list_path="$3"
  local method_name="$4"
  local dataset_root="$5"

  local method_root="${dataset_root}/${method_name}"
  local job_label="${dataset_tag}:${mode}:${method_name}"
  local job_log="${LOG_BASE}/${dataset_tag}_${mode}_${method_name}.log"
  local gpu
  mkdir -p "${method_root}"
  gpu="$(wait_for_free_gpu)"

  echo "[launch] label=${job_label} gpu=${gpu} method_root=${method_root}" >&2

  (
    export PYTHONPATH="${REPO}:${DIFFSYNTH}:${TRAIN0419}"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export FORMAL_COMPARE_PY="${PY}"
    export FORMAL_COMPARE_ENTRY="${ENTRY}"
    export FORMAL_COMPARE_MODE="${mode}"
    export FORMAL_COMPARE_METHOD="${method_name}"
    export FORMAL_COMPARE_LIST="${list_path}"
    export FORMAL_COMPARE_METHOD_ROOT="${method_root}"
    export FORMAL_COMPARE_OVERWRITE="${OVERWRITE}"

    "${PY}" -u - <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path


PYTHON = os.environ["FORMAL_COMPARE_PY"]
ENTRY = Path(os.environ["FORMAL_COMPARE_ENTRY"])
MODE = os.environ["FORMAL_COMPARE_MODE"]
METHOD = os.environ["FORMAL_COMPARE_METHOD"]
LIST_PATH = Path(os.environ["FORMAL_COMPARE_LIST"])
METHOD_ROOT = Path(os.environ["FORMAL_COMPARE_METHOD_ROOT"])
OVERWRITE = os.environ.get("FORMAL_COMPARE_OVERWRITE", "0") == "1"


def first_existing_path(payload: dict, keys: list[str]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def build_command(payload: dict, output_video_path: Path) -> list[str]:
    prompt = str(payload.get("input_caption", "")).strip()
    if not prompt:
        raise ValueError(f"missing input_caption in {payload}")

    cmd = [
        PYTHON,
        str(ENTRY),
        "--mode",
        MODE,
        "--model-preset",
        METHOD,
        "--prompt",
        prompt,
        "--output-video-path",
        str(output_video_path),
        "--num-inference-steps",
        "40",
        "--cfg-scale",
        "5.0",
        "--seed",
        "42",
        "--fps",
        "30",
        "--num-frames",
        "24",
    ]

    if MODE == "ti2v":
        context_path = first_existing_path(
            payload,
            [
                "input_video",
                "input_video_24f",
                "input_video_16f",
                "input_video_4f",
                "input_video_randomf",
                "source_video",
            ],
        )
        if context_path is None:
            raise ValueError(f"missing context video path for ti2v: {output_video_path.stem}")
        cmd.extend(["--context-path", context_path, "--conditioning-mode", "input_image_only"])
        first_frame_path = first_existing_path(payload, ["input_image"])
        if first_frame_path is not None:
            cmd.extend(["--first-frame-path", first_frame_path])
    else:
        cmd.extend(["--conditioning-mode", "context_aware"])

    if OVERWRITE:
        cmd.append("--overwrite")

    return cmd


def iter_json_paths(list_path: Path):
    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        yield Path(line)


def main() -> int:
    METHOD_ROOT.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    json_paths = list(iter_json_paths(LIST_PATH))
    print(f"[method:start] mode={MODE} method={METHOD} cases={len(json_paths)} root={METHOD_ROOT}", flush=True)

    for index, json_path in enumerate(json_paths, start=1):
        case_name = json_path.stem
        output_video_path = METHOD_ROOT / f"{case_name}.mp4"
        output_log_path = METHOD_ROOT / f"{case_name}.log"

        if output_video_path.exists() and not OVERWRITE:
            print(f"[case:skip] {index}/{len(json_paths)} case={case_name} output_exists=1", flush=True)
            continue

        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            cmd = build_command(payload, output_video_path)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{case_name}:prepare:{exc}")
            print(f"[case:prepare_failed] {index}/{len(json_paths)} case={case_name} error={exc}", flush=True)
            continue

        print(f"[case:start] {index}/{len(json_paths)} case={case_name}", flush=True)
        with output_log_path.open("w", encoding="utf-8") as handle:
            handle.write(" ".join(subprocess.list2cmdline([part]) for part in cmd) + "\n\n")
            handle.flush()
            result = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, check=False)

        if result.returncode != 0:
            failures.append(f"{case_name}:run:{result.returncode}")
            print(
                f"[case:failed] {index}/{len(json_paths)} case={case_name} rc={result.returncode} log={output_log_path}",
                flush=True,
            )
            continue

        print(f"[case:done] {index}/{len(json_paths)} case={case_name} video={output_video_path}", flush=True)

    if failures:
        print(f"[method:failed] mode={MODE} method={METHOD} failures={len(failures)}", flush=True)
        for item in failures:
            print(f"[failure] {item}", flush=True)
        return 1

    print(f"[method:done] mode={MODE} method={METHOD} cases={len(json_paths)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
  ) >"${job_log}" 2>&1 &

  local pid=$!
  GPU_PID["${gpu}"]="${pid}"
  GPU_LABEL["${gpu}"]="${job_label}"
  echo "${pid}"
}

wait_method_jobs() {
  local dataset_tag="$1"
  local mode="$2"
  shift 2
  local pids=("$@")
  local pid

  for pid in "${pids[@]}"; do
    if wait "${pid}"; then
      echo "[job:done] dataset=${dataset_tag} mode=${mode} pid=${pid}"
    else
      local rc=$?
      echo "[job:failed] dataset=${dataset_tag} mode=${mode} pid=${pid} rc=${rc}" >&2
      FAILED_JOBS+=("${dataset_tag}:${mode}:pid${pid}:rc${rc}")
    fi

    local gpu
    for gpu in "${GPU_POOL[@]}"; do
      if [[ "${GPU_PID[$gpu]:-}" == "${pid}" ]]; then
        unset 'GPU_PID[$gpu]'
        unset 'GPU_LABEL[$gpu]'
      fi
    done
  done
}

run_dataset_mode() {
  local dataset_tag="$1"
  local mode="$2"
  local list_path
  list_path="$(dataset_list_path "${dataset_tag}")"
  local dataset_root
  dataset_root="$(mode_result_root "${mode}" "${dataset_tag}")"
  mkdir -p "${dataset_root}"

  echo "============================================================"
  echo "[dataset_mode:start] dataset=${dataset_tag} mode=${mode}"
  echo "[dataset_mode:list]  ${list_path}"
  echo "[dataset_mode:root]  ${dataset_root}"
  echo "============================================================"

  local pid_wan_base
  local pid_openvid
  local pid_pybullet
  pid_wan_base="$(launch_method_job "${mode}" "${dataset_tag}" "${list_path}" "wan_base" "${dataset_root}")"
  pid_openvid="$(launch_method_job "${mode}" "${dataset_tag}" "${list_path}" "openvid_lora_step10000" "${dataset_root}")"
  pid_pybullet="$(launch_method_job "${mode}" "${dataset_tag}" "${list_path}" "openvid_0613pybullet_lora_step000500" "${dataset_root}")"

  wait_method_jobs "${dataset_tag}" "${mode}" "${pid_wan_base}" "${pid_openvid}" "${pid_pybullet}"

  if ((${#FAILED_JOBS[@]} > 0)); then
    echo "[dataset_mode:warning] some jobs failed before bench: ${FAILED_JOBS[*]}" >&2
  fi

  local bench_gpu
  bench_gpu="$(wait_for_free_gpu)"
  echo "[bench:start] dataset=${dataset_tag} mode=${mode} gpu=${bench_gpu}"
  CUDA_VISIBLE_DEVICES="${bench_gpu}" BENCH_CUDA_VISIBLE_DEVICES="${bench_gpu}" bash "${BENCH_SCRIPT}" "${dataset_root}"
  echo "[bench:done] dataset=${dataset_tag} mode=${mode} gpu=${bench_gpu}"
}

for dataset_tag in ${TARGET_DATASETS}; do
  for mode in ${TARGET_MODES}; do
    run_dataset_mode "${dataset_tag}" "${mode}"
  done
done

if ((${#FAILED_JOBS[@]} > 0)); then
  echo "[all_done_with_errors] ${FAILED_JOBS[*]}" >&2
  exit 1
fi

echo "[all_done] wan_base + two loras formal comparison finished successfully"
