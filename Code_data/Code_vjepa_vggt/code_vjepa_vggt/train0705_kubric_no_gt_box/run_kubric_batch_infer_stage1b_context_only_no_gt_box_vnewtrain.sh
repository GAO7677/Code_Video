#!/usr/bin/env bash
# =============================================================================
# Unified launcher for Kubric batch inference.
# Preferred business-facing inputs:
# - GPU pair: GPU_PAIR=6,7
# - multiple GPU pairs: GPU_PAIRS="6,7 7,6 3,5 5,3"
# - test json txt: TEST_JSON_TXT=/data/.../test_5.txt
# - weights: WEIGHTS_ROOT=/data/.../step-001000
# - method name: METHOD_NAME=train_stage1b_kubric0708_step1000
# - output root: OUTPUT_ROOT=/data/.../train0705_kubric_test5_compare_0708
# - output frames: OUTPUT_FRAMES=49
# - ctx: CTX=8
# - multiple ctx counts: CTX=1,4,8,12,16,20
# - auto split input txt across GPU workers: AUTO_SPLIT_INPUT=1
# - disable object branch ablation: DISABLE_OBJECT_BRANCH=1
#
# Direct one-run example:
# GPU_PAIR=6,7 \
# TEST_JSON_TXT=/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
# WEIGHTS_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-001000 \
# METHOD_NAME=train_stage1b_kubric0708_step1000 \
# OUTPUT_ROOT=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0708 \
# OUTPUT_FRAMES=49 \
# CTX=8 \
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh
#
# Direct auto-split example (3 worker pairs -> 3 shard txt + 3 processes):
# GPU_PAIR="3,3 5,6 7,7" \
# AUTO_SPLIT_INPUT=1 \
# TEST_JSON_TXT=/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
# WEIGHTS_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-001000 \
# METHOD_NAME=train_stage1b_kubric0708_step1000 \
# OUTPUT_ROOT=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0708_split \
# OUTPUT_FRAMES=49 \
# CTX=8 \
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh
#
# Direct one-run no-object-branch ablation:
# GPU_PAIR=0,0 \
# TEST_JSON_TXT=/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
# WEIGHTS_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-002500 \
# METHOD_NAME=train_stage1b_diffsynth_native0705_step2500_no_object_branch \
# OUTPUT_ROOT=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0705_no_object_branch \
# OUTPUT_FRAMES=49 \
# CTX=8 \
# DISABLE_OBJECT_BRANCH=1 \
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh
#
# Sweep on one GPU pair:
# GPU_PAIR=6,7 \
# TEST_JSON_TXT=/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
# WEIGHTS_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-001000 \
# METHOD_NAME=train_stage1b_kubric0708_step1000 \
# OUTPUT_ROOT=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0708_ctxn \
# OUTPUT_FRAMES=49 \
# CTX=1,4,8,12,16,20 \
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh
#
# Sweep on multiple GPU pairs:
# GPU_PAIRS="6,7 7,6 3,5 5,3" \
# TEST_JSON_TXT=/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
# WEIGHTS_ROOT=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-001000 \
# METHOD_NAME=train_stage1b_kubric0708_step1000 \
# OUTPUT_ROOT=/data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0708_ctxn \
# OUTPUT_FRAMES=49 \
# CTX=1,4,8,12,16,20 \
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_kubric_batch_infer_stage1b_context_only_no_gt_box_vnewtrain.sh
# =============================================================================
set -euo pipefail

PROJ=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
DIFFSYNTH_ROOT=/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main
PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
INFER_SCRIPT_OBJECT="${PROJ}/code_vjepa_vggt/train0705_kubric_no_gt_box/wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v.py"
INFER_SCRIPT_NO_OBJECT="${PROJ}/code_vjepa_vggt/train0705_kubric_no_gt_box/wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_no_object_branch_v2v.py"
DEFAULT_NEGATIVE_PROMPT="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

RUN_MODE="${RUN_MODE:-}"
GPU_PAIR="${GPU_PAIR:-}"
GPU_PAIRS="${GPU_PAIRS:-}"
TEST_JSON_TXT="${TEST_JSON_TXT:-}"
METHOD_NAME="${METHOD_NAME:-}"
OUTPUT_FRAMES="${OUTPUT_FRAMES:-}"
CTX="${CTX:-}"
CTX_NUM="${CTX_NUM:-}"
CTX_NUMS="${CTX_NUMS:-}"
USER_VISIBLE_GPU_IDS="${VISIBLE_GPU_IDS:-}"
USER_INFERENCE_GPU_PAIRS="${INFERENCE_GPU_PAIRS:-}"
USER_CONTEXT_FRAMES="${CONTEXT_FRAMES:-}"
USER_CONTEXT_FRAME_VALUES="${CONTEXT_FRAME_VALUES:-}"

VISIBLE_GPU_IDS="${VISIBLE_GPU_IDS:-${GPU_PAIR:-5,6}}"
INFERENCE_GPU_PAIRS="${INFERENCE_GPU_PAIRS:-${GPU_PAIRS:-}}"
INFERENCE_DEVICES="${INFERENCE_DEVICES:-cuda:0,cuda:1}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708/checkpoints/step-001000}"
INPUT_JSON_LIST_PATH="${INPUT_JSON_LIST_PATH:-${TEST_JSON_TXT:-/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt}}"
MODEL_NAME="${MODEL_NAME:-${METHOD_NAME:-train_stage1b_kubric0708_step1000}}"
MODEL_NAME_PREFIX="${MODEL_NAME_PREFIX:-${MODEL_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/gaoya/AAA_test_video/0623/test/v2v/train0705_kubric_test5_compare_0708}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-40}"
OUTPUT_NUM_FRAMES="${OUTPUT_NUM_FRAMES:-${OUTPUT_FRAMES:-49}}"
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-896}"
INPUT_COVER_CROP_WIDTH="${INPUT_COVER_CROP_WIDTH:-896}"
INPUT_COVER_CROP_HEIGHT="${INPUT_COVER_CROP_HEIGHT:-512}"
CONTEXT_FRAMES="${CONTEXT_FRAMES:-${CTX_NUM:-${CTX:-8}}}"
CONTEXT_FRAME_VALUES="${CONTEXT_FRAME_VALUES:-${CTX_NUMS:-${CTX:-1,2,3,4,6,8,9,12,16,20}}}"
SAMPLING_MODE="${SAMPLING_MODE:-prefix}"
CFG_SCALE="${CFG_SCALE:-5.0}"
SEED="${SEED:-42}"
FPS="${FPS:-30}"
NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-${DEFAULT_NEGATIVE_PROMPT}}"
LIMIT="${LIMIT:-}"
FORCE="${FORCE:-0}"
OVERWRITE="${OVERWRITE:-0}"
DISABLE_OBJECT_BRANCH="${DISABLE_OBJECT_BRANCH:-0}"
OBJECT_BRANCH_RATIO_GUARD_MAX_RATIO="${OBJECT_BRANCH_RATIO_GUARD_MAX_RATIO:-}"
OBJECT_BRANCH_RATIO_GUARD_MAX_BLOCK_ID="${OBJECT_BRANCH_RATIO_GUARD_MAX_BLOCK_ID:-}"
OBJECT_ADAPTER_MLP_RESIDUAL_MAX_RATIO="${OBJECT_ADAPTER_MLP_RESIDUAL_MAX_RATIO:-}"
OBJECT_BRANCH_AUTO_FALLBACK_MAX_ACTIVE_SLOTS="${OBJECT_BRANCH_AUTO_FALLBACK_MAX_ACTIVE_SLOTS:-}"
OBJECT_BRANCH_AUTO_FALLBACK_TRIGGER_COUNT="${OBJECT_BRANCH_AUTO_FALLBACK_TRIGGER_COUNT:-}"
AUTO_SPLIT_INPUT="${AUTO_SPLIT_INPUT:-0}"
SPLIT_WORK_ROOT="${SPLIT_WORK_ROOT:-/data/gaoya/agent-data/cache/kubric_batch_infer_splits}"
SHARD_RUN_ID="${SHARD_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$$}"

if [ "${DISABLE_OBJECT_BRANCH}" = "1" ]; then
  INFER_SCRIPT="${INFER_SCRIPT_NO_OBJECT}"
else
  INFER_SCRIPT="${INFER_SCRIPT_OBJECT}"
fi

infer_run_mode() {
  if [ -n "${RUN_MODE}" ]; then
    echo "${RUN_MODE}"
    return
  fi
  if [ -n "${GPU_PAIRS}" ] || [ -n "${USER_INFERENCE_GPU_PAIRS}" ]; then
    echo "sweep"
    return
  fi
  if [ -n "${CTX}" ]; then
    if [[ "${CTX}" == *","* ]]; then
      echo "sweep"
    else
      echo "direct"
    fi
    return
  fi
  if [ -n "${CTX_NUMS}" ] || [ -n "${USER_CONTEXT_FRAME_VALUES}" ]; then
    echo "sweep"
    return
  fi
  if [ -n "${CTX_NUM}" ] || [ -n "${USER_CONTEXT_FRAMES}" ]; then
    echo "direct"
    return
  fi
  if [ -n "${GPU_PAIR}" ] || [ -n "${USER_VISIBLE_GPU_IDS}" ]; then
    echo "direct"
    return
  fi
  echo "direct"
}

check_gpu_pair_has_faulty_gpu4() {
  local raw_pair="$1"
  local clean_pair
  local gpu_id
  clean_pair="$(echo "${raw_pair}" | tr -d '[:space:]')"
  IFS=',' read -r -a pair_gpu_ids <<< "${clean_pair}"
  for gpu_id in "${pair_gpu_ids[@]}"; do
    if [ "${gpu_id}" = "4" ]; then
      echo "ERROR: gpu4 故障, 禁止使用。当前 GPU pair=${raw_pair}" >&2
      exit 1
    fi
  done
}

prepare_launch_layout() {
  local raw_pair="$1"
  local requested_inference_devices="$2"
  local -n out_visible_gpu_ids="$3"
  local -n out_inference_devices="$4"
  local clean_pair
  local gpu_id
  local -a pair_gpu_ids=()
  local -a unique_gpu_ids=()

  clean_pair="$(echo "${raw_pair}" | tr -d '[:space:]')"
  IFS=',' read -r -a pair_gpu_ids <<< "${clean_pair}"
  for gpu_id in "${pair_gpu_ids[@]}"; do
    if [ -z "${gpu_id}" ]; then
      continue
    fi
    if [ "${#unique_gpu_ids[@]}" -eq 0 ] || [ "${unique_gpu_ids[-1]}" != "${gpu_id}" ]; then
      unique_gpu_ids+=("${gpu_id}")
    fi
  done

  if [ "${#unique_gpu_ids[@]}" -eq 0 ]; then
    echo "ERROR: GPU pair 解析后为空: ${raw_pair}" >&2
    exit 1
  fi

  out_visible_gpu_ids="$(IFS=,; echo "${unique_gpu_ids[*]}")"
  if [ "${#unique_gpu_ids[@]}" -lt 2 ]; then
    out_inference_devices="none"
  else
    out_inference_devices="${requested_inference_devices}"
  fi
}

normalize_ctx_values() {
  local raw_list="$1"
  local -n out_array="$2"
  local raw_ctx
  local ctx
  out_array=()
  IFS=',' read -r -a raw_ctx_values <<< "${raw_list}"
  if [ "${#raw_ctx_values[@]}" -eq 0 ]; then
    echo "ERROR: CONTEXT_FRAME_VALUES 不能为空" >&2
    exit 1
  fi
  for raw_ctx in "${raw_ctx_values[@]}"; do
    ctx="$(echo "${raw_ctx}" | xargs)"
    if [ -z "${ctx}" ]; then
      continue
    fi
    if ! [[ "${ctx}" =~ ^[0-9]+$ ]]; then
      echo "ERROR: 非法 context 长度: ${ctx}" >&2
      exit 1
    fi
    if [ "${ctx}" -le 0 ]; then
      echo "ERROR: context 长度必须为正整数，收到 ${ctx}" >&2
      exit 1
    fi
    out_array+=("${ctx}")
  done
  if [ "${#out_array[@]}" -eq 0 ]; then
    echo "ERROR: CONTEXT_FRAME_VALUES 解析后为空" >&2
    exit 1
  fi
}

parse_unique_gpu_ids() {
  local raw_value="$1"
  local -n out_array="$2"
  local clean_value
  local gpu_id
  local existing_gpu_id
  out_array=()
  clean_value="$(echo "${raw_value}" | tr -d '[:space:]')"
  IFS=',' read -r -a raw_gpu_ids <<< "${clean_value}"
  for gpu_id in "${raw_gpu_ids[@]}"; do
    if [ -z "${gpu_id}" ]; then
      continue
    fi
    for existing_gpu_id in "${out_array[@]}"; do
      if [ "${existing_gpu_id}" = "${gpu_id}" ]; then
        gpu_id=""
        break
      fi
    done
    if [ -n "${gpu_id}" ]; then
      out_array+=("${gpu_id}")
    fi
  done
}

read_list_entries() {
  local list_path="$1"
  local -n out_array="$2"
  local raw_line
  local line
  out_array=()
  while IFS= read -r raw_line || [ -n "${raw_line}" ]; do
    line="$(echo "${raw_line}" | xargs)"
    if [ -z "${line}" ]; then
      continue
    fi
    if [[ "${line}" == \#* ]]; then
      continue
    fi
    out_array+=("${line}")
  done < "${list_path}"
}

apply_limit_to_entries() {
  local -n entries_ref="$1"
  if [ -z "${LIMIT}" ]; then
    return
  fi
  if ! [[ "${LIMIT}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: LIMIT 必须是非负整数，收到 ${LIMIT}" >&2
    exit 1
  fi
  entries_ref=("${entries_ref[@]:0:${LIMIT}}")
}

sanitize_path_tag() {
  local raw_value="$1"
  local sanitized
  sanitized="$(echo "${raw_value}" | sed 's/[^0-9A-Za-z_.-]/_/g')"
  sanitized="${sanitized#_}"
  sanitized="${sanitized%_}"
  if [ -z "${sanitized}" ]; then
    sanitized="run"
  fi
  echo "${sanitized}"
}

write_split_txt_files() {
  local input_list_path="$1"
  local requested_shards="$2"
  local split_dir="$3"
  local -n out_split_files="$4"
  local -n out_split_tags="$5"
  local -a entries=()
  local -a local_split_files=()
  local -a local_split_tags=()
  local entry_count
  local shard_count
  local shard_index
  local entry_index
  local shard_file
  local shard_tag

  read_list_entries "${input_list_path}" entries
  apply_limit_to_entries entries
  entry_count="${#entries[@]}"
  if [ "${entry_count}" -le 0 ]; then
    echo "ERROR: 输入列表为空: ${input_list_path}" >&2
    exit 1
  fi

  shard_count="${requested_shards}"
  if [ "${shard_count}" -gt "${entry_count}" ]; then
    shard_count="${entry_count}"
  fi
  if [ "${shard_count}" -le 0 ]; then
    echo "ERROR: shard_count 非法: ${shard_count}" >&2
    exit 1
  fi

  mkdir -p "${split_dir}"
  for ((shard_index = 0; shard_index < shard_count; shard_index++)); do
    printf -v shard_tag "shard%02dof%02d" "$((shard_index + 1))" "${shard_count}"
    shard_file="${split_dir}/${shard_tag}.txt"
    : > "${shard_file}"
    local_split_files+=("${shard_file}")
    local_split_tags+=("${shard_tag}")
  done

  for ((entry_index = 0; entry_index < entry_count; entry_index++)); do
    shard_index=$((entry_index % shard_count))
    printf '%s\n' "${entries[entry_index]}" >> "${local_split_files[shard_index]}"
  done

  out_split_files=("${local_split_files[@]}")
  out_split_tags=("${local_split_tags[@]}")
}

build_direct_split_worker_pairs() {
  local raw_value="$1"
  local -n out_worker_pairs="$2"
  local pair_token
  out_worker_pairs=()
  read -r -a raw_pair_tokens <<< "${raw_value}"
  for pair_token in "${raw_pair_tokens[@]}"; do
    if [ -z "${pair_token}" ]; then
      continue
    fi
    check_gpu_pair_has_faulty_gpu4 "${pair_token}"
    out_worker_pairs+=("${pair_token}")
  done
  if [ "${#out_worker_pairs[@]}" -eq 0 ]; then
    echo "ERROR: GPU_PAIR 解析后为空: ${raw_value}" >&2
    exit 1
  fi
}

collect_shard_artifact_files() {
  local search_root="$1"
  local artifact_prefix="$2"
  local -n out_array="$3"
  mapfile -t out_array < <(find "${search_root}" -type f -name "${artifact_prefix}_*_${SHARD_RUN_ID}.json" | sort)
}

aggregate_sharded_outputs() {
  local output_root="$1"
  local -a shard_manifest_files=()
  local -a shard_summary_files=()
  local -a shard_result_files=()

  collect_shard_artifact_files "${output_root}" "batch_manifest" shard_manifest_files
  collect_shard_artifact_files "${output_root}" "summary" shard_summary_files
  collect_shard_artifact_files "${output_root}" "result" shard_result_files

  if [ "${#shard_result_files[@]}" -eq 0 ]; then
    echo "ERROR: 未找到 shard result 文件，无法汇总。run_id=${SHARD_RUN_ID}" >&2
    exit 1
  fi

  "${PYTHON_BIN}" - "${output_root}" "${SHARD_RUN_ID}" "${INPUT_JSON_LIST_PATH}" "${shard_manifest_files[@]}" -- "${shard_summary_files[@]}" -- "${shard_result_files[@]}" <<'PY'
import json
import sys
from pathlib import Path

argv = sys.argv[1:]
output_root = Path(argv[0])
run_id = argv[1]
input_list_path = argv[2]

sections: list[list[str]] = [[]]
for token in argv[3:]:
    if token == "--":
        sections.append([])
        continue
    sections[-1].append(token)

manifest_paths = [Path(p) for p in sections[0]]
summary_paths = [Path(p) for p in sections[1]] if len(sections) > 1 else []
result_paths = [Path(p) for p in sections[2]] if len(sections) > 2 else []
if not result_paths:
    raise SystemExit(f"no shard result files found for run_id={run_id}")

result_payloads = []
for path in result_paths:
    with path.open("r", encoding="utf-8") as handle:
        result_payloads.append(json.load(handle))

step_output_dir = result_paths[0].parent
combined_entries = []
num_total = 0
num_success = 0
num_failed = 0
num_skipped = 0
method_name = result_payloads[0].get("method")
checkpoint_dir = result_payloads[0].get("checkpoint_dir")
shard_tags = []
for payload in result_payloads:
    combined_entries.extend(payload.get("entries", []))
    num_total += int(payload.get("num_total", 0))
    num_success += int(payload.get("num_success", 0))
    num_failed += int(payload.get("num_failed", 0))
    num_skipped += int(payload.get("num_skipped", 0))
    shard_tag = payload.get("shard_tag")
    if shard_tag is not None:
      shard_tags.append(shard_tag)

combined_result = {
    "checkpoint_dir": checkpoint_dir,
    "method": method_name,
    "sharded": True,
    "shard_run_id": run_id,
    "shard_tags": shard_tags,
    "num_total": num_total,
    "num_success": num_success,
    "num_failed": num_failed,
    "num_skipped": num_skipped,
    "entries": combined_entries,
}
with (step_output_dir / "result.json").open("w", encoding="utf-8") as handle:
    json.dump(combined_result, handle, indent=2, ensure_ascii=False)
    handle.write("\n")

summary_payloads = []
for path in summary_paths:
    with path.open("r", encoding="utf-8") as handle:
        summary_payloads.append(json.load(handle))

weights_root = summary_payloads[0].get("weights_root") if summary_payloads else None
step_name = summary_payloads[0].get("step") if summary_payloads else None
combined_summary = {
    "weights_root": weights_root,
    "output_root": str(output_root),
    "step": step_name,
    "sharded": True,
    "shard_run_id": run_id,
    "shard_tags": shard_tags,
    "num_total": num_total,
    "num_success": num_success,
    "num_failed": num_failed,
    "num_skipped": num_skipped,
}
with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
    json.dump(combined_summary, handle, indent=2, ensure_ascii=False)
    handle.write("\n")

manifest_payloads = []
for path in manifest_paths:
    with path.open("r", encoding="utf-8") as handle:
        manifest_payloads.append(json.load(handle))

combined_manifest = dict(manifest_payloads[0]) if manifest_payloads else {"input_json_list_path": input_list_path}
combined_manifest["input_json_list_path"] = input_list_path
combined_manifest["num_items"] = num_total
combined_manifest["sharded"] = True
combined_manifest["shard_run_id"] = run_id
combined_manifest["shard_tags"] = shard_tags
combined_manifest["shard_manifests"] = [str(path) for path in manifest_paths]
combined_manifest["device"] = None
combined_manifest["aux_device"] = None
combined_manifest["inference_devices"] = None
with (output_root / "batch_manifest.json").open("w", encoding="utf-8") as handle:
    json.dump(combined_manifest, handle, indent=2, ensure_ascii=False)
    handle.write("\n")
PY
}

run_one_inference() {
  local gpu_pair="$1"
  local context_frames="$2"
  local run_output_root="$3"
  local run_model_name="$4"
  local step_output_dir_name="${5:-}"
  local run_input_json_list_path="${6:-${INPUT_JSON_LIST_PATH}}"
  local shard_tag="${7:-}"
  local disable_limit_flag="${8:-0}"
  local launch_visible_gpu_ids
  local launch_inference_devices
  local -a cmd

  check_gpu_pair_has_faulty_gpu4 "${gpu_pair}"
  prepare_launch_layout "${gpu_pair}" "${INFERENCE_DEVICES}" launch_visible_gpu_ids launch_inference_devices
  cmd=(
    env
    PYTHONNOUSERSITE=1
    PYTHONPATH="${PROJ}:${DIFFSYNTH_ROOT}"
    CUDA_VISIBLE_DEVICES="${launch_visible_gpu_ids}"
    "${PYTHON_BIN}"
    "${INFER_SCRIPT}"
    --weights-root "${WEIGHTS_ROOT}"
    --input-json-list-path "${run_input_json_list_path}"
    --model-name "${run_model_name}"
    --output-root "${run_output_root}"
    --height "${HEIGHT}"
    --width "${WIDTH}"
    --input-cover-crop-width "${INPUT_COVER_CROP_WIDTH}"
    --input-cover-crop-height "${INPUT_COVER_CROP_HEIGHT}"
    --context-frames "${context_frames}"
    --sampling-mode "${SAMPLING_MODE}"
    --num-inference-steps "${NUM_INFERENCE_STEPS}"
    --cfg-scale "${CFG_SCALE}"
    --seed "${SEED}"
    --fps "${FPS}"
    --negative-prompt "${NEGATIVE_PROMPT}"
    --output-num-frames "${OUTPUT_NUM_FRAMES}"
  )

  if [ -n "${launch_inference_devices}" ] && [ "${launch_inference_devices}" != "none" ]; then
    cmd+=(--inference-devices "${launch_inference_devices}")
  fi
  if [ "${DISABLE_OBJECT_BRANCH}" = "1" ]; then
    cmd+=(--disable-object-branch)
  fi
  if [ -n "${step_output_dir_name}" ]; then
    cmd+=(--step-output-dir-name "${step_output_dir_name}")
  fi
  if [ -n "${shard_tag}" ]; then
    cmd+=(--shard-tag "${shard_tag}")
  fi
  if [ -n "${LIMIT}" ] && [ "${disable_limit_flag}" != "1" ]; then
    cmd+=(--limit "${LIMIT}")
  fi
  if [ "${FORCE}" = "1" ]; then
    cmd+=(--force)
  fi
  if [ "${OVERWRITE}" = "1" ]; then
    cmd+=(--overwrite)
  fi
  if [ -n "${OBJECT_BRANCH_RATIO_GUARD_MAX_RATIO}" ]; then
    cmd+=(--object-branch-ratio-guard-max-ratio "${OBJECT_BRANCH_RATIO_GUARD_MAX_RATIO}")
  fi
  if [ -n "${OBJECT_BRANCH_RATIO_GUARD_MAX_BLOCK_ID}" ]; then
    cmd+=(--object-branch-ratio-guard-max-block-id "${OBJECT_BRANCH_RATIO_GUARD_MAX_BLOCK_ID}")
  fi
  if [ -n "${OBJECT_ADAPTER_MLP_RESIDUAL_MAX_RATIO}" ]; then
    cmd+=(--object-adapter-mlp-residual-max-ratio "${OBJECT_ADAPTER_MLP_RESIDUAL_MAX_RATIO}")
  fi
  if [ -n "${OBJECT_BRANCH_AUTO_FALLBACK_MAX_ACTIVE_SLOTS}" ]; then
    cmd+=(--object-branch-auto-fallback-max-active-slots "${OBJECT_BRANCH_AUTO_FALLBACK_MAX_ACTIVE_SLOTS}")
  fi
  if [ -n "${OBJECT_BRANCH_AUTO_FALLBACK_TRIGGER_COUNT}" ]; then
    cmd+=(--object-branch-auto-fallback-trigger-count "${OBJECT_BRANCH_AUTO_FALLBACK_TRIGGER_COUNT}")
  fi

  echo "[kubric-batch] gpu_pair=${gpu_pair} context_frames=${context_frames}"
  echo "[kubric-batch] cuda_visible_devices=${launch_visible_gpu_ids} inference_devices=${launch_inference_devices}"
  echo "[kubric-batch] disable_object_branch=${DISABLE_OBJECT_BRANCH}"
  echo "[kubric-batch] infer_script=${INFER_SCRIPT}"
  echo "[kubric-batch] input_json_list_path=${run_input_json_list_path}"
  if [ -n "${shard_tag}" ]; then
    echo "[kubric-batch] shard_tag=${shard_tag}"
  fi
  echo "[kubric-batch] output=${run_output_root}"
  echo "[kubric-batch] model_name=${run_model_name}"
  echo "[kubric-batch] command: ${cmd[*]}"
  "${cmd[@]}"
}

run_sweep_for_pair() {
  local gpu_pair="$1"
  shift
  local ctx
  local ctx_tag
  local ctx_output_root
  local ctx_model_name

  for ctx in "$@"; do
    printf -v ctx_tag "ctx%02d" "${ctx}"
    ctx_output_root="${OUTPUT_ROOT}/${ctx_tag}"
    ctx_model_name="${MODEL_NAME_PREFIX}_${ctx_tag}"
    run_one_inference "${gpu_pair}" "${ctx}" "${ctx_output_root}" "${ctx_model_name}"
  done
}

run_direct_mode() {
  if [ -n "${INFERENCE_GPU_PAIRS}" ]; then
    echo "ERROR: RUN_MODE=direct 时不要设置 INFERENCE_GPU_PAIRS" >&2
    exit 1
  fi
  if [[ "${VISIBLE_GPU_IDS}" == *" "* ]]; then
    echo "ERROR: GPU_PAIR 包含多个 worker pair 时，需要设置 AUTO_SPLIT_INPUT=1" >&2
    exit 1
  fi
  run_one_inference "${VISIBLE_GPU_IDS}" "${CONTEXT_FRAMES}" "${OUTPUT_ROOT}" "${MODEL_NAME}" "__METHOD_NAME__"
}

run_direct_auto_split_mode() {
  if [ -n "${INFERENCE_GPU_PAIRS}" ]; then
    echo "ERROR: AUTO_SPLIT_INPUT=1 暂不支持与 INFERENCE_GPU_PAIRS 同时使用" >&2
    exit 1
  fi
  if [ ! -f "${INPUT_JSON_LIST_PATH}" ]; then
    echo "ERROR: 输入列表不存在: ${INPUT_JSON_LIST_PATH}" >&2
    exit 1
  fi

  local -a worker_pairs=()
  local -a split_files=()
  local -a split_tags=()
  local -a child_pids=()
  local -a active_workers=()
  local pair_source
  local run_tag
  local split_dir
  local shard_index
  local shard_file
  local shard_tag
  local worker_pair
  local child_pid
  local status
  local -a failed_workers=()

  pair_source="${GPU_PAIR:-${VISIBLE_GPU_IDS}}"
  build_direct_split_worker_pairs "${pair_source}" worker_pairs
  run_tag="$(sanitize_path_tag "${MODEL_NAME}_${CONTEXT_FRAMES}_${SHARD_RUN_ID}")"
  split_dir="${SPLIT_WORK_ROOT%/}/${run_tag}"
  write_split_txt_files "${INPUT_JSON_LIST_PATH}" "${#worker_pairs[@]}" "${split_dir}" split_files split_tags

  echo "[kubric-batch] auto_split_input=1"
  echo "[kubric-batch] shard_run_id=${SHARD_RUN_ID}"
  echo "[kubric-batch] split_dir=${split_dir}"
  echo "[kubric-batch] worker_pairs=${worker_pairs[*]}"
  if [ "${#split_files[@]}" -eq 0 ]; then
    echo "ERROR: split txt 生成失败，没有任何 shard 文件" >&2
    exit 1
  fi

  for shard_index in "${!split_files[@]}"; do
    shard_file="${split_files[shard_index]}"
    shard_tag="${split_tags[shard_index]}_${SHARD_RUN_ID}"
    worker_pair="${worker_pairs[shard_index]}"
    echo "[kubric-batch] shard_plan ${split_tags[shard_index]} gpu_pair=${worker_pair} txt=${shard_file}"
    (
      run_one_inference "${worker_pair}" "${CONTEXT_FRAMES}" "${OUTPUT_ROOT}" "${MODEL_NAME}" "__METHOD_NAME__" "${shard_file}" "${shard_tag}" "1"
    ) &
    child_pids+=("$!")
    active_workers+=("${worker_pair}:${shard_tag}")
  done

  for shard_index in "${!child_pids[@]}"; do
    child_pid="${child_pids[shard_index]}"
    if wait "${child_pid}"; then
      echo "[kubric-batch] shard worker done ${active_workers[shard_index]}"
    else
      status=$?
      echo "[kubric-batch] shard worker failed ${active_workers[shard_index]} exit_code=${status}" >&2
      failed_workers+=("${active_workers[shard_index]}:${status}")
    fi
  done

  if [ "${#failed_workers[@]}" -gt 0 ]; then
    echo "[kubric-batch] failed_workers=${failed_workers[*]}" >&2
    exit 1
  fi

  aggregate_sharded_outputs "${OUTPUT_ROOT}"
  rm -rf "${split_dir}"
  echo "[kubric-batch] auto-split direct mode done. outputs under ${OUTPUT_ROOT}"
}

run_sweep_mode() {
  mkdir -p "${OUTPUT_ROOT}"
  declare -a ctx_values=()
  normalize_ctx_values "${CONTEXT_FRAME_VALUES}" ctx_values

  if [ -n "${INFERENCE_GPU_PAIRS}" ]; then
    read -r -a gpu_pairs <<< "${INFERENCE_GPU_PAIRS}"
    if [ "${#gpu_pairs[@]}" -eq 0 ]; then
      echo "ERROR: INFERENCE_GPU_PAIRS 解析后为空" >&2
      exit 1
    fi

    declare -a child_pids=()
    declare -a active_gpu_pairs=()
    local pair_index
    local ctx_index
    local gpu_pair
    for pair_index in "${!gpu_pairs[@]}"; do
      gpu_pair="${gpu_pairs[pair_index]}"
      declare -a assigned_ctx_values=()
      for ctx_index in "${!ctx_values[@]}"; do
        if [ $((ctx_index % ${#gpu_pairs[@]})) -eq "${pair_index}" ]; then
          assigned_ctx_values+=("${ctx_values[ctx_index]}")
        fi
      done
      if [ "${#assigned_ctx_values[@]}" -eq 0 ]; then
        continue
      fi

      active_gpu_pairs+=("${gpu_pair}")
      (
        run_sweep_for_pair "${gpu_pair}" "${assigned_ctx_values[@]}"
      ) &
      child_pids+=("$!")
    done

    if [ "${#child_pids[@]}" -eq 0 ]; then
      echo "ERROR: 没有可运行的 GPU pair / ctx 任务" >&2
      exit 1
    fi

    echo "[kubric-batch] running ${#child_pids[@]} sweep worker(s)"
    echo "[kubric-batch] gpu_pairs=${active_gpu_pairs[*]}"

    declare -a failed_workers=()
    local child_pid
    local status
    for pair_index in "${!child_pids[@]}"; do
      child_pid="${child_pids[pair_index]}"
      gpu_pair="${active_gpu_pairs[pair_index]}"
      if wait "${child_pid}"; then
        echo "[kubric-batch] worker done gpu_pair=${gpu_pair}"
      else
        status=$?
        echo "[kubric-batch] worker failed gpu_pair=${gpu_pair} exit_code=${status}" >&2
        failed_workers+=("${gpu_pair}:${status}")
      fi
    done

    if [ "${#failed_workers[@]}" -gt 0 ]; then
      echo "[kubric-batch] failed_workers=${failed_workers[*]}" >&2
      exit 1
    fi
    echo "[kubric-batch] sweep done. outputs under ${OUTPUT_ROOT}"
    return
  fi

  run_sweep_for_pair "${VISIBLE_GPU_IDS}" "${ctx_values[@]}"
  echo "[kubric-batch] sweep done. outputs under ${OUTPUT_ROOT}"
}

RUN_MODE="$(infer_run_mode)"

case "${RUN_MODE}" in
  direct)
    if [ "${AUTO_SPLIT_INPUT}" = "1" ]; then
      run_direct_auto_split_mode
    else
      run_direct_mode
    fi
    ;;
  sweep)
    if [ "${AUTO_SPLIT_INPUT}" = "1" ]; then
      echo "ERROR: AUTO_SPLIT_INPUT=1 当前只支持 RUN_MODE=direct（单个 context 值）" >&2
      exit 1
    fi
    run_sweep_mode
    ;;
  *)
    echo "ERROR: RUN_MODE 只支持 direct 或 sweep，收到 ${RUN_MODE}" >&2
    exit 1
    ;;
esac
