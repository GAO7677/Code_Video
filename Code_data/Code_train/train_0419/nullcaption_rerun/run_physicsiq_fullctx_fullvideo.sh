#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=/data/gaoya/miniconda3/envs/wan/bin/python
SCRIPT_ROOT=/home/gaoya/Code_Video/Code_data/Code_train/train_0419
VACE_ROOT=/data/gaoya/ckpt/Wan-AI-Wan2.1-VACE-1.3B
DATA_ROOT=/data/gaoya/dataset/physics-iq-benchmark/mytest
BENCH_ROOT=/data/gaoya/AAA_test_video/Benchmark/stage0_V2V_nullcaption
WORK_ROOT=${BENCH_ROOT}/tools/fullctx_runs/physicsiq_fullvideo
META_ROOT=${WORK_ROOT}/meta
GEN_ROOT=${WORK_ROOT}/generated
RUNTIME_ROOT=${WORK_ROOT}/runtime
LOG_ROOT=${WORK_ROOT}/logs

CASES=(
  0002_perspective-center_trimmed-ball-and-block-fall
  0008_perspective-center_trimmed-ball-hits-duck
  0011_perspective-center_trimmed-ball-hits-nothing
  0014_perspective-center_trimmed-ball-in-basket
  0017_perspective-center_trimmed-ball-in-sand
)
GPUS=(0 1)

mkdir -p "${META_ROOT}" "${GEN_ROOT}" "${RUNTIME_ROOT}" "${LOG_ROOT}"

CASES_JOINED=$(printf '%s\n' "${CASES[@]}")
export CASES_JOINED
export DATA_ROOT META_ROOT

"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

cases = [line.strip() for line in os.environ["CASES_JOINED"].splitlines() if line.strip()]
data_root = Path(os.environ["DATA_ROOT"])
meta_root = Path(os.environ["META_ROOT"])

caption_paths = []
nullcaption_paths = []
context_lengths = set()
fps_values = set()
full_video_lengths = set()

for case_name in cases:
    src = data_root / case_name / "meta.json"
    data = json.loads(src.read_text(encoding="utf-8"))
    fps_values.add(int(float(data.get("fps") or 30)))
    context_range = data.get("context_frame_range") or [0, 0]
    future_range = data.get("future_frame_range") or [0, 0]
    context_length = int(context_range[1]) - int(context_range[0]) + 1
    future_length = int(future_range[1]) - int(future_range[0]) + 1
    full_length = context_length + future_length
    context_lengths.add(context_length)
    full_video_lengths.add(full_length)

    case_root = meta_root / case_name
    case_root.mkdir(parents=True, exist_ok=True)

    caption_data = dict(data)
    caption_data["caption"] = str(data.get("caption") or "")
    caption_path = case_root / "caption.json"
    caption_path.write_text(json.dumps(caption_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    caption_paths.append(str(caption_path))

    null_data = dict(data)
    null_data["caption"] = ""
    if "description" in null_data:
        null_data["description"] = ""
    null_path = case_root / "nullcaption.json"
    null_path.write_text(json.dumps(null_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    nullcaption_paths.append(str(null_path))

if len(context_lengths) != 1 or len(fps_values) != 1 or len(full_video_lengths) != 1:
    raise SystemExit(
        f"Expected shared context/fps/full lengths, got context={context_lengths}, fps={fps_values}, full={full_video_lengths}"
    )

(meta_root / "caption_all.txt").write_text("\n".join(caption_paths) + "\n", encoding="utf-8")
(meta_root / "nullcaption_all.txt").write_text("\n".join(nullcaption_paths) + "\n", encoding="utf-8")

manifest = {
    "cases": cases,
    "shared_context_frames": next(iter(context_lengths)),
    "shared_fps": next(iter(fps_values)),
    "shared_full_video_frames": next(iter(full_video_lengths)),
}
(meta_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

SHARED_CONTEXT_FRAMES=$("${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
print(json.loads(Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V_nullcaption/tools/fullctx_runs/physicsiq_fullvideo/meta/manifest.json").read_text())["shared_context_frames"])
PY
)
SHARED_FPS=$("${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
print(json.loads(Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V_nullcaption/tools/fullctx_runs/physicsiq_fullvideo/meta/manifest.json").read_text())["shared_fps"])
PY
)
SHARED_FULL_VIDEO_FRAMES=$("${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
print(json.loads(Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V_nullcaption/tools/fullctx_runs/physicsiq_fullvideo/meta/manifest.json").read_text())["shared_full_video_frames"])
PY
)

run_batch() {
  local gpu=$1
  local variant=$2
  local meta_list=$3
  local output_dir=${GEN_ROOT}/${variant}_fullctx_fullvideo
  local runtime_dir=${RUNTIME_ROOT}/${variant}_fullctx_fullvideo
  local log_path=${LOG_ROOT}/${variant}_fullctx_fullvideo.log
  mkdir -p "${output_dir}" "${runtime_dir}"

  CUDA_VISIBLE_DEVICES=${gpu} "${PYTHON_BIN}" "${SCRIPT_ROOT}/batch_eval_vace.py" \
    --vace_root "${VACE_ROOT}" \
    --meta_list_path "${meta_list}" \
    --output_root "${output_dir}" \
    --runtime_root "${runtime_dir}" \
    --model_name "physicsiq_${variant}_fullctx_fullvideo" \
    --mode v2v_clipref \
    --device cuda:0 \
    --height 544 \
    --width 720 \
    --fps "${SHARED_FPS}" \
    --num_frames "${SHARED_FULL_VIDEO_FRAMES}" \
    --context_frames "${SHARED_CONTEXT_FRAMES}" \
    --num_inference_steps 50 \
    --cfg_scale 5.0 \
    --seed 42 \
    --overwrite \
    2>&1 | tee "${log_path}"
}

(
  run_batch "${GPUS[0]}" caption "${META_ROOT}/caption_all.txt"
) &
PID0=$!

(
  run_batch "${GPUS[1]}" nullcaption "${META_ROOT}/nullcaption_all.txt"
) &
PID1=$!

wait "${PID0}" "${PID1}"

cat > "${WORK_ROOT}/README.txt" <<EOF
data_root=${DATA_ROOT}
cases=$(printf '%s,' "${CASES[@]}")
variants=caption,nullcaption
context_policy=use_full_context_video_frames_from_meta
output_policy=match_full_video_frame_count
shared_context_frames=${SHARED_CONTEXT_FRAMES}
shared_fps=${SHARED_FPS}
shared_full_video_frames=${SHARED_FULL_VIDEO_FRAMES}
generated_root=${GEN_ROOT}
runtime_root=${RUNTIME_ROOT}
log_root=${LOG_ROOT}
EOF

echo "done"
