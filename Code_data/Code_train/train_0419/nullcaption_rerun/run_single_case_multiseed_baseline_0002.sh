#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=/data/gaoya/miniconda3/envs/wan/bin/python
SCRIPT_ROOT=/home/gaoya/Code_Video/Code_data/Code_train/train_0419
VACE_ROOT=/data/gaoya/ckpt/Wan-AI-Wan2.1-VACE-1.3B
CASE_NAME=0002_perspective-center_trimmed-ball-and-block-fall
CASE_META=/data/gaoya/dataset/physics-iq-benchmark/mytest/${CASE_NAME}/meta.json
BENCH_ROOT=/data/gaoya/AAA_test_video/Benchmark/stage0_V2V_nullcaption
WORK_ROOT=${BENCH_ROOT}/tools/seed_sweeps/0002_baseline_multiseed
META_ROOT=${WORK_ROOT}/meta
GEN_ROOT=${WORK_ROOT}/generated
RUNTIME_ROOT=${WORK_ROOT}/runtime
LOG_ROOT=${WORK_ROOT}/logs
GPUS=(0 1)
SEEDS=(42 52 62 72 82)

mkdir -p "${META_ROOT}" "${GEN_ROOT}" "${RUNTIME_ROOT}" "${LOG_ROOT}"
export CASE_META META_ROOT SEEDS_CSV
SEEDS_CSV="$(IFS=, ; echo "${SEEDS[*]}")"

"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

case_meta = Path(os.environ["CASE_META"])
meta_root = Path(os.environ["META_ROOT"])
seeds = [int(x) for x in os.environ["SEEDS_CSV"].split(",") if x.strip()]
data = json.loads(case_meta.read_text(encoding="utf-8"))

context_range = data.get("context_frame_range") or [0, 0]
future_range = data.get("future_frame_range") or [0, 0]
context_frames = int(context_range[1]) - int(context_range[0]) + 1
future_frames = int(future_range[1]) - int(future_range[0]) + 1
full_frames = context_frames + future_frames
fps = int(float(data.get("fps") or 30))

caption_data = dict(data)
caption_data["caption"] = str(data.get("caption") or "")
(meta_root / "caption.json").write_text(
    json.dumps(caption_data, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

null_data = dict(data)
null_data["caption"] = ""
if "description" in null_data:
    null_data["description"] = ""
(meta_root / "nullcaption.json").write_text(
    json.dumps(null_data, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

(meta_root / "caption.txt").write_text(str((meta_root / "caption.json").resolve()) + "\n", encoding="utf-8")
(meta_root / "nullcaption.txt").write_text(str((meta_root / "nullcaption.json").resolve()) + "\n", encoding="utf-8")

manifest = {
    "case_name": str(data.get("sample_id") or ""),
    "case_meta": str(case_meta),
    "shared_context_frames": context_frames,
    "shared_future_frames": future_frames,
    "shared_full_video_frames": full_frames,
    "shared_fps": fps,
    "seeds": seeds,
}
(meta_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest, ensure_ascii=False, indent=2))
PY

SHARED_CONTEXT_FRAMES=$("${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
print(json.loads(Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V_nullcaption/tools/seed_sweeps/0002_baseline_multiseed/meta/manifest.json").read_text())["shared_context_frames"])
PY
)
SHARED_FPS=$("${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
print(json.loads(Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V_nullcaption/tools/seed_sweeps/0002_baseline_multiseed/meta/manifest.json").read_text())["shared_fps"])
PY
)
SHARED_FULL_VIDEO_FRAMES=$("${PYTHON_BIN}" - <<'PY'
import json
from pathlib import Path
print(json.loads(Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V_nullcaption/tools/seed_sweeps/0002_baseline_multiseed/meta/manifest.json").read_text())["shared_full_video_frames"])
PY
)

run_job() {
  local gpu=$1
  local variant=$2
  local seed=$3
  local meta_list_path=$4

  local seed_tag
  seed_tag=$(printf '%04d' "${seed}")
  local output_dir=${GEN_ROOT}/${variant}/seed_${seed_tag}
  local runtime_dir=${RUNTIME_ROOT}/${variant}/seed_${seed_tag}
  local log_path=${LOG_ROOT}/${variant}_seed_${seed_tag}.log

  mkdir -p "${output_dir}" "${runtime_dir}"
  echo "[start] variant=${variant} seed=${seed} gpu=${gpu}"

  CUDA_VISIBLE_DEVICES=${gpu} "${PYTHON_BIN}" "${SCRIPT_ROOT}/batch_eval_vace.py" \
    --vace_root "${VACE_ROOT}" \
    --meta_list_path "${meta_list_path}" \
    --output_root "${output_dir}" \
    --runtime_root "${runtime_dir}" \
    --model_name "case0002_${variant}_fullctx_fullvideo_seed${seed_tag}" \
    --mode v2v_clipref \
    --device cuda:0 \
    --height 544 \
    --width 720 \
    --fps "${SHARED_FPS}" \
    --num_frames "${SHARED_FULL_VIDEO_FRAMES}" \
    --context_frames "${SHARED_CONTEXT_FRAMES}" \
    --num_inference_steps 50 \
    --cfg_scale 5.0 \
    --seed "${seed}" \
    --overwrite \
    2>&1 | tee "${log_path}"
}

max_parallel=${#GPUS[@]}
running=0
gpu_index=0

for seed in "${SEEDS[@]}"; do
  for variant in caption nullcaption; do
    gpu=${GPUS[$gpu_index]}
    gpu_index=$(((gpu_index + 1) % max_parallel))
    if [[ "${variant}" == "caption" ]]; then
      meta_list="${META_ROOT}/caption.txt"
    else
      meta_list="${META_ROOT}/nullcaption.txt"
    fi
    (
      run_job "${gpu}" "${variant}" "${seed}" "${meta_list}"
    ) &
    running=$((running + 1))
    if (( running >= max_parallel )); then
      wait -n
      running=$((running - 1))
    fi
  done
done

wait

export WORK_ROOT META_ROOT GEN_ROOT RUNTIME_ROOT LOG_ROOT
"${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

work_root = Path(os.environ["WORK_ROOT"])
meta_root = Path(os.environ["META_ROOT"])
gen_root = Path(os.environ["GEN_ROOT"])
runtime_root = Path(os.environ["RUNTIME_ROOT"])
log_root = Path(os.environ["LOG_ROOT"])
manifest = json.loads((meta_root / "manifest.json").read_text(encoding="utf-8"))

rows = []
for seed in manifest.get("seeds", []):
    seed_tag = f"{int(seed):04d}"
    for variant in ("caption", "nullcaption"):
        output_dir = gen_root / variant / f"seed_{seed_tag}"
        runtime_dir = runtime_root / variant / f"seed_{seed_tag}"
        videos = sorted(output_dir.glob("*.mp4"))
        jsons = sorted(output_dir.glob("*.json"))
        summary_paths = sorted((runtime_dir / "metadata").rglob("*_summary.json"))
        rows.append(
            {
                "seed": int(seed),
                "variant": variant,
                "output_dir": str(output_dir),
                "output_video": str(videos[0]) if videos else None,
                "output_json": str(jsons[0]) if jsons else None,
                "runtime_summary": str(summary_paths[0]) if summary_paths else None,
                "log_path": str(log_root / f"{variant}_seed_{seed_tag}.log"),
            }
        )

report = {
    "work_root": str(work_root),
    "manifest_path": str(meta_root / "manifest.json"),
    "rows": rows,
}
(work_root / "result_index.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2))
PY

cat > "${WORK_ROOT}/README.txt" <<EOF
case_meta=${CASE_META}
case_name=${CASE_NAME}
variants=caption,nullcaption
seeds=$(printf '%s,' "${SEEDS[@]}")
context_policy=use_full_context_video_frames_from_meta
output_policy=match_full_video_frame_count
shared_context_frames=${SHARED_CONTEXT_FRAMES}
shared_fps=${SHARED_FPS}
shared_full_video_frames=${SHARED_FULL_VIDEO_FRAMES}
generated_root=${GEN_ROOT}
runtime_root=${RUNTIME_ROOT}
log_root=${LOG_ROOT}
result_index_json=${WORK_ROOT}/result_index.json
EOF

echo "done"
