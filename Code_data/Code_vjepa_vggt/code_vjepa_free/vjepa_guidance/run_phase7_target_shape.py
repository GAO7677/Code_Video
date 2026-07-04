#!/usr/bin/env python3
"""
Run command:
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main:/home/gaoya/Code_Video/Code_data/Code_train/train_0419 \
CUDA_VISIBLE_DEVICES=6,7 /data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_phase7_target_shape.py \
  --weights-root /data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500 \
  --input-json /data/gaoya/AAA_test_video/0623/testdataset/025_Solid_Mechanics_0002_perspective-center_trimmed/025_Solid_Mechanics_0002_perspective-center_trimmed.json \
  --source-video /data/gaoya/AAA_test_video/0623/testdataset/025_Solid_Mechanics_0002_perspective-center_trimmed/physicIQ_0002_clip_2p5s_3p5s.mp4 \
  --device cuda:0 \
  --vjepa-device cuda:1
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
PROBE_SCRIPT = THIS_DIR / "probe_energy_persistence.py"
SCORE_SCRIPT = THIS_DIR / "score_guided_videos.py"
DEFAULT_OUTPUT_DIR = Path("/data/gaoya/agent-data/outputs/probe_sweep")
DEFAULT_WAN_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
DEFAULT_VJEPA_CKPT = Path("/data/gaoya/ckpt/VJEPA2/vith.pt")
DEFAULT_PYTHON_BIN = Path("/data/gaoya/miniconda3/envs/wan/bin/python")
DEFAULT_PYTHONPATH = (
    "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:"
    "/home/gaoya/Code_Video/DiffSynth-Studio-main:"
    "/home/gaoya/Code_Video/Code_data/Code_train/train_0419"
)
DEFAULT_SCORE_PYTHONPATH = "/home/gaoya/Code_Video/Code_data/Code_try0526"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the single-case Phase 7 target-shape sweep (generation) and then "
            "score the resulting videos with read-only physv_eval metrics."
        )
    )
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--context-path", type=Path, default=None)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--wan-root", type=Path, default=DEFAULT_WAN_ROOT)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--vjepa-device", type=str, default="cuda:1")
    parser.add_argument("--vjepa-model", type=str, default="vith")
    parser.add_argument("--vjepa-ckpt", type=Path, default=DEFAULT_VJEPA_CKPT)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON_BIN)
    parser.add_argument("--pythonpath", type=str, default=DEFAULT_PYTHONPATH)
    parser.add_argument("--score-pythonpath", type=str, default=DEFAULT_SCORE_PYTHONPATH)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--anchor-timing", type=float, default=0.35)
    parser.add_argument("--probe-every-n", type=int, default=2)
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--skip-score", action="store_true")
    return parser.parse_args()


def _resolve_context_path(input_json: Path, explicit_context_path: Path | None) -> Path:
    if explicit_context_path is not None:
        return explicit_context_path.expanduser().resolve()
    import json

    payload = json.loads(input_json.read_text(encoding="utf-8"))
    context_path = payload.get("context_path")
    if not context_path:
        raise FileNotFoundError(
            "No --context-path provided and input JSON does not contain context_path"
        )
    return Path(context_path).expanduser().resolve()


def _run(cmd: list[str], *, env: dict[str, str]) -> None:
    print("[RUN]", subprocess.list2cmdline(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    args = parse_args()
    input_json = args.input_json.expanduser().resolve()
    if not input_json.is_file():
        raise FileNotFoundError(f"Input JSON not found: {input_json}")

    context_path = _resolve_context_path(input_json, args.context_path)
    if not context_path.is_file():
        raise FileNotFoundError(f"Context path not found: {context_path}")

    source_video = args.source_video.expanduser().resolve()
    if not source_video.is_file():
        raise FileNotFoundError(f"Source video not found: {source_video}")

    weights_root = args.weights_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    phase_dir = output_dir / "phase7"
    videos_dir = phase_dir / "videos"
    summary_json = phase_dir / "phase7_summary.json"
    score_json = phase_dir / "phase7_multimetric_scores.json"

    generate_env = os.environ.copy()
    generate_env["PYTHONPATH"] = args.pythonpath

    score_env = os.environ.copy()
    score_env["PYTHONPATH"] = args.score_pythonpath
    score_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if not args.skip_generate:
        generate_cmd = [
            str(args.python_bin),
            str(PROBE_SCRIPT),
            "--weights-root", str(weights_root),
            "--input-json", str(input_json),
            "--context-path", str(context_path),
            "--output-dir", str(output_dir),
            "--wan-root", str(args.wan_root.expanduser().resolve()),
            "--device", args.device,
            "--vjepa-device", args.vjepa_device,
            "--vjepa-model", args.vjepa_model,
            "--vjepa-ckpt", str(args.vjepa_ckpt.expanduser().resolve()),
            "--phase", "7",
            "--probe-every-n", str(args.probe_every_n),
            "--num-frames", str(args.num_frames),
            "--height", str(args.height),
            "--width", str(args.width),
            "--num-inference-steps", str(args.num_inference_steps),
            "--cfg-scale", str(args.cfg_scale),
            "--seed", str(args.seed),
            "--anchor-timing", str(args.anchor_timing),
        ]
        _run(generate_cmd, env=generate_env)

    if args.skip_score:
        return

    if not videos_dir.is_dir():
        raise FileNotFoundError(f"Phase 7 videos directory not found: {videos_dir}")
    if not summary_json.is_file():
        raise FileNotFoundError(f"Phase 7 summary not found: {summary_json}")

    score_cmd = [
        str(args.python_bin),
        str(SCORE_SCRIPT),
        "--videos-dir", str(videos_dir),
        "--summary-json", str(summary_json),
        "--out-json", str(score_json),
        "--source-video", str(source_video),
        "--physics-iq",
        "--videophy2-task", "pc",
        "--videophy2-device", "cuda:0",
        "--cosmos-reason1",
    ]
    _run(score_cmd, env=score_env)


if __name__ == "__main__":
    main()
