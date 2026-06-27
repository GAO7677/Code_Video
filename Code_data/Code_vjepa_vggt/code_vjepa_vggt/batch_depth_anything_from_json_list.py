from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


DEFAULT_LIST = "/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt"
DEFAULT_OUTPUT_ROOT = "/data/gaoya/agent-data/outputs/depth_anything_test_5_sources"
DEFAULT_DEPTH_SCRIPT = "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/run_depth_anything_on_video.py"
DEFAULT_CHECKPOINT = "/data/gaoya/ckpt/LiheYoung-depth_anything_vitl14_raw/checkpoints/depth_anything_vitl14.pth"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-list", default=DEFAULT_LIST)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--depth-script", default=DEFAULT_DEPTH_SCRIPT)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _safe_stem(json_path: Path) -> str:
    return json_path.stem


def _resolve_source_video(payload: dict) -> str:
    for key in ("source_video", "input_video", "context_video"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise KeyError("could not resolve source video from json payload")


def main() -> None:
    args = _parse_args()
    list_path = Path(args.json_list).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    depth_script = Path(args.depth_script).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    records = []
    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        json_path = Path(line).expanduser().resolve()
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        source_video = Path(_resolve_source_video(payload)).expanduser().resolve()
        output_video = output_root / f"{_safe_stem(json_path)}__source_depth.mp4"
        records.append(
            {
                "json_path": str(json_path),
                "source_video": str(source_video),
                "output_video": str(output_video),
            }
        )
        if output_video.is_file() and not args.overwrite:
            print(f"[skip] {output_video}")
            continue
        cmd = [
            "bash",
            "-lc",
            (
                f"CUDA_VISIBLE_DEVICES={args.gpu} "
                "PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:"
                "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt:"
                "/home/gaoya/MimicBrush-main/depthanything "
                "/home/gaoya/miniconda3/envs/wan-cu128/bin/python "
                f"{depth_script} "
                f"--input-video {source_video} "
                f"--output-video {output_video} "
                f"--checkpoint {checkpoint}"
            ),
        ]
        print(f"[run] {json_path.name} -> {output_video.name}")
        subprocess.run(cmd, check=True)

    summary_path = output_root / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"[done] summary={summary_path}")


if __name__ == "__main__":
    main()
