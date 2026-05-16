"""Select a few Genesis rigid samples and render them with Blender Cycles."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PREVIEW_SCRIPT = SCRIPT_DIR / "build_blender_cycles_case_preview.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--quality", choices=("preview", "final"), default="final")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--frame_stride", type=int, default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    return parser.parse_args()


def is_renderable_sample(path: Path) -> bool:
    return (
        (path / "meta.json").exists()
        and (path / "physics" / "rigid_kinematics.npz").exists()
        and (path / "physics" / "anchor_targets.npz").exists()
    )


def iter_sample_dirs(dataset_root: Path) -> list[Path]:
    samples = []
    for count_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir() and p.name.startswith("count_")):
        for sample_dir in sorted(p for p in count_dir.iterdir() if p.is_dir()):
            if is_renderable_sample(sample_dir):
                samples.append(sample_dir)
    return samples


def choose_samples(sample_dirs: list[Path], limit: int) -> list[Path]:
    chosen = []
    seen_object_ids: set[str] = set()
    seen_cases: set[str] = set()
    for sample_dir in sample_dirs:
        meta = json.loads((sample_dir / "meta.json").read_text(encoding="utf-8"))
        object_key = str(meta.get("object_id") or meta.get("scene_id") or sample_dir.name)
        case_key = str(meta.get("case_name") or sample_dir.name)
        novelty = int(object_key not in seen_object_ids) + int(case_key not in seen_cases)
        if novelty <= 0 and len(chosen) >= limit:
            continue
        chosen.append(sample_dir)
        seen_object_ids.add(object_key)
        seen_cases.add(case_key)
        if len(chosen) >= limit:
            break
    return chosen


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    sample_dirs = iter_sample_dirs(dataset_root)
    selected = choose_samples(sample_dirs, args.limit)
    if not selected:
        raise RuntimeError(f"No renderable samples found under {dataset_root}")

    manifest = []
    for index, sample_dir in enumerate(selected, start=1):
        case_output = output_root / f"{index:02d}_{sample_dir.name}"
        cmd = [
            sys.executable,
            str(PREVIEW_SCRIPT),
            "--sample_dir",
            str(sample_dir),
            "--output_root",
            str(case_output),
            "--quality",
            str(args.quality),
            "--device",
            str(args.device),
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        if args.denoise:
            cmd.append("--denoise")
        for key in ("width", "height", "samples", "frame_stride", "max_frames", "fps"):
            value = getattr(args, key)
            if value is not None:
                cmd.extend([f"--{key}", str(value)])
        subprocess.run(cmd, check=True)
        manifest.append(
            {
                "sample_dir": str(sample_dir),
                "output_dir": str(case_output),
                "quality": args.quality,
            }
        )

    (output_root / "render_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
