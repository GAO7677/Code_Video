#!/usr/bin/env python3
"""Build a self-contained, native-resolution CYCLES benchmark package.

The source dataset is never modified.  The builder copies only the CYCLES
reference/context data, captions, provenance needed to regenerate aligned GT,
and small physics annotations.  Ten legacy 640x360 CYCLES cases are rendered
again at 896x512 before the common CYCLES/RigidBench GT export is run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2


DEFAULT_SOURCE = Path("/data/gaoya/AAA_test_video/physv_v2v_0819")
DEFAULT_OUTPUT = Path("/data/gaoya/AAA_test_video/physv_v2v_0819_strict")
DEFAULT_BLENDER = Path("/data/gaoya/agent-data/tools/blender-3.6.23-linux-x64/blender")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREVIEW_RUNNER = PROJECT_ROOT / "scripts/run_physv_cycles_previews.py"
TRUTH_RUNNER = PROJECT_ROOT / "scripts/generate_physv_cycles_aligned_truth.py"

COPY_FILES = (
    "metadata.json",
    "manifest.json",
    "meta.json",
    "export_summary.json",
    "physics_supervision.json",
    "physics_supervision.npz",
    "contacts.json",
    "videos/rgb_cycles.mp4",
    "videos/rgb_cycles.json",
    "context/context8_cycles.mp4",
    "context/context16_cycles.mp4",
    "raw/trajectories.npz",
    "raw/masks.npz",
    "raw/simulator_render_metadata.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu", default="0", help="One physical GPU for Blender/CYCLES rendering.")
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--ffmpeg", type=Path, default=Path(shutil.which("ffmpeg") or "ffmpeg"))
    parser.add_argument("--resume", action="store_true", help="Reuse an existing staging directory.")
    return parser.parse_args()


def run(command: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    print("+", " ".join(str(value) for value in command), flush=True)
    subprocess.run(command, check=True, env=env, cwd=cwd)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path, *, resume: bool) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if resume and destination.is_file() and destination.stat().st_size == source.stat().st_size:
        return
    shutil.copy2(source, destination)


def cycle_config(sample_dir: Path) -> dict[str, Any]:
    config = load_json(sample_dir / "videos/rgb_cycles.json")
    resolution = config.get("resolution") or [config["video"]["width"], config["video"]["height"]]
    video = config.get("video") or {}
    return {
        "resolution": [int(resolution[0]), int(resolution[1])],
        "frame_count": int(config.get("frame_count") or video.get("nb_frames", 0)),
        "fps": int(config.get("fps") or video.get("avg_frame_rate", "30/1").split("/")[0]),
    }


def stage_samples(source_root: Path, output_root: Path, *, resume: bool) -> tuple[list[Path], list[str]]:
    source_samples = sorted(
        path for path in (source_root / "samples").iterdir()
        if path.is_dir() and (path / "metadata.json").is_file()
    )
    strict_samples = output_root / "samples"
    low_resolution: list[str] = []
    for source in source_samples:
        destination = strict_samples / source.name
        for relative in COPY_FILES:
            if resume and relative == "videos/rgb_cycles.mp4" and destination.joinpath(relative).is_file():
                capture = cv2.VideoCapture(str(destination / relative))
                already_strict = (
                    capture.isOpened()
                    and int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 896
                    and int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 512
                    and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 90
                    and abs(float(capture.get(cv2.CAP_PROP_FPS)) - 30.0) < 0.1
                )
                capture.release()
                if already_strict:
                    continue
            copy_file(source / relative, destination / relative, resume=resume)
        captions = source / "captions"
        for caption in captions.iterdir():
            if caption.is_file():
                copy_file(caption, destination / "captions" / caption.name, resume=resume)
        config = cycle_config(source)
        if config["resolution"] != [896, 512]:
            low_resolution.append(source.name)
    return sorted(strict_samples.iterdir()), low_resolution


def regenerate_contexts(sample_dir: Path, ffmpeg: Path) -> None:
    video = sample_dir / "videos/rgb_cycles.mp4"
    for count, name in ((8, "context8_cycles.mp4"), (16, "context16_cycles.mp4")):
        target = sample_dir / "context" / name
        temporary = target.with_suffix(".tmp.mp4")
        run(
            [
                str(ffmpeg), "-y", "-loglevel", "error", "-i", str(video),
                "-frames:v", str(count), "-an", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
            ]
        )
        temporary.replace(target)


def write_first_frame(sample_dir: Path) -> None:
    capture = cv2.VideoCapture(str(sample_dir / "videos/rgb_cycles.mp4"))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not decode first CYCLES frame: {sample_dir}")
    target = sample_dir / "frames/00000.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), frame):
        raise RuntimeError(f"Could not write {target}")


def render_low_resolution_cases(
    output_root: Path,
    case_ids: list[str],
    blender: Path,
    ffmpeg: Path,
    gpu: str,
) -> None:
    if not case_ids:
        return
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    for case_id in case_ids:
        target = output_root / "samples" / case_id / "videos" / "rgb_cycles.mp4"
        capture = cv2.VideoCapture(str(target))
        existing_video = (
            capture.isOpened()
            and int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 896
            and int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 512
            and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 90
            and abs(float(capture.get(cv2.CAP_PROP_FPS)) - 30.0) < 0.1
        )
        capture.release()
        if existing_video:
            print(json.dumps({"case_id": case_id, "status": "render_exists", "video": str(target)}), flush=True)
            continue
        run(
            [
                sys.executable, str(PREVIEW_RUNNER), case_id,
                "--dataset-root", str(output_root),
                "--cache-root", str(output_root / ".strict_render_cache"),
                "--blender", str(blender), "--ffmpeg", str(ffmpeg),
                "--gpu", gpu, "--width", "896", "--height", "512", "--samples", "32",
                "--engine", "CYCLES",
            ],
            env=env,
            cwd=PROJECT_ROOT,
        )
        regenerate_contexts(output_root / "samples" / case_id, ffmpeg)


def recursively_replace(value: Any, old_root: str, new_root: str) -> Any:
    if isinstance(value, str):
        return value.replace(old_root, new_root)
    if isinstance(value, list):
        return [recursively_replace(item, old_root, new_root) for item in value]
    if isinstance(value, dict):
        return {key: recursively_replace(item, old_root, new_root) for key, item in value.items()}
    return value


def rewrite_test_jsons(source_root: Path, output_root: Path) -> None:
    source_dir = source_root / "testjsons/v2v_jsons/physv_v2v_0819_all_cycles"
    output_dir = output_root / "testjsons/v2v_jsons/physv_v2v_0819_all_cycles"
    output_dir.mkdir(parents=True, exist_ok=True)
    for source_json in sorted(source_dir.glob("*.json")):
        data = recursively_replace(load_json(source_json), str(source_root), str(output_root))
        case_id = str(data["sample_id"])
        sample = output_root / "samples" / case_id
        truth = output_root / "truth/cases" / case_id
        data["source_video"] = str(sample / "videos/rgb_cycles.mp4")
        data["input_video"] = str(sample / "context/context8_cycles.mp4")
        data["input_video_8f"] = str(sample / "context/context8_cycles.mp4")
        data["input_video_16f"] = str(sample / "context/context16_cycles.mp4")
        data["input_image"] = str(sample / "frames/00000.png")
        data["metadata_json"] = str(sample / "metadata.json")
        data["manifest_json"] = str(sample / "manifest.json")
        data["captions_json"] = str(sample / "captions/captions.json")
        data["physics_supervision_npz"] = str(sample / "physics_supervision.npz")
        data["physics_supervision_summary"] = str(sample / "physics_supervision.json")
        data["contacts_json"] = str(sample / "contacts.json")
        data["trajectories_npz"] = str(sample / "raw/trajectories.npz")
        data["masks_npz"] = str(truth / "rigidbench/masks.npz")
        data["depth_npz"] = str(truth / "rigidbench/depth.npz")
        data.setdefault("conditioning", {})["target_video"] = "videos/rgb_cycles.mp4"
        data["video_spec"] = {"width": 896, "height": 512, "fps": 30.0, "source_frame_count": 90}
        data["render_variant"] = "strict_cycles_pbr_896x512"
        data["conditioning_note"] = (
            "The strict benchmark reads context8_cycles.mp4; the continuation reference is rgb_cycles.mp4."
        )
        (output_dir / source_json.name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    output_testjsons = output_root / "testjsons"
    for source_list in sorted((source_root / "testjsons").glob("*.txt")):
        text = source_list.read_text(encoding="utf-8")
        text = text.replace(str(source_dir), str(output_dir))
        (output_testjsons / source_list.name).write_text(text, encoding="utf-8")


def materialize_adapters(truth_root: Path) -> None:
    adapter_dataset = truth_root / "rigidbench_dataset/samples"
    for case_dir in sorted((truth_root / "cases").iterdir()):
        if not case_dir.is_dir():
            continue
        adapter = case_dir / "rigidbench"
        video = adapter / "video.mp4"
        if video.is_symlink():
            source = video.resolve()
            video.unlink()
            shutil.copy2(source, video)
        target = adapter_dataset / case_dir.name
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.exists():
            shutil.rmtree(target)
        shutil.copytree(adapter, target)


def validate_case(output_root: Path, case_id: str) -> dict[str, Any]:
    sample = output_root / "samples" / case_id
    truth = output_root / "truth/cases" / case_id
    video = sample / "videos/rgb_cycles.mp4"
    capture = cv2.VideoCapture(str(video))
    actual_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    with np_load(truth / "dynamic_masks.npz") as masks_data:
        masks = masks_data["masks_thw"]
        union = masks_data["union_thw"]
    with np_load(truth / "cycles_depth.npz") as depth_data:
        depth = depth_data["depth"]
    with np_load(truth / "trajectory_pixels.npz") as trajectory_data:
        centers = trajectory_data["centers_tnc"]
    metadata = load_json(truth / "truth_metadata.json")
    checks = {
        "video": [actual_width, actual_height, actual_frames, round(actual_fps, 3)] == [896, 512, 90, 30.0],
        "mask": list(masks.shape[1:]) == [90, 512, 896] and union.shape == (90, 512, 896),
        "depth": list(depth.shape) == [90, 512, 896],
        "trajectory": centers.shape[0] == 90 and centers.shape[2] == 3,
        "metadata": metadata.get("resolution") == [896, 512] and metadata.get("frame_count") == 90,
        "adapter_video_is_copy": not (truth / "rigidbench/video.mp4").is_symlink(),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Validation failed for {case_id}: {checks}")
    return {
        "sample_id": case_id,
        "resolution": [896, 512],
        "frame_count": 90,
        "fps": 30,
        "dynamic_actor_count": int(masks.shape[0]),
        "video_sha256": sha256(video),
        "truth_depth_sha256": sha256(truth / "rigidbench/depth.npz"),
        "checks": checks,
    }


class np_load:
    """Small context manager so the builder stays dependency-light."""

    def __init__(self, path: Path):
        import numpy as np
        self.path = path
        self.np = np
        self.data = None

    def __enter__(self):
        self.data = self.np.load(self.path, allow_pickle=False)
        return self.data

    def __exit__(self, *_):
        if self.data is not None:
            self.data.close()


def write_strict_metadata(source_root: Path, output_root: Path, records: list[dict[str, Any]]) -> None:
    source_meta = load_json(source_root / "dataset_meta.json")
    metadata = copy.deepcopy(source_meta)
    metadata.update(
        {
            "dataset": "PhysV V2V 0819 Strict CYCLES Benchmark",
            "schema_version": "physv_v2v_0819_strict_cycles_v1",
            "protocol": {
                "reference_video": "CYCLES rgb_cycles.mp4",
                "width": 896,
                "height": 512,
                "fps": 30,
                "frame_count": 90,
                "official_rigidbench": False,
            },
            "records": records,
        }
    )
    (output_root / "dataset_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "dataset": metadata["dataset"],
        "schema_version": metadata["schema_version"],
        "protocol": metadata["protocol"],
        "sample_count": len(records),
        "samples": records,
        "excluded": [
            "non-CYCLES RGB/depth/mask/contact/trajectory MP4 files",
            "original PyRender raw depth/masks/instance_ids",
            "model-specific training caches",
        ],
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "benchmark.yaml").write_text(
        """name: physv_v2v_0819_strict_cycles
version: 1
reference_video: rgb_cycles.mp4
width: 896
height: 512
fps: 30
frames: 90
cases: 70
protocol: rigidbench-style-native-cycles
official_rigidbench: false
""",
        encoding="utf-8",
    )
    readme = f"""# PhysV V2V 0819 Strict CYCLES Benchmark

This is a self-contained benchmark package derived from the original dataset.

- cases: {len(records)}
- reference: `rgb_cycles.mp4`
- resolution: `896x512`
- frame rate: `30 FPS`
- frames per case: `90`
- protocol: `RigidBench-style · native CYCLES`
- official RigidBench full score: `false`

The package excludes the original non-CYCLES MP4s and model-specific training caches.
Each case contains CYCLES RGB/context, captions, provenance trajectory data, and
aligned mask/depth/trajectory GT under `truth/cases/<case_id>/`.
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing strict benchmark: {output_root}")
    if not args.blender.is_file():
        raise FileNotFoundError(args.blender)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "reports").mkdir(exist_ok=True)

    samples, low_resolution = stage_samples(source_root, output_root, resume=args.resume)
    print(json.dumps({"staged_samples": len(samples), "low_resolution": low_resolution}, ensure_ascii=False), flush=True)
    render_low_resolution_cases(output_root, low_resolution, args.blender, args.ffmpeg, args.gpu)
    for sample in samples:
        regenerate_contexts(sample, args.ffmpeg)
        write_first_frame(sample)
    rewrite_test_jsons(source_root, output_root)

    truth_root = output_root / "truth"
    run(
        [
            sys.executable, str(TRUTH_RUNNER),
            "--dataset-root", str(output_root),
            "--output-root", str(truth_root),
            "--gpus", args.gpu,
        ],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu},
        cwd=PROJECT_ROOT,
    )
    materialize_adapters(truth_root)
    records = [validate_case(output_root, sample.name) for sample in samples]
    write_strict_metadata(source_root, output_root, records)
    (output_root / "reports/integrity_report.json").write_text(
        json.dumps({"status": "complete", "records": records}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    render_cache = output_root / ".strict_render_cache"
    if render_cache.exists():
        shutil.rmtree(render_cache)
    print(json.dumps({"status": "complete", "cases": len(records), "output_root": str(output_root)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
