"""Prepare observed RGB only; no scene GT, future states, models, or CUDA."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

COUNT = 8
SCHEMA = "visual_context_rgb8_v1"
INPUT_KEYS = {"schema", "frames", "observed_indices", "time_s", "shape", "pixels_sha256"}


def context_paths(source: Path) -> list[Path]:
    paths = [source / f"frame_{i + 1:04d}.png" for i in range(COUNT)]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise ValueError(f"missing observed frames: {', '.join(missing)}")
    return paths


def read_rgb(paths: list[Path]) -> np.ndarray:
    if len(paths) != COUNT:
        raise ValueError("Exactly eight observed frames are required")
    frames = []
    for path in paths:
        with Image.open(path) as im:
            frames.append(np.asarray(im.convert("RGB")).copy())
    if any(frame.shape != frames[0].shape for frame in frames):
        raise ValueError("Observed frame dimensions differ")
    return np.stack(frames)


def pixel_hash(rgb: np.ndarray) -> str:
    return hashlib.sha256(rgb.tobytes()).hexdigest()


def load_model_input(input_path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(input_path.read_text())
    if set(payload) != INPUT_KEYS or payload["schema"] != SCHEMA:
        raise ValueError("Unexpected input fields or schema")
    expected = [f"rgb_{i:02d}.png" for i in range(COUNT)]
    if payload["frames"] != expected or payload["observed_indices"] != list(range(COUNT)):
        raise ValueError("Only RGB0-7 may enter the model")
    times = np.asarray(payload["time_s"], dtype=np.float64)
    if times.shape != (COUNT,) or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError("Invalid observation times")
    if abs(times[0]) > 1e-9:
        raise ValueError("Observation times must start at zero")
    rgb = read_rgb([input_path.parent / name for name in expected])
    if list(rgb.shape) != payload["shape"] or pixel_hash(rgb) != payload["pixels_sha256"]:
        raise ValueError("Observation pixels or shape changed")
    return rgb, times


def prepare_case(case: dict, output: Path) -> dict:
    if set(case) != {"case_id", "family", "frames_dir"}:
        raise ValueError("Source cases may only specify id, family, and frame directory")
    key = case["case_id"]
    if not isinstance(key, str) or not key or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in key):
        raise ValueError("Unsafe case id")
    source = Path(case["frames_dir"])
    paths = context_paths(source)
    rgb = read_rgb(paths)
    metadata_path = source / "render_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    fps = float(metadata["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid observed frame rate")
    target = output / key
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    target.mkdir(parents=True)
    names = [f"rgb_{i:02d}.png" for i in range(COUNT)]
    for path, name in zip(paths, names):
        shutil.copyfile(path, target / name)
    payload = dict(schema=SCHEMA, frames=names, observed_indices=list(range(COUNT)),
                   time_s=(np.arange(COUNT) / fps).tolist(), shape=list(rgb.shape),
                   pixels_sha256=pixel_hash(rgb))
    input_path = target / "input.json"
    input_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    verified, _ = load_model_input(input_path)
    if not np.array_equal(rgb, verified):
        raise RuntimeError("Exported observation pixels changed")
    return dict(case_id=key, family=case["family"], status="rgb8_ready",
                split="quality_pilot_unassigned", usage="development_preflight_only",
                input_json=str(input_path), source_frames_dir=str(source),
                source_frame_names=[p.name for p in paths], pixels_sha256=pixel_hash(rgb),
                source_camera_framing=metadata.get("camera", {}).get("framing_profile"),
                static_scene_gt_in_input=False, mask_in_input=False, camera_in_input=False,
                depth_extracted=False, metric_scene_ready=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    cases = json.loads(args.cases.read_text())
    if not isinstance(cases, list) or not cases:
        raise ValueError("Expected a nonempty source-case list")
    keys = [case["case_id"] for case in cases]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate case ids")
    args.output.mkdir(parents=True)
    rows = []
    for case in cases:
        try:
            rows.append(prepare_case(case, args.output))
        except (ValueError, OSError, KeyError) as exc:
            rows.append(dict(case_id=case["case_id"], family=case["family"],
                             status="blocked", reason=str(exc)))
    report = dict(schema="visual_context_preflight_v1", records=rows,
                  ready_count=sum(r["status"] == "rgb8_ready" for r in rows),
                  all_requested_contexts_ready=all(r["status"] == "rgb8_ready" for r in rows),
                  training_ready=False,
                  note="Input bundles contain observed RGB and times only; no feature extraction or training.")
    (args.output / "preflight.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

