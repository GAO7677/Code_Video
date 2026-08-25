#!/usr/bin/env python3
"""Create a few PBR-textured Eevee renders for the random-position demos."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from Dataset_physv_v2v_0819.scripts.generate_external_variants_demo import (  # noqa: E402
    _blueprint_from_meta,
)
from Dataset_physv_v2v_0819.scripts.render_sim_0705 import render_blueprint_case  # noqa: E402


DEFAULT_SOURCE_ROOT = Path(
    "/data/gaoya/agent-data/datasets/pybullet0717_prompt_physics_consistency_v1"
) / "external_collision_random_position_demo"
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/datasets/pybullet0717_prompt_physics_consistency_v1"
) / "external_collision_texture_realism_demo"
DEFAULT_CASES = (
    "F2/0717_f2_attempt001614__randpos01",
    "F2/0717_f2_attempt001096__randpos01",
    "F3/0717_f3_attempt000307__randpos01",
)
BACKGROUND_PROFILES = (
    "warehouse_cobalt",
    "machine_shop_amber",
    "color_studio",
    "glasshouse_mint",
    "courtyard_terracotta",
    "foundry_safety",
    "garage_teal",
    "neon_studio",
)
DEFAULT_BLENDER = Path("/data/gaoya/agent-data/tools/blender-3.6.23-linux-x64/blender")
DEFAULT_FFMPEG = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/ffmpeg")


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def find_case(root: Path, relative: str) -> Path:
    case = root / "cases" / relative
    if not (case / "case_manifest.json").is_file():
        raise FileNotFoundError(case / "case_manifest.json")
    return case


def encode_video(ffmpeg: Path, frames_dir: Path, target: Path, fps: int) -> None:
    temporary = target.with_suffix(".tmp.mp4")
    run(
        [
            str(ffmpeg),
            "-y",
            "-loglevel",
            "warning",
            "-framerate",
            str(fps),
            "-start_number",
            "1",
            "-i",
            str(frames_dir / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary),
        ]
    )
    temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--blender", type=Path, default=DEFAULT_BLENDER)
    parser.add_argument("--ffmpeg", type=Path, default=DEFAULT_FFMPEG)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--exposure", type=float, default=-0.15)
    parser.add_argument("--frame-limit", type=int, default=0)
    parser.add_argument(
        "--background-profile",
        choices=BACKGROUND_PROFILES,
        help="Use one profile for every case; otherwise cycle through the profile list.",
    )
    parser.add_argument(
        "--all-backgrounds",
        action="store_true",
        help="Render one background variant for each profile, distributed across the selected source cases.",
    )
    parser.add_argument("--gpu", default="7", help="GPU exposed to Blender's EGL context; GPU 4 is never used.")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    work_root = output_root / "_work"
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    blender_script = Path(__file__).with_name("render_texture_realism_eevee.py").resolve()
    selected = tuple(args.cases or DEFAULT_CASES)
    generated: list[dict[str, object]] = []
    source_records: list[dict[str, object]] = []
    env = os.environ.copy()
    # Eevee uses an EGL context in background mode. Restrict it to a free,
    # explicitly selected device and never expose the forbidden GPU 4.
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["EGL_DEVICE_ID"] = "0"

    try:
        if args.all_backgrounds:
            jobs = [
                (index, BACKGROUND_PROFILES[index], selected[index % len(selected)])
                for index in range(len(BACKGROUND_PROFILES))
            ]
        else:
            jobs = [
                (index, args.background_profile or BACKGROUND_PROFILES[index % len(BACKGROUND_PROFILES)], relative_case)
                for index, relative_case in enumerate(selected)
            ]

        for job_index, background_profile, relative_case in jobs:
            baseline_case = find_case(source_root, relative_case)
            baseline_manifest = json.loads((baseline_case / "case_manifest.json").read_text(encoding="utf-8"))
            baseline_meta_path = Path(str(baseline_manifest["meta"]))
            baseline_meta = json.loads(baseline_meta_path.read_text(encoding="utf-8"))
            case_key = str(baseline_manifest["sample_key"])
            family = str(baseline_manifest.get("family_key", baseline_meta.get("blueprint", {}).get("family_key", "")))
            rendered_case_key = f"{case_key}__{background_profile}"
            target_case = output_root / "cases" / family / rendered_case_key
            frames_dir = work_root / rendered_case_key / "frames"
            staging_case = work_root / rendered_case_key / "physics_staging"
            frames_dir.mkdir(parents=True, exist_ok=True)
            staging_case.mkdir(parents=True, exist_ok=True)

            # Reconstruct exactly the already-rendered blueprint and save its
            # PyBullet poses. This does not change the random initialization.
            blueprint = _blueprint_from_meta(baseline_meta, case_key)
            staged_manifest = render_blueprint_case(
                blueprint=blueprint,
                seed=int(baseline_manifest.get("seed", baseline_meta.get("seed", 0))),
                output_root=staging_case,
                width=args.width,
                height=args.height,
                scene_style="indoor_realistic",
                preserve_states=True,
            )
            staged_meta = Path(str(staged_manifest["meta"]))
            staged_states = Path(str(staged_manifest["states"]))
            staged_meta_payload = json.loads(staged_meta.read_text(encoding="utf-8"))
            fps = int(staged_meta_payload.get("fps", 30))

            run(
                [
                    str(args.blender),
                    "-b",
                    "--python",
                    str(blender_script),
                    "--",
                    "--meta",
                    str(staged_meta),
                    "--states",
                    str(staged_states),
                    "--output-dir",
                    str(frames_dir),
                    "--width",
                    str(args.width),
                    "--height",
                    str(args.height),
                    "--samples",
                    str(args.samples),
                    "--exposure",
                    str(args.exposure),
                    "--frame-limit",
                    str(args.frame_limit),
                    "--background-profile",
                    background_profile,
                ],
                env=env,
            )
            render_report = json.loads((frames_dir / "render_metadata.json").read_text(encoding="utf-8"))
            frame_count = int(render_report["frame_count"])
            frame_paths = sorted(frames_dir.glob("frame_*.png"))
            if len(frame_paths) != frame_count:
                raise RuntimeError(f"{case_key}: expected {frame_count} frames, got {len(frame_paths)}")

            video_dir = target_case / "videos"
            meta_dir = target_case / "meta"
            video_dir.mkdir(parents=True, exist_ok=True)
            meta_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / f"{rendered_case_key}__eevee_pbr.mp4"
            encode_video(args.ffmpeg, frames_dir, video_path, fps)

            improved_meta = dict(baseline_meta)
            improved_meta.update(
                {
                    "render_variant": "independent_background_pbr_fast_eevee",
                    "render_engine": "BLENDER_EEVEE",
                    "cycles_used": False,
                    "background_profile": background_profile,
                    "background_variant_index": job_index,
                    "source_case_key": case_key,
                    "render_metadata": render_report,
                    "output_video": str(video_path),
                    "state_source": str(staged_states),
                }
            )
            if "effective_camera" in render_report:
                improved_meta["camera"] = render_report["effective_camera"]
            improved_meta_path = meta_dir / f"{rendered_case_key}__eevee_pbr.json"
            improved_meta_path.write_text(
                json.dumps(improved_meta, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            item = {
                "sample_key": f"{rendered_case_key}__eevee_pbr",
                "family_key": family,
                "seed": baseline_manifest.get("seed"),
                "output_root": str(target_case),
                "video": str(video_path),
                "meta": str(improved_meta_path),
                "source_case_id": case_key,
                "source_video": str(baseline_manifest["video"]),
                "source_meta": str(baseline_manifest["meta"]),
                "variant_index": job_index,
                "variant_seed": baseline_manifest.get("variant_seed"),
                "perturbations": baseline_manifest.get("perturbations", {}),
                "collision_summary": baseline_manifest.get("collision_summary", {}),
                "texture_render": render_report,
                "background_profile": background_profile,
            }
            generated.append(item)
            source_records.append(
                {
                    "source_case_id": case_key,
                    "background_profile": background_profile,
                    "baseline_video": str(baseline_manifest["video"]),
                    "baseline_renderer": "pyrender fast indoor_realistic",
                    "improved_video": str(video_path),
                    "improved_renderer": f"Blender Eevee + independent {background_profile} studio + PolyHaven PBR assets",
                }
            )
            print(json.dumps({"case": case_key, "video": str(video_path), "frames": frame_count}, ensure_ascii=False), flush=True)
    finally:
        shutil.rmtree(work_root, ignore_errors=True)

    (output_root / "manifest.json").write_text(json.dumps(generated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_root / "source_records.json").write_text(json.dumps(source_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_root / "README.json").write_text(
        json.dumps(
            {
                "purpose": "side-by-side texture realism demos: existing pyrender baseline versus fast Eevee PBR with independent backgrounds and motion-following cameras",
                "reference_page": "http://localhost:8844/physv-v2v-0819-test70-no-event-timing-40step",
                "engine": "BLENDER_EEVEE",
                "cycles_used": False,
                "texture_profile": "polyhaven_pbr_fast_eevee_textured_objects",
                "background_profiles": list(BACKGROUND_PROFILES),
                "camera_mode": "smoothed_dynamic_centroid_follow",
                "cases": list(selected),
                "background_jobs": [
                    {"source_case": relative_case, "background_profile": background_profile, "index": job_index}
                    for job_index, background_profile, relative_case in jobs
                ],
                "gpu": str(args.gpu),
                "no_cache_generated": True,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"generated={len(generated)} output_root={output_root}")


if __name__ == "__main__":
    main()
