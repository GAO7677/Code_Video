#!/usr/bin/env python3
"""Generate a high-diversity, physically consistent render set for the F2/F3 demos.

The source cases are reconstructed into new PyBullet runs before the Eevee
pass.  Geometry, mass-at-constant-density, appearance texture selection,
background tint/texture mapping, and fixed-camera composition are recorded in
the output metadata.  The source dataset and earlier demo roots are never
modified.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import shutil
import subprocess
import sys
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from Dataset_physv_v2v_0819.scripts.generate_external_variants_demo import (  # noqa: E402
    _blueprint_from_meta,
)
from Dataset_physv_v2v_0819.scripts.render_sim_0705 import (  # noqa: E402
    render_blueprint_case,
)
from Dataset_physv_v2v_0819.scripts.render_blueprint_states_only import (  # noqa: E402
    render_blueprint_states_only,
)
from Dataset_physv_v2v_0819.scripts.scene_generators_0705 import (  # noqa: E402
    _collision_vertical_extent,
    _horizontal_clearance_radius,
)


SOURCE_ROOT_DEFAULT = Path(
    "/data/gaoya/agent-data/datasets/pybullet0717_prompt_physics_consistency_v1"
) / "external_collision_random_position_demo"
OUTPUT_ROOT_DEFAULT = Path(
    "/data/gaoya/agent-data/datasets/pybullet0717_prompt_physics_consistency_v1"
) / "external_collision_texture_realism_demo_high_diversity"
BLENDER_DEFAULT = Path("/data/gaoya/agent-data/tools/blender-3.6.23-linux-x64/blender")
FFMPEG_DEFAULT = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/ffmpeg")

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

# Keep the same source-to-background assignment as the existing eight-demo
# page, so the new page remains an apples-to-apples visual replacement.
JOBS = (
    ("F2/0717_f2_attempt001614__randpos01", "warehouse_cobalt"),
    ("F2/0717_f2_attempt001096__randpos01", "machine_shop_amber"),
    ("F3/0717_f3_attempt000307__randpos01", "color_studio"),
    ("F2/0717_f2_attempt001614__randpos01", "glasshouse_mint"),
    ("F2/0717_f2_attempt001096__randpos01", "courtyard_terracotta"),
    ("F3/0717_f3_attempt000307__randpos01", "foundry_safety"),
    ("F2/0717_f2_attempt001614__randpos01", "garage_teal"),
    ("F2/0717_f2_attempt001096__randpos01", "neon_studio"),
)

# These are downloaded PBR sets already present in the local asset pack.  An
# override is allowed to cross material categories because the final display
# hue is assigned separately by the Eevee renderer.
TEXTURE_POOL = (
    "hessian_230",
    "rusty_metal_03",
    "oak_wood_planks",
    "denim_fabric_03",
    "fabric_leather_01",
    "dark_wood",
    "painted_metal_shutter",
    "rubberized_track",
)


def _run(command: list[str], *, env: dict[str, str] | None = None, log=None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env, stdout=log, stderr=subprocess.STDOUT if log else None)


def _encode_video(ffmpeg: Path, frames_dir: Path, target: Path, fps: int, *, log=None) -> None:
    temporary = target.with_suffix(".tmp.mp4")
    _run(
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
        ],
        log=log,
    )
    temporary.replace(target)


def _load_source_case(source_root: Path, relative_case: str) -> tuple[dict[str, Any], dict[str, Any]]:
    case_root = source_root / "cases" / relative_case
    manifest_path = case_root / "case_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    meta_path = Path(str(manifest["meta"]))
    return manifest, json.loads(meta_path.read_text(encoding="utf-8"))


def _shape_volume(shape: str, size: dict[str, float]) -> float:
    if shape == "sphere":
        return 4.0 / 3.0 * math.pi * size["radius"] ** 3
    if shape in {"box", "rounded_box", "wedge"}:
        return 8.0 * size["hx"] * size["hy"] * size["hz"]
    if shape in {"cylinder", "puck", "wheel_thick", "spool"}:
        radius = size.get("radius", size.get("flange_radius", size.get("core_radius", 0.0)))
        height = size.get("height", size.get("width", 0.0))
        return math.pi * radius**2 * height
    if shape == "capsule":
        return math.pi * size["radius"] ** 2 * size["height"] + 4.0 / 3.0 * math.pi * size["radius"] ** 3
    if shape == "ellipsoid":
        return 4.0 / 3.0 * math.pi * size["rx"] * size["ry"] * size["rz"]
    if shape == "cone_frustum":
        return math.pi * size["height"] * (
            size["r_base"] ** 2 + size["r_base"] * size["r_top"] + size["r_top"] ** 2
        ) / 3.0
    # The current F2/F3 source set does not use these shapes, but keeping a
    # positive fallback makes the generator fail gracefully on a future demo.
    return max(float(np.prod(list(size.values()))), 1e-9)


def _scale_object_size(obj, rng: np.random.Generator) -> tuple[dict[str, float], dict[str, float]]:
    """Sample isotropic plus anisotropic geometry scales for one actor."""

    shape = obj.shape
    old = {str(key): float(value) for key, value in obj.size.items()}
    global_scale = float(rng.uniform(0.76, 1.28))
    axis_scales = rng.uniform(0.86, 1.18, size=3)
    new = dict(old)
    if shape == "sphere":
        new["radius"] *= global_scale
    elif shape in {"box", "rounded_box", "wedge"}:
        for key, axis_index in (("hx", 0), ("hy", 1), ("hz", 2)):
            if key in new:
                new[key] *= global_scale * float(axis_scales[axis_index])
        if "corner_radius" in new:
            new["corner_radius"] = min(
                new["corner_radius"] * global_scale,
                0.42 * min(new.get("hx", 1.0), new.get("hy", 1.0), new.get("hz", 1.0)),
            )
    elif shape in {"cylinder", "puck", "wheel_thick", "spool"}:
        for key in ("radius", "core_radius", "flange_radius", "r_base", "r_top"):
            if key in new:
                new[key] *= global_scale * float(axis_scales[0])
        for key in ("height", "width", "flange_width"):
            if key in new:
                new[key] *= global_scale * float(axis_scales[2])
    else:
        for key in list(new):
            new[key] *= global_scale
    return new, {
        "global": global_scale,
        "x": float(axis_scales[0]),
        "y": float(axis_scales[1]),
        "z": float(axis_scales[2]),
    }


def _resolve_spawn_position_against_static(
    obj,
    position: list[float],
    static_objects: list[Any],
) -> tuple[list[float], list[float]]:
    """Move a resized dynamic actor out of legacy fixed geometry at spawn."""
    original_xy = np.asarray(position[:2], dtype=np.float64)
    resolved_xy = original_xy.copy()
    actor_radius = _horizontal_clearance_radius(obj)
    for _ in range(4):
        changed = False
        for static in static_objects:
            delta = resolved_xy - np.asarray(static.position[:2], dtype=np.float64)
            distance = float(np.linalg.norm(delta))
            required = actor_radius + _horizontal_clearance_radius(static) + 0.012
            if distance >= required:
                continue
            direction = delta / distance if distance > 1e-8 else np.asarray((1.0, 0.0))
            resolved_xy = np.asarray(static.position[:2], dtype=np.float64) + direction * required
            changed = True
        if not changed:
            break
    resolved = [float(resolved_xy[0]), float(resolved_xy[1]), float(position[2])]
    adjustment = [float(resolved_xy[0] - original_xy[0]), float(resolved_xy[1] - original_xy[1])]
    return resolved, adjustment


def _diversify_blueprint(
    blueprint,
    *,
    source_case_id: str,
    job_index: int,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    objects = []
    geometry_records: list[dict[str, Any]] = []
    texture_assets: dict[str, str] = {}
    static_objects = [obj for obj in blueprint.objects if not obj.dynamic]
    for object_index, obj in enumerate(blueprint.objects):
        # The original 0717 F1-F10 metadata predates the stricter 0819
        # initialization contract.  Static supports are still valid fixed
        # scene geometry for these legacy families; use the explicit role
        # understood by the current validator while leaving their collision
        # geometry unchanged.
        if not obj.dynamic:
            appearance_group = f"diverse_{job_index:02d}_{object_index:02d}"
            texture_assets[obj.name] = TEXTURE_POOL[(job_index * 3 + object_index * 5) % len(TEXTURE_POOL)]
            objects.append(
                replace(
                    obj,
                    role="anchored_occluder",
                    metadata={
                        **obj.metadata,
                        "appearance_group": appearance_group,
                        "diversity_variant": True,
                        "source_case_id": source_case_id,
                        "job_index": job_index,
                        "static_geometry_preserved": True,
                    },
                )
            )
            geometry_records.append(
                {
                    "object": obj.name,
                    "shape": obj.shape,
                    "original_size": obj.size,
                    "new_size": obj.size,
                    "scale": {"global": 1.0, "x": 1.0, "y": 1.0, "z": 1.0},
                    "original_mass": float(obj.mass),
                    "new_mass_constant_density": float(obj.mass),
                    "static_geometry_preserved": True,
                }
            )
            continue
        new_size, scale_record = _scale_object_size(obj, rng)
        old_volume = _shape_volume(obj.shape, obj.size)
        new_volume = _shape_volume(obj.shape, new_size)
        provisional = replace(obj, size=new_size)
        old_extent = _collision_vertical_extent(obj)
        new_extent = _collision_vertical_extent(provisional)
        new_position = list(obj.position)
        # Keep the changed shape grounded at the same physical contact plane.
        new_position[2] += new_extent - old_extent
        # A subset of legacy F1-F10 objects has a conservative source extent
        # that is slightly below the current plane-audit threshold.  Raising
        # only those floor-contact actors preserves the visible setup while
        # preventing the stricter reconstruction preflight from rejecting the
        # derived case.
        if new_position[2] < new_extent + 0.020:
            new_position[2] = new_extent + 0.020
        new_position, spawn_xy_adjustment = _resolve_spawn_position_against_static(
            provisional,
            new_position,
            static_objects,
        )
        # Constant density makes the geometry change physically meaningful;
        # friction/restitution/damping remain unchanged.
        new_mass = float(obj.mass * new_volume / max(old_volume, 1e-9))
        angular_velocity_scale = 0.75 if obj.shape == "wheel_thick" else 1.0
        new_angular_velocity = tuple(
            float(value * angular_velocity_scale) for value in obj.angular_velocity
        )
        appearance_group = f"diverse_{job_index:02d}_{object_index:02d}"
        texture_assets[obj.name] = TEXTURE_POOL[(job_index * 3 + object_index * 5) % len(TEXTURE_POOL)]
        objects.append(
            replace(
                provisional,
                mass=new_mass,
                position=tuple(float(value) for value in new_position),
                angular_velocity=new_angular_velocity,
                metadata={
                    **obj.metadata,
                    "appearance_group": appearance_group,
                    "diversity_variant": True,
                    "geometry_scale": scale_record,
                    "source_case_id": source_case_id,
                    "job_index": job_index,
                },
            )
        )
        geometry_records.append(
            {
                "object": obj.name,
                "shape": obj.shape,
                "original_size": obj.size,
                "new_size": new_size,
                "scale": scale_record,
                "original_mass": float(obj.mass),
                "new_mass_constant_density": new_mass,
                "original_ground_extent_m": old_extent,
                "new_ground_extent_m": new_extent,
                "spawn_xy_adjustment_m": spawn_xy_adjustment,
                "angular_velocity_scale": angular_velocity_scale,
            }
        )

    appearance = {
        "background_variant": int(job_index + 1),
        "palette_offset": int((job_index * 2 + len(objects)) % 4),
        "texture_offset": int((job_index * 3 + len(objects)) % len(TEXTURE_POOL)),
        "texture_assets": texture_assets,
    }
    variant = replace(
        blueprint,
        sample_key=f"{source_case_id}__diverse{job_index:02d}",
        objects=tuple(objects),
        metadata={
            **blueprint.metadata,
            "diversity_variant": True,
            "source_case_id": source_case_id,
            "job_index": int(job_index),
            "variant_seed": int(seed),
            "changed_variables": [
                "true_geometry_size",
                "mass_constant_density",
                "background_tint_and_texture_mapping",
                "object_texture_asset",
                "object_display_palette",
            ],
            "appearance_variation": appearance,
            "geometry_records": geometry_records,
        },
    )
    return variant, {
        "variant_seed": int(seed),
        "geometry_records": geometry_records,
        "appearance": appearance,
    }


def _stage_and_render_job(
    *,
    job_index: int,
    relative_case: str,
    background_profile: str,
    source_root: Path,
    output_root: Path,
    blender: Path,
    ffmpeg: Path,
    gpu: str,
    exposure: float,
    samples: int,
    camera_distance_scale: float,
    stage_width: int,
    stage_height: int,
    seed: int,
    worker_index: int,
) -> dict[str, Any]:
    source_manifest, source_meta = _load_source_case(source_root, relative_case)
    source_case_id = str(source_manifest["sample_key"])
    blueprint = _blueprint_from_meta(source_meta, source_case_id)
    variant_seed = int(seed + job_index * 1009)
    variant, diversity = _diversify_blueprint(
        blueprint,
        source_case_id=source_case_id,
        job_index=job_index,
        seed=variant_seed,
    )
    job_key = f"{variant.sample_key}__{background_profile}"
    job_work = output_root / "_work" / f"worker_{worker_index:02d}" / job_key
    stage_root = job_work / "physics_staging"
    frames_dir = job_work / "frames"
    stage_root.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "logs" / f"worker_{worker_index:02d}__job_{job_index:02d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        staged_manifest = render_blueprint_states_only(
            blueprint=variant,
            seed=variant_seed,
            output_root=stage_root,
        )
        staged_meta_path = Path(str(staged_manifest["meta"]))
        staged_states_path = Path(str(staged_manifest["states"]))
        staged_meta = json.loads(staged_meta_path.read_text(encoding="utf-8"))
        staged_meta["appearance_variation"] = diversity["appearance"]
        staged_meta["appearance_texture_assets"] = diversity["appearance"]["texture_assets"]
        staged_meta["appearance_palette_offset"] = diversity["appearance"]["palette_offset"]
        staged_meta["appearance_texture_offset"] = diversity["appearance"]["texture_offset"]
        staged_meta["background_variant"] = diversity["appearance"]["background_variant"]
        staged_meta["diversity"] = diversity
        staged_meta_path.write_text(json.dumps(staged_meta, ensure_ascii=False, indent=2), encoding="utf-8")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["EGL_DEVICE_ID"] = "0"
        _run(
            [
                str(blender),
                "-b",
                "--python",
                str(Path(__file__).with_name("render_texture_realism_eevee.py")),
                "--",
                "--meta",
                str(staged_meta_path),
                "--states",
                str(staged_states_path),
                "--output-dir",
                str(frames_dir),
                "--width",
                "1280",
                "--height",
                "720",
                "--samples",
                str(samples),
                "--exposure",
                str(exposure),
                "--camera-distance-scale",
                str(camera_distance_scale),
                "--frame-limit",
                "0",
                "--background-profile",
                background_profile,
            ],
            env=env,
            log=log,
        )
        report = json.loads((frames_dir / "render_metadata.json").read_text(encoding="utf-8"))
        frame_count = int(report["frame_count"])
        frame_paths = sorted(frames_dir.glob("frame_*.png"))
        if len(frame_paths) != frame_count:
            raise RuntimeError(f"{job_key}: expected {frame_count} frames, got {len(frame_paths)}")

        final_case = output_root / "cases" / str(source_manifest["family_key"]) / job_key
        video_dir = final_case / "videos"
        meta_dir = final_case / "meta"
        video_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / f"{job_key}__eevee_pbr.mp4"
        _encode_video(ffmpeg, frames_dir, video_path, int(staged_meta.get("fps", 30)), log=log)

        final_states = meta_dir / f"{job_key}_states.npz"
        shutil.copy2(staged_states_path, final_states)
        report["state_source"] = str(final_states)
        final_meta = dict(staged_meta)
        final_meta.update(
            {
                "render_variant": "high_diversity_background_pbr_fast_eevee",
                "render_engine": "BLENDER_EEVEE",
                "cycles_used": False,
                "background_profile": background_profile,
                "source_case_key": source_case_id,
                "source_video": str(source_manifest["video"]),
                "source_meta": str(source_manifest["meta"]),
                "render_metadata": report,
                "output_video": str(video_path),
                "state_source": str(final_states),
                "diversity": diversity,
            }
        )
        final_meta_path = meta_dir / f"{job_key}__eevee_pbr.json"
        final_meta_path.write_text(json.dumps(final_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    shutil.rmtree(job_work, ignore_errors=True)
    return {
        "sample_key": f"{job_key}__eevee_pbr",
        "family_key": source_manifest["family_key"],
        "seed": variant_seed,
        "output_root": str(final_case),
        "video": str(video_path),
        "meta": str(final_meta_path),
        "source_case_id": source_case_id,
        "source_video": str(source_manifest["video"]),
        "source_meta": str(source_manifest["meta"]),
        "background_profile": background_profile,
        "gpu": str(gpu),
        "worker_index": int(worker_index),
        "diversity": diversity,
    }


def _write_worker_record(output_root: Path, worker_index: int, jobs: list[dict[str, Any]], errors: list[dict[str, Any]]) -> None:
    record_dir = output_root / "_worker_jobs"
    record_dir.mkdir(parents=True, exist_ok=True)
    (record_dir / f"worker_{worker_index:02d}.json").write_text(
        json.dumps({"worker_index": worker_index, "jobs": jobs, "errors": errors}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def finalize_output(output_root: Path, source_root: Path) -> None:
    records = []
    errors = []
    for path in sorted((output_root / "_worker_jobs").glob("worker_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.extend(payload.get("jobs", []))
        errors.extend(payload.get("errors", []))
    records.sort(key=lambda item: str(item.get("sample_key", "")))
    (output_root / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "README.json").write_text(
        json.dumps(
            {
                "purpose": "high-diversity F2/F3 demo renders",
                "source_root": str(source_root),
                "num_generated": len(records),
                "errors": errors,
                "changed_variables": [
                    "true_geometry_size",
                    "mass_constant_density",
                    "object_display_palette",
                    "object_pbr_texture_asset_and_mapping",
                    "background_tint",
                    "background_pbr_texture_mapping",
                    "background_hdri_rotation",
                ],
                "unchanged_physics_coefficients": [
                    "gravity",
                    "friction",
                    "restitution",
                    "linear_damping",
                    "angular_damping",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"finalized={len(records)} errors={len(errors)} output_root={output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_DEFAULT)
    parser.add_argument("--blender", type=Path, default=BLENDER_DEFAULT)
    parser.add_argument("--ffmpeg", type=Path, default=FFMPEG_DEFAULT)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--only-job", type=int)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--exposure", type=float, default=-0.15)
    parser.add_argument("--camera-distance-scale", type=float, default=1.0)
    parser.add_argument("--stage-width", type=int, default=320)
    parser.add_argument("--stage-height", type=int, default=180)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.finalize:
        finalize_output(output_root, source_root)
        return
    if args.worker_count <= 0 or not 0 <= args.worker_index < args.worker_count:
        raise ValueError("worker-index must be in [0, worker-count)")

    selected = list(enumerate(JOBS))
    if args.only_job is not None:
        selected = [(index, job) for index, job in selected if index == args.only_job]
    else:
        selected = [(index, job) for index, job in selected if index % args.worker_count == args.worker_index]
    print(f"worker={args.worker_index}/{args.worker_count} gpu={args.gpu} jobs={[index for index, _ in selected]}", flush=True)
    completed: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for job_index, (relative_case, background_profile) in selected:
        print(f"START job={job_index} source={relative_case} background={background_profile} gpu={args.gpu}", flush=True)
        try:
            item = _stage_and_render_job(
                job_index=job_index,
                relative_case=relative_case,
                background_profile=background_profile,
                source_root=source_root,
                output_root=output_root,
                blender=args.blender.resolve(),
                ffmpeg=args.ffmpeg.resolve(),
                gpu=str(args.gpu),
                exposure=float(args.exposure),
                samples=int(args.samples),
                camera_distance_scale=float(args.camera_distance_scale),
                stage_width=int(args.stage_width),
                stage_height=int(args.stage_height),
                seed=int(args.seed),
                worker_index=int(args.worker_index),
            )
            completed.append(item)
            print(f"DONE job={job_index} video={item['video']}", flush=True)
        except Exception as exc:  # keep other independent jobs running
            error = {"job_index": job_index, "error": repr(exc), "traceback": traceback.format_exc()}
            errors.append(error)
            print(f"ERROR job={job_index}: {exc}", flush=True)
    _write_worker_record(output_root, int(args.worker_index), completed, errors)
    print(f"worker_done={args.worker_index} completed={len(completed)} errors={len(errors)}", flush=True)


if __name__ == "__main__":
    main()
