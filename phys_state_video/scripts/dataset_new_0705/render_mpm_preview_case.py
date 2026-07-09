#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import site
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np


def _sanitize_user_site_for_genesis() -> None:
    user_site = site.getusersitepackages()
    if not user_site:
        return
    user_site = os.path.abspath(user_site)
    sys.path[:] = [p for p in sys.path if os.path.abspath(p) != user_site]


_sanitize_user_site_for_genesis()

import genesis as gs


DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/agent-data/outputs/mpm_preview")
DEFAULT_FPS = 20


@dataclass(frozen=True)
class CameraSpec:
    res: tuple[int, int]
    pos: tuple[float, float, float]
    lookat: tuple[float, float, float]
    fov: float


@dataclass(frozen=True)
class SimSpec:
    dt: float
    substeps: int
    horizon: int
    gravity: tuple[float, float, float]
    mpm_lower_bound: tuple[float, float, float]
    mpm_upper_bound: tuple[float, float, float]
    grid_density: int


@dataclass(frozen=True)
class CaseSpec:
    key: str
    title: str
    description: str
    sim: SimSpec
    camera: CameraSpec
    preview_stride: int
    mpm_vis_mode: str
    floor_color: tuple[float, float, float, float]
    soft_color: tuple[float, float, float, float]
    rigid_color: tuple[float, float, float, float]


_GS_INITIALIZED = False


CASE_LIBRARY: dict[str, CaseSpec] = {
    "elastic_drop": CaseSpec(
        key="elastic_drop",
        title="Elastic block drop",
        description="A soft elastic block drops onto the floor and deforms on impact.",
        sim=SimSpec(
            dt=4e-3,
            substeps=12,
            horizon=180,
            gravity=(0.0, 0.0, -9.81),
            mpm_lower_bound=(-0.9, -0.9, -0.1),
            mpm_upper_bound=(0.9, 0.9, 1.6),
            grid_density=96,
        ),
        camera=CameraSpec(
            res=(960, 540),
            pos=(2.4, -2.0, 1.35),
            lookat=(0.0, 0.0, 0.35),
            fov=38.0,
        ),
        preview_stride=1,
        mpm_vis_mode="visual",
        floor_color=(0.70, 0.68, 0.64, 1.0),
        soft_color=(0.80, 0.47, 0.36, 1.0),
        rigid_color=(0.24, 0.30, 0.37, 1.0),
    ),
    "sphere_impact_soft_block": CaseSpec(
        key="sphere_impact_soft_block",
        title="Rigid sphere impact on soft block",
        description="A rigid sphere flies into a soft block resting near the floor to validate rigid-MPM coupling.",
        sim=SimSpec(
            dt=4e-3,
            substeps=15,
            horizon=220,
            gravity=(0.0, 0.0, -9.81),
            mpm_lower_bound=(-1.0, -1.0, -0.1),
            mpm_upper_bound=(1.0, 1.0, 1.8),
            grid_density=128,
        ),
        camera=CameraSpec(
            res=(960, 540),
            pos=(2.8, -2.3, 1.45),
            lookat=(0.15, 0.0, 0.34),
            fov=35.0,
        ),
        preview_stride=1,
        mpm_vis_mode="visual",
        floor_color=(0.69, 0.67, 0.63, 1.0),
        soft_color=(0.76, 0.44, 0.34, 1.0),
        rigid_color=(0.23, 0.34, 0.48, 1.0),
    ),
}


def _init_genesis() -> None:
    global _GS_INITIALIZED
    if _GS_INITIALIZED:
        return
    gs.init(backend=gs.gpu)
    _GS_INITIALIZED = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a minimal Genesis MPM preview case.")
    parser.add_argument(
        "--case-key",
        default="sphere_impact_soft_block",
        choices=sorted(CASE_LIBRARY.keys()),
        help="Which preview case to simulate.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="", help="Optional explicit output folder name.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--width", type=int, default=0, help="Optional override for camera width.")
    parser.add_argument("--height", type=int, default=0, help="Optional override for camera height.")
    parser.add_argument(
        "--mpm-vis-mode",
        choices=["visual", "particle"],
        default="",
        help="Override how MPM bodies are rendered.",
    )
    parser.add_argument(
        "--save-every-frame",
        action="store_true",
        help="If set, write all RGB frames to disk instead of only preview keyframes.",
    )
    return parser.parse_args()


def _camera_spec(case: CaseSpec, width: int, height: int) -> CameraSpec:
    if width > 0 and height > 0:
        return CameraSpec(
            res=(int(width), int(height)),
            pos=case.camera.pos,
            lookat=case.camera.lookat,
            fov=case.camera.fov,
        )
    return case.camera


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _build_output_dirs(case_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": case_dir,
        "rgb": case_dir / "rgb",
        "video": case_dir / "video",
        "debug": case_dir / "debug",
    }
    for path in dirs.values():
        _ensure_dir(path)
    return dirs


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _record_frame_indices(num_steps: int) -> list[int]:
    if num_steps <= 1:
        return [0]
    mid = num_steps // 2
    return sorted({0, mid, num_steps - 1})


def _render_rgb(camera: Any) -> np.ndarray:
    rgb = camera.render(rgb=True, depth=False, segmentation=False, normal=False)
    if isinstance(rgb, tuple):
        rgb = rgb[0]
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _soft_block_state(block: Any) -> dict[str, Any]:
    pts = _to_numpy(block.get_particles_pos())
    pts = pts.reshape(-1, 3)
    centroid = pts.mean(axis=0)
    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    bbox_size = bbox_max - bbox_min
    return {
        "n_particles": int(pts.shape[0]),
        "centroid": centroid.tolist(),
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "bbox_size": bbox_size.tolist(),
    }


def _rigid_body_state(body: Any) -> dict[str, Any]:
    pos = _to_numpy(body.get_pos()).reshape(-1)[:3]
    quat = _to_numpy(body.get_quat()).reshape(-1)[:4]
    vel = _to_numpy(body.get_vel()).reshape(-1)[:3]
    ang = _to_numpy(body.get_ang()).reshape(-1)[:3]
    return {
        "pos": pos.tolist(),
        "quat": quat.tolist(),
        "vel": vel.tolist(),
        "ang": ang.tolist(),
    }


def _scene_common(case: CaseSpec, camera: CameraSpec) -> tuple[Any, Any]:
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=case.sim.dt,
            substeps=case.sim.substeps,
            gravity=case.sim.gravity,
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=case.sim.mpm_lower_bound,
            upper_bound=case.sim.mpm_upper_bound,
            grid_density=case.sim.grid_density,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=camera.pos,
            camera_lookat=camera.lookat,
            camera_fov=camera.fov,
            res=camera.res,
            max_FPS=max(DEFAULT_FPS, 60),
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
            visualize_mpm_boundary=False,
            ambient_light=(0.20, 0.20, 0.20),
            plane_reflection=True,
        ),
        renderer=gs.renderers.Rasterizer(),
        show_viewer=False,
    )
    cam = scene.add_camera(
        res=camera.res,
        pos=camera.pos,
        lookat=camera.lookat,
        fov=camera.fov,
        GUI=False,
    )
    return scene, cam


def _build_case_entities(scene: Any, case: CaseSpec, mpm_vis_mode: str) -> dict[str, Any]:
    scene.add_entity(
        morph=gs.morphs.Plane(),
        surface=gs.surfaces.Default(color=case.floor_color),
    )

    if case.key == "elastic_drop":
        soft_block = scene.add_entity(
            material=gs.materials.MPM.Elastic(
                E=5.0e4,
                nu=0.25,
                rho=240.0,
                sampler="pbs-16",
                model="neohooken",
            ),
            morph=gs.morphs.Box(
                pos=(0.0, 0.0, 0.58),
                size=(0.28, 0.28, 0.28),
                euler=(12.0, 0.0, 18.0),
            ),
            surface=gs.surfaces.Default(
                color=case.soft_color,
                vis_mode=mpm_vis_mode,
            ),
        )
        return {"soft_block": soft_block}

    if case.key == "sphere_impact_soft_block":
        soft_block = scene.add_entity(
            material=gs.materials.MPM.Elastic(
                E=4.5e4,
                nu=0.24,
                rho=220.0,
                sampler="pbs-16",
                model="neohooken",
            ),
            morph=gs.morphs.Box(
                pos=(0.15, 0.0, 0.20),
                size=(0.32, 0.24, 0.24),
                euler=(0.0, 0.0, 10.0),
            ),
            surface=gs.surfaces.Default(
                color=case.soft_color,
                vis_mode=mpm_vis_mode,
            ),
        )
        sphere = scene.add_entity(
            material=gs.materials.Rigid(rho=900.0, friction=0.45, coup_friction=0.9),
            morph=gs.morphs.Sphere(
                pos=(-0.60, 0.0, 0.46),
                radius=0.10,
            ),
            surface=gs.surfaces.Default(color=case.rigid_color),
        )
        return {"soft_block": soft_block, "rigid_sphere": sphere}

    raise ValueError(f"unsupported case key: {case.key}")


def _apply_case_initial_conditions(case_key: str, entities: dict[str, Any]) -> None:
    if case_key == "elastic_drop":
        return
    if case_key == "sphere_impact_soft_block":
        sphere = entities["rigid_sphere"]
        sphere.set_dofs_velocity((2.4, 0.0, -0.15, 0.0, 6.0, 0.0))
        return
    raise ValueError(f"unsupported case key: {case_key}")


def render_case(
    *,
    case_key: str,
    output_root: Path,
    run_name: str,
    fps: int,
    width: int,
    height: int,
    mpm_vis_mode_override: str,
    save_every_frame: bool,
) -> dict[str, Any]:
    _init_genesis()
    case = CASE_LIBRARY[case_key]
    camera = _camera_spec(case, width=width, height=height)
    mpm_vis_mode = mpm_vis_mode_override or case.mpm_vis_mode
    run_label = run_name or case_key
    case_dir = output_root / run_label
    dirs = _build_output_dirs(case_dir)

    scene, cam = _scene_common(case, camera)
    entities = _build_case_entities(scene, case, mpm_vis_mode)
    scene.build()
    _apply_case_initial_conditions(case.key, entities)

    preview_frames: list[np.ndarray] = []
    saved_keyframes: list[str] = []
    record_steps = _record_frame_indices(case.sim.horizon)

    initial_state = {
        "soft_block": _soft_block_state(entities["soft_block"]),
    }
    if "rigid_sphere" in entities:
        initial_state["rigid_sphere"] = _rigid_body_state(entities["rigid_sphere"])

    for step_idx in range(case.sim.horizon):
        scene.step()
        rgb = _render_rgb(cam)
        if save_every_frame:
            frame_path = dirs["rgb"] / f"{step_idx:06d}.png"
            imageio.imwrite(frame_path, rgb)
        elif step_idx in record_steps:
            frame_path = dirs["rgb"] / f"{step_idx:06d}.png"
            imageio.imwrite(frame_path, rgb)
            saved_keyframes.append(frame_path.name)
        preview_frames.append(rgb)

    video_path = dirs["video"] / "preview.mp4"
    imageio.mimwrite(video_path, preview_frames, fps=fps, quality=8)

    final_state = {
        "soft_block": _soft_block_state(entities["soft_block"]),
    }
    if "rigid_sphere" in entities:
        final_state["rigid_sphere"] = _rigid_body_state(entities["rigid_sphere"])

    manifest = {
        "case_key": case.key,
        "title": case.title,
        "description": case.description,
        "output_root": str(case_dir),
        "video": str(video_path),
        "keyframes": saved_keyframes,
        "fps": int(fps),
        "num_frames": int(len(preview_frames)),
        "mpm_vis_mode": mpm_vis_mode,
        "camera": asdict(camera),
        "sim": asdict(case.sim),
        "initial_state": initial_state,
        "final_state": final_state,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "genesis_backend": "gpu",
        },
        "notes": [
            "This is a minimal MPM preview intended for physics and rendering validation before dataset-scale integration.",
            "Genesis in the current environment warns about torch 2.7.x being outside its supported range; preview success does not fully de-risk larger runs.",
        ],
    }
    manifest_path = dirs["root"] / "manifest.json"
    manifest_path.write_text(json.dumps(_to_jsonable(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    manifest = render_case(
        case_key=args.case_key,
        output_root=args.output_root,
        run_name=args.run_name,
        fps=args.fps,
        width=args.width,
        height=args.height,
        mpm_vis_mode_override=args.mpm_vis_mode,
        save_every_frame=args.save_every_frame,
    )
    print(json.dumps(_to_jsonable(manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
