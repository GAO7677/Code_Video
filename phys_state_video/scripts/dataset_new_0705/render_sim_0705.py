from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import numpy as np

import generate_sim_preview_gallery as legacy

from .common_specs import CameraSpec, MaterialSpec, ScenarioBlueprint
from .material_catalog_0705 import (
    build_hdri_catalog,
    build_lighting_catalog,
    build_material_catalog,
    build_surface_catalog,
)
from .scene_generators_0705 import generate_scenario_blueprint


def _legacy_texture_asset(material: MaterialSpec) -> str:
    if material.texture_asset:
        return material.texture_asset
    if material.texture_path:
        return material.key
    return ""


def register_material_assets() -> None:
    for material in build_material_catalog().values():
        asset_key = _legacy_texture_asset(material)
        if not asset_key or not material.texture_path:
            continue
        legacy.WOOD_TEXTURES[asset_key] = Path(material.texture_path)


def _legacy_object_spec_from_blueprint_object(obj) -> legacy.ObjectSpec:
    materials = build_material_catalog()
    material = materials[obj.material_key]
    return legacy.ObjectSpec(
        name=obj.name,
        shape=obj.shape,
        color=list(material.base_color),
        mass=float(obj.mass),
        position=list(obj.position),
        size=dict(obj.size),
        dynamic=bool(obj.dynamic),
        restitution=float(obj.restitution),
        friction=float(obj.friction),
        linear_damping=float(obj.linear_damping),
        angular_damping=float(obj.angular_damping),
        orientation_euler_deg=list(obj.orientation_euler_deg),
        linear_velocity=list(obj.linear_velocity),
        angular_velocity=list(obj.angular_velocity),
        role=obj.role,
        texture_style=material.texture_style,
        texture_asset=_legacy_texture_asset(material),
    )


def blueprint_to_legacy_scenario(blueprint: ScenarioBlueprint, seed: int) -> legacy.ScenarioSpec:
    surfaces = build_surface_catalog()
    surface = surfaces[blueprint.surface_key]
    family_label = f"{blueprint.family_key} v2"
    objects = [_legacy_object_spec_from_blueprint_object(obj) for obj in blueprint.objects]
    floor_mu = float(np.clip(surface.floor_friction_range.midpoint(), 0.05, 1.20))
    return legacy.ScenarioSpec(
        key=blueprint.sample_key,
        family=family_label,
        title=blueprint.title,
        description=blueprint.description,
        gravity=blueprint.gravity,
        floor_friction=floor_mu,
        objects=objects,
        seed=seed,
        pre_roll_s=blueprint.pre_roll_s,
        sim_type="rigid_realism_v2",
    )


def _style_bg(surface_key: str) -> list[float]:
    surfaces = build_surface_catalog()
    background_mode = surfaces[surface_key].background_mode
    if background_mode == "industrial":
        return [0.56, 0.56, 0.58]
    if background_mode == "warm_studio":
        return [0.54, 0.50, 0.46]
    return [0.58, 0.56, 0.53]


def _paint_mesh_from_material(mesh, material: MaterialSpec) -> None:
    temp_obj = legacy.ObjectSpec(
        name=f"material_probe_{material.key}",
        shape="box",
        color=list(material.base_color),
        mass=0.0,
        position=[0.0, 0.0, 0.0],
        size={"hx": 0.5, "hy": 0.5, "hz": 0.5},
        dynamic=False,
        texture_style=material.texture_style,
        texture_asset=_legacy_texture_asset(material),
    )
    legacy._apply_procedural_material(mesh, temp_obj)


class RealismPreviewRenderer:
    def __init__(
        self,
        *,
        camera: CameraSpec,
        surface_key: str,
        lighting_key: str,
        width: int,
        height: int,
    ) -> None:
        materials = build_material_catalog()
        surfaces = build_surface_catalog()
        lightings = build_lighting_catalog()
        surface = surfaces[surface_key]
        lighting = lightings[lighting_key]

        self.scene = legacy.pyrender.Scene(
            bg_color=_style_bg(surface_key),
            ambient_light=[0.08 + lighting.ambient_boost, 0.08 + lighting.ambient_boost, 0.08 + lighting.ambient_boost],
        )

        floor = legacy.trimesh.creation.box(extents=[14.0, 12.0, 0.04])
        _paint_mesh_from_material(floor, materials[surface.floor_material_key])
        self.scene.add(legacy.pyrender.Mesh.from_trimesh(floor, smooth=False), pose=legacy._tr(0.0, 2.0, -0.04))

        wall = legacy.trimesh.creation.box(extents=[14.0, 0.04, 3.6])
        _paint_mesh_from_material(wall, materials[surface.wall_material_key])
        self.scene.add(legacy.pyrender.Mesh.from_trimesh(wall, smooth=False), pose=legacy._tr(0.0, 5.0, 1.8))

        left_fill = legacy.pyrender.SpotLight(
            color=[1.0, 0.97, 0.93],
            intensity=58.0 * lighting.key_light_intensity,
            innerConeAngle=0.42,
            outerConeAngle=1.0,
        )
        right_fill = legacy.pyrender.SpotLight(
            color=[0.94, 0.97, 1.0],
            intensity=36.0 * lighting.fill_light_intensity,
            innerConeAngle=0.45,
            outerConeAngle=1.0,
        )
        rim = legacy.pyrender.SpotLight(
            color=[1.0, 1.0, 1.0],
            intensity=28.0 * lighting.rim_light_intensity,
            innerConeAngle=0.35,
            outerConeAngle=0.95,
        )
        self.scene.add(
            left_fill,
            pose=legacy._look_at(np.array([-1.9, -1.6, 2.7]), np.array([0.25, 0.35, 0.35]), np.array([0.0, 0.0, 1.0])),
        )
        self.scene.add(
            right_fill,
            pose=legacy._look_at(np.array([2.1, -0.9, 2.25]), np.array([-0.1, 0.8, 0.45]), np.array([0.0, 0.0, 1.0])),
        )
        self.scene.add(
            rim,
            pose=legacy._look_at(np.array([0.0, 2.8, 2.4]), np.array([0.0, 0.4, 0.3]), np.array([0.0, 0.0, 1.0])),
        )

        camera_node = legacy.pyrender.PerspectiveCamera(
            yfov=np.radians(camera.yfov_deg),
            aspectRatio=width / height,
        )
        self.scene.add(
            camera_node,
            pose=legacy._look_at(
                np.asarray(camera.eye, dtype=np.float64),
                np.asarray(camera.target, dtype=np.float64),
                np.asarray(camera.up, dtype=np.float64),
            ),
        )
        self.renderer = legacy.pyrender.OffscreenRenderer(width, height)
        self.nodes: dict[str, legacy.pyrender.Node] = {}
        self.shadow_strength = lighting.shadow_strength

    def add_object(self, obj: legacy.ObjectSpec) -> None:
        mesh = legacy._make_mesh(obj)
        node = self.scene.add(
            legacy.pyrender.Mesh.from_trimesh(mesh, smooth=True),
            pose=legacy._tr(obj.position[0], obj.position[1], obj.position[2]),
        )
        self.nodes[obj.name] = node

    def update_pose(self, name: str, pos: list[float], quat: list[float]) -> None:
        self.scene.set_pose(self.nodes[name], pose=legacy._pb_pose(pos, quat))

    def render(self):
        flags = legacy.RenderFlags.SHADOWS_SPOT if self.shadow_strength > 0.1 else 0
        color, _ = self.renderer.render(self.scene, flags=flags)
        return color

    def cleanup(self) -> None:
        self.renderer.delete()


@contextmanager
def override_legacy_runtime(
    *,
    output_root: Path,
    camera: CameraSpec,
    width: int,
    height: int,
) -> Iterator[None]:
    old_state = {
        "OUTPUT_ROOT": legacy.OUTPUT_ROOT,
        "VIDEO_DIR": legacy.VIDEO_DIR,
        "META_DIR": legacy.META_DIR,
        "CAM_EYE": legacy.CAM_EYE.copy(),
        "CAM_TARGET": legacy.CAM_TARGET.copy(),
        "CAM_UP": legacy.CAM_UP.copy(),
        "IMG_W": legacy.IMG_W,
        "IMG_H": legacy.IMG_H,
    }
    try:
        legacy.OUTPUT_ROOT = output_root
        legacy.VIDEO_DIR = output_root / "videos"
        legacy.META_DIR = output_root / "meta"
        legacy.VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        legacy.META_DIR.mkdir(parents=True, exist_ok=True)
        legacy.CAM_EYE = np.asarray(camera.eye, dtype=np.float64)
        legacy.CAM_TARGET = np.asarray(camera.target, dtype=np.float64)
        legacy.CAM_UP = np.asarray(camera.up, dtype=np.float64)
        legacy.IMG_W = int(width)
        legacy.IMG_H = int(height)
        yield
    finally:
        legacy.OUTPUT_ROOT = old_state["OUTPUT_ROOT"]
        legacy.VIDEO_DIR = old_state["VIDEO_DIR"]
        legacy.META_DIR = old_state["META_DIR"]
        legacy.CAM_EYE = old_state["CAM_EYE"]
        legacy.CAM_TARGET = old_state["CAM_TARGET"]
        legacy.CAM_UP = old_state["CAM_UP"]
        legacy.IMG_W = old_state["IMG_W"]
        legacy.IMG_H = old_state["IMG_H"]


def render_blueprint_case(
    *,
    blueprint: ScenarioBlueprint,
    seed: int,
    output_root: Path,
    width: int = 1280,
    height: int = 720,
) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    register_material_assets()
    scenario = blueprint_to_legacy_scenario(blueprint, seed=seed)

    with override_legacy_runtime(output_root=output_root, camera=blueprint.camera, width=width, height=height):
        legacy.p.connect(legacy.p.DIRECT)
        try:
            renderer = RealismPreviewRenderer(
                camera=blueprint.camera,
                surface_key=blueprint.surface_key,
                lighting_key=blueprint.lighting_key,
                width=width,
                height=height,
            )
            try:
                meta = legacy.run_scenario(renderer, scenario, overlay_text=False)
            finally:
                renderer.cleanup()
        finally:
            legacy.p.disconnect()

    meta_path = output_root / "meta" / f"{scenario.key}.json"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    payload["camera"]["yfov_deg"] = blueprint.camera.yfov_deg
    payload["camera"]["eye"] = list(blueprint.camera.eye)
    payload["camera"]["target"] = list(blueprint.camera.target)
    payload["camera"]["up"] = list(blueprint.camera.up)
    payload["surface_key"] = blueprint.surface_key
    payload["lighting_key"] = blueprint.lighting_key
    payload["tags"] = list(blueprint.tags)
    payload["blueprint"] = {
        "family_key": blueprint.family_key,
        "sample_key": blueprint.sample_key,
        "surface_key": blueprint.surface_key,
        "lighting_key": blueprint.lighting_key,
        "camera_key": blueprint.camera_key,
        "metadata": blueprint.metadata,
    }
    payload["materials"] = {
        obj.name: obj.material_key for obj in blueprint.objects
    }
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "sample_key": blueprint.sample_key,
        "family_key": blueprint.family_key,
        "seed": seed,
        "output_root": str(output_root),
        "video": str(output_root / "videos" / f"{scenario.key}.mp4"),
        "meta": str(meta_path),
        "states": str(output_root / "meta" / f"{scenario.key}_states.npz"),
        "width": width,
        "height": height,
        "hdri_catalog": build_hdri_catalog(),
    }
    (output_root / "case_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def render_generated_case(
    *,
    family_key: str,
    sample_key: str,
    seed: int,
    output_root: Path,
    width: int = 1280,
    height: int = 720,
) -> dict:
    blueprint = generate_scenario_blueprint(family_key=family_key, sample_key=sample_key, seed=seed)
    return render_blueprint_case(
        blueprint=blueprint,
        seed=seed,
        output_root=output_root,
        width=width,
        height=height,
    )
