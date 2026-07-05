from __future__ import annotations

import json
import math
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
    build_indoor_asset_pack_manifest,
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


def _load_image(path: str) -> np.ndarray | None:
    if not path:
        return None
    src = Path(path)
    if not src.exists():
        return None
    image = legacy.cv2.imread(str(src), legacy.cv2.IMREAD_COLOR)
    if image is None:
        return None
    return legacy.cv2.cvtColor(image, legacy.cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _sample_texture_array(material: MaterialSpec, rng: np.random.Generator | None = None) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, float, float]:
    if rng is None:
        rng = np.random.default_rng(0)
    albedo = _load_image(material.texture_path)
    normal = _load_image(material.normal_path)
    roughness = _load_image(material.roughness_path)
    repeat = float(rng.uniform(*material.texture_repeat_range))
    rotation_deg = float(rng.uniform(*material.texture_rotation_deg_range))
    return albedo, normal, roughness, repeat, rotation_deg


def _sample_map_bilinear(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    if image is None or len(uv) == 0:
        return np.zeros((len(uv), 3), dtype=np.float32)
    h, w = image.shape[:2]
    uv = uv - np.floor(uv)
    x = np.clip(uv[:, 0] * (w - 1), 0.0, w - 1.0)
    y = np.clip((1.0 - uv[:, 1]) * (h - 1), 0.0, h - 1.0)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = (x - x0).astype(np.float32)[:, None]
    wy = (y - y0).astype(np.float32)[:, None]
    c00 = image[y0, x0]
    c01 = image[y1, x0]
    c10 = image[y0, x1]
    c11 = image[y1, x1]
    return (
        c00 * (1.0 - wx) * (1.0 - wy)
        + c10 * wx * (1.0 - wy)
        + c01 * (1.0 - wx) * wy
        + c11 * wx * wy
    ).astype(np.float32)


def _apply_texture_2d_pattern(
    mesh,
    material: MaterialSpec,
    *,
    seed: int,
    uv_repeat_override: float | None = None,
    tone_bias_override: float | None = None,
) -> None:
    rng = np.random.default_rng(seed)
    albedo, normal_map, rough_map, repeat, rotation_deg = _sample_texture_array(material, rng)
    repeat = uv_repeat_override if uv_repeat_override is not None else repeat
    tone_jitter = float(rng.uniform(*material.tone_jitter_range))
    if tone_bias_override is not None:
        tone_jitter = tone_bias_override
    mix = float(rng.uniform(*material.mix_variation_range))
    angle = math.radians(rotation_deg)
    rot = np.asarray([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=np.float32)

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)
    if len(verts) == 0:
        return

    mins = verts.min(axis=0)
    spans = np.maximum(verts.max(axis=0) - mins, 1e-6)
    uv_xy = np.stack([(verts[:, 0] - mins[0]) / spans[0], (verts[:, 1] - mins[1]) / spans[1]], axis=1)
    uv_xz = np.stack([(verts[:, 0] - mins[0]) / spans[0], (verts[:, 2] - mins[2]) / spans[2]], axis=1)
    uv_yz = np.stack([(verts[:, 1] - mins[1]) / spans[1], (verts[:, 2] - mins[2]) / spans[2]], axis=1)
    uv_xy = (uv_xy @ rot) * repeat
    uv_xz = (uv_xz @ rot) * repeat
    uv_yz = (uv_yz @ rot) * repeat

    base = np.tile(np.asarray(material.base_color, dtype=np.float32), (len(verts), 1))
    colors = base.copy()
    dominant_uv = uv_xy
    if albedo is not None:
        abs_normals = np.abs(normals)
        dominant = np.argmax(abs_normals, axis=1)
        if np.any(dominant == 0):
            colors[dominant == 0] = _sample_map_bilinear(albedo, uv_yz[dominant == 0])
        if np.any(dominant == 1):
            colors[dominant == 1] = _sample_map_bilinear(albedo, uv_xz[dominant == 1])
        if np.any(dominant == 2):
            colors[dominant == 2] = _sample_map_bilinear(albedo, uv_xy[dominant == 2])
    else:
        if material.texture_style == "fabric":
            weave = 0.5 + 0.5 * np.sin(verts[:, 0] * 18.0 + verts[:, 1] * 11.0)
            bands = 0.5 + 0.5 * np.sin(verts[:, 2] * 4.0 + verts[:, 0] * 2.0)
            colors = colors * (0.84 + 0.10 * weave[:, None]) * (0.90 + 0.08 * bands[:, None])
        elif material.texture_style == "metal":
            sheen = 0.45 + 0.55 * (verts[:, 2] - verts[:, 2].min()) / max(float(np.ptp(verts[:, 2])), 1e-6)
            colors = colors * (0.78 + 0.22 * sheen[:, None])
        elif material.texture_style == "painted":
            brush = 0.5 + 0.5 * np.sin(verts[:, 0] * 6.0 + verts[:, 2] * 3.0)
            colors = colors * (0.86 + 0.10 * brush[:, None])
        else:
            bands = 0.5 + 0.5 * np.sin(verts[:, 0] * 3.4 + verts[:, 2] * 2.2)
            colors = colors * (0.86 + 0.12 * bands[:, None])

    if rough_map is not None:
        rough_abs = np.mean(rough_map, axis=2, keepdims=False)
        rough = _sample_map_bilinear(rough_abs[..., None].repeat(3, axis=2), dominant_uv).mean(axis=1)
        colors = colors * (0.85 + 0.15 * (1.0 - rough)[:, None])
    if normal_map is not None:
        abs_normals = np.abs(normals)
        dominant = np.argmax(abs_normals, axis=1)
        if np.any(dominant == 0):
            nmap = _sample_map_bilinear(normal_map, uv_yz[dominant == 0])
            bump = 0.55 + 0.45 * np.clip(nmap[:, 2], 0.0, 1.0)
            colors[dominant == 0] *= bump[:, None]
        if np.any(dominant == 1):
            nmap = _sample_map_bilinear(normal_map, uv_xz[dominant == 1])
            bump = 0.55 + 0.45 * np.clip(nmap[:, 2], 0.0, 1.0)
            colors[dominant == 1] *= bump[:, None]
        if np.any(dominant == 2):
            nmap = _sample_map_bilinear(normal_map, uv_xy[dominant == 2])
            bump = 0.55 + 0.45 * np.clip(nmap[:, 2], 0.0, 1.0)
            colors[dominant == 2] *= bump[:, None]

    low_freq = 0.5 + 0.5 * np.sin(verts[:, 0] * 1.1 + verts[:, 1] * 0.8 + verts[:, 2] * 0.4)
    if tone_jitter > 0.0:
        colors = colors * (1.0 - tone_jitter) + np.asarray(material.accent_color, dtype=np.float32)[None, :] * tone_jitter
    colors = colors * (0.90 + 0.10 * low_freq[:, None])
    colors = colors * (0.96 + 0.04 * mix)
    colors = np.clip(colors, 0.0, 1.0)
    mesh.visual.vertex_colors = colors


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
        return [0.46, 0.47, 0.48]
    if background_mode == "warm_studio":
        return [0.52, 0.49, 0.45]
    return [0.54, 0.53, 0.50]


def _add_mesh_with_material(scene, mesh, material: MaterialSpec, pose, *, seed: int) -> None:
    _apply_texture_2d_pattern(mesh, material, seed=seed)
    scene.add(legacy.pyrender.Mesh.from_trimesh(mesh, smooth=False), pose=pose)


def _build_decor_object(
    *,
    name: str,
    shape: str,
    size: dict[str, float],
    position: tuple[float, float, float],
    material_key: str,
    role: str = "support",
    dynamic: bool = False,
    orientation_euler_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> legacy.ObjectSpec:
    material = build_material_catalog()[material_key]
    return legacy.ObjectSpec(
        name=name,
        shape=shape,
        color=list(material.base_color),
        mass=0.0,
        position=list(position),
        size=dict(size),
        dynamic=dynamic,
        restitution=0.0,
        friction=0.85,
        linear_damping=0.0,
        angular_damping=0.0,
        orientation_euler_deg=list(orientation_euler_deg),
        linear_velocity=[0.0, 0.0, 0.0],
        angular_velocity=[0.0, 0.0, 0.0],
        role=role,
        texture_style=material.texture_style,
        texture_asset=_legacy_texture_asset(material),
    )


def _wall_hanging_curtain(material: MaterialSpec, *, width: float, height: float) -> legacy.trimesh.Trimesh:
    curtain = legacy.trimesh.creation.box(extents=[width, 0.03, height])
    _paint_mesh_from_material(curtain, material)
    curtain.apply_translation([0.0, 0.0, 0.0])
    return curtain


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
        scene_style: str = "indoor_realistic",
    ) -> None:
        materials = build_material_catalog()
        surfaces = build_surface_catalog()
        lightings = build_lighting_catalog()
        surface = surfaces[surface_key]
        lighting = lightings[lighting_key]
        self.scene_style = scene_style

        self.scene = legacy.pyrender.Scene(
            bg_color=_style_bg(surface_key) if scene_style != "indoor_realistic" else [0.09, 0.09, 0.10],
            ambient_light=[0.08 + lighting.ambient_boost, 0.08 + lighting.ambient_boost, 0.08 + lighting.ambient_boost],
        )

        if scene_style == "indoor_realistic":
            self._add_indoor_room(materials, surface, lighting)
        else:
            self._add_simple_stage(materials, surface, lighting)

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

    def _add_simple_stage(self, materials, surface, lighting) -> None:
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

    def _add_indoor_room(self, materials, surface, lighting) -> None:
        rng = np.random.default_rng(314159)
        floor = legacy.trimesh.creation.box(extents=[14.4, 12.8, 0.05])
        _apply_texture_2d_pattern(
            floor,
            materials[surface.floor_material_key],
            seed=int(rng.integers(0, 1_000_000)),
            uv_repeat_override=float(rng.uniform(1.2, 1.9)),
        )
        self.scene.add(legacy.pyrender.Mesh.from_trimesh(floor, smooth=False), pose=legacy._tr(0.0, 2.0, -0.05))

        back_wall = legacy.trimesh.creation.box(extents=[14.4, 0.05, 3.8])
        _apply_texture_2d_pattern(
            back_wall,
            materials[surface.wall_material_key],
            seed=int(rng.integers(0, 1_000_000)),
            uv_repeat_override=float(rng.uniform(0.7, 1.3)),
        )
        self.scene.add(legacy.pyrender.Mesh.from_trimesh(back_wall, smooth=False), pose=legacy._tr(0.0, 5.05, 1.9))

        left_wall = legacy.trimesh.creation.box(extents=[0.05, 12.8, 3.8])
        _apply_texture_2d_pattern(
            left_wall,
            materials["wall_beige"],
            seed=int(rng.integers(0, 1_000_000)),
            uv_repeat_override=float(rng.uniform(0.6, 1.1)),
        )
        self.scene.add(legacy.pyrender.Mesh.from_trimesh(left_wall, smooth=False), pose=legacy._tr(-7.2, 1.95, 1.9))

        right_wall = legacy.trimesh.creation.box(extents=[0.05, 12.8, 3.8])
        _apply_texture_2d_pattern(
            right_wall,
            materials["wall_beige"],
            seed=int(rng.integers(0, 1_000_000)),
            uv_repeat_override=float(rng.uniform(0.6, 1.1)),
        )
        self.scene.add(legacy.pyrender.Mesh.from_trimesh(right_wall, smooth=False), pose=legacy._tr(7.2, 1.95, 1.9))

        ceiling = legacy.trimesh.creation.box(extents=[14.4, 12.8, 0.04])
        _apply_texture_2d_pattern(
            ceiling,
            materials["wall_beige"],
            seed=int(rng.integers(0, 1_000_000)),
            uv_repeat_override=float(rng.uniform(0.5, 0.9)),
        )
        self.scene.add(legacy.pyrender.Mesh.from_trimesh(ceiling, smooth=False), pose=legacy._tr(0.0, 2.0, 3.82))

        baseboard = legacy.trimesh.creation.box(extents=[14.25, 0.04, 0.10])
        _apply_texture_2d_pattern(
            baseboard,
            materials["wood_dark"],
            seed=int(rng.integers(0, 1_000_000)),
            uv_repeat_override=float(rng.uniform(1.5, 2.6)),
        )
        self.scene.add(legacy.pyrender.Mesh.from_trimesh(baseboard, smooth=False), pose=legacy._tr(0.0, 5.00, 0.05))

        side_board_l = legacy.trimesh.creation.box(extents=[0.04, 12.4, 0.10])
        _apply_texture_2d_pattern(
            side_board_l,
            materials["wood_dark"],
            seed=int(rng.integers(0, 1_000_000)),
            uv_repeat_override=float(rng.uniform(1.5, 2.6)),
        )
        self.scene.add(legacy.pyrender.Mesh.from_trimesh(side_board_l, smooth=False), pose=legacy._tr(-7.15, 1.95, 0.05))

        side_board_r = legacy.trimesh.creation.box(extents=[0.04, 12.4, 0.10])
        _apply_texture_2d_pattern(
            side_board_r,
            materials["wood_dark"],
            seed=int(rng.integers(0, 1_000_000)),
            uv_repeat_override=float(rng.uniform(1.5, 2.6)),
        )
        self.scene.add(legacy.pyrender.Mesh.from_trimesh(side_board_r, smooth=False), pose=legacy._tr(7.15, 1.95, 0.05))

        ceiling_beam = legacy.trimesh.creation.box(extents=[14.1, 0.10, 0.08])
        _apply_texture_2d_pattern(
            ceiling_beam,
            materials["wood_dark"],
            seed=int(rng.integers(0, 1_000_000)),
            uv_repeat_override=float(rng.uniform(1.8, 2.8)),
        )
        self.scene.add(legacy.pyrender.Mesh.from_trimesh(ceiling_beam, smooth=False), pose=legacy._tr(0.0, 2.0, 3.74))

        window_frame = legacy.trimesh.creation.box(extents=[2.2, 0.03, 1.4])
        _apply_texture_2d_pattern(
            window_frame,
            materials["wood_dark"],
            seed=int(rng.integers(0, 1_000_000)),
            uv_repeat_override=float(rng.uniform(1.0, 1.8)),
        )
        self.scene.add(legacy.pyrender.Mesh.from_trimesh(window_frame, smooth=False), pose=legacy._tr(4.1, 5.01, 2.15))

        window_glow = legacy.trimesh.creation.box(extents=[1.95, 0.01, 1.16])
        glow_material = build_material_catalog()["plastic_white"]
        _apply_texture_2d_pattern(
            window_glow,
            glow_material,
            seed=int(rng.integers(0, 1_000_000)),
            uv_repeat_override=float(rng.uniform(0.6, 0.9)),
        )
        self.scene.add(legacy.pyrender.Mesh.from_trimesh(window_glow, smooth=False), pose=legacy._tr(4.1, 4.985, 2.15))

        curtain_rail = legacy.trimesh.creation.cylinder(radius=0.02, height=2.55, sections=16)
        _apply_texture_2d_pattern(
            curtain_rail,
            materials["painted_metal_teal"],
            seed=int(rng.integers(0, 1_000_000)),
            uv_repeat_override=float(rng.uniform(1.2, 2.0)),
        )
        self.scene.add(legacy.pyrender.Mesh.from_trimesh(curtain_rail, smooth=False), pose=legacy._tr(4.1, 4.965, 2.78))

        curtain_left = legacy.trimesh.creation.box(extents=[0.85, 0.02, 1.32])
        _apply_texture_2d_pattern(
            curtain_left,
            materials["fabric_curtain"],
            seed=int(rng.integers(0, 1_000_000)),
            uv_repeat_override=float(rng.uniform(2.6, 4.0)),
        )
        self.scene.add(legacy.pyrender.Mesh.from_trimesh(curtain_left, smooth=False), pose=legacy._tr(3.1, 4.965, 1.95))

        curtain_right = legacy.trimesh.creation.box(extents=[0.85, 0.02, 1.32])
        _apply_texture_2d_pattern(
            curtain_right,
            materials["fabric_curtain"],
            seed=int(rng.integers(0, 1_000_000)),
            uv_repeat_override=float(rng.uniform(2.6, 4.0)),
        )
        self.scene.add(legacy.pyrender.Mesh.from_trimesh(curtain_right, smooth=False), pose=legacy._tr(5.1, 4.965, 1.95))

        shelf = legacy.trimesh.creation.box(extents=[1.1, 0.40, 1.0])
        _apply_texture_2d_pattern(
            shelf,
            materials["wood_plywood"],
            seed=int(rng.integers(0, 1_000_000)),
            uv_repeat_override=float(rng.uniform(1.2, 2.0)),
        )
        self.scene.add(legacy.pyrender.Mesh.from_trimesh(shelf, smooth=False), pose=legacy._tr(-5.4, 3.95, 0.50))

        crate = legacy.trimesh.creation.box(extents=[0.70, 0.50, 0.40])
        _apply_texture_2d_pattern(
            crate,
            materials["cardboard_kraft"],
            seed=int(rng.integers(0, 1_000_000)),
            uv_repeat_override=float(rng.uniform(1.0, 1.8)),
        )
        self.scene.add(legacy.pyrender.Mesh.from_trimesh(crate, smooth=False), pose=legacy._tr(-5.05, 4.3, 0.20))

        stool_top = legacy.trimesh.creation.cylinder(radius=0.28, height=0.06, sections=32)
        _apply_texture_2d_pattern(
            stool_top,
            materials["leather_brown"],
            seed=int(rng.integers(0, 1_000_000)),
            uv_repeat_override=float(rng.uniform(1.0, 1.6)),
        )
        self.scene.add(legacy.pyrender.Mesh.from_trimesh(stool_top, smooth=False), pose=legacy._tr(5.0, 3.1, 0.58))

        stool_leg = legacy.trimesh.creation.cylinder(radius=0.05, height=0.55, sections=20)
        _apply_texture_2d_pattern(
            stool_leg,
            materials["painted_metal_teal"],
            seed=int(rng.integers(0, 1_000_000)),
            uv_repeat_override=float(rng.uniform(1.0, 1.8)),
        )
        self.scene.add(legacy.pyrender.Mesh.from_trimesh(stool_leg, smooth=False), pose=legacy._tr(5.0, 3.1, 0.28))

        desk = legacy.trimesh.creation.box(extents=[1.4, 0.75, 0.08])
        _apply_texture_2d_pattern(
            desk,
            materials["wood_dark"],
            seed=int(rng.integers(0, 1_000_000)),
            uv_repeat_override=float(rng.uniform(1.4, 2.2)),
        )
        self.scene.add(legacy.pyrender.Mesh.from_trimesh(desk, smooth=False), pose=legacy._tr(5.2, 2.4, 0.80))

        desk_leg = legacy.trimesh.creation.box(extents=[0.08, 0.08, 0.78])
        _apply_texture_2d_pattern(
            desk_leg,
            materials["concrete_painted"],
            seed=int(rng.integers(0, 1_000_000)),
            uv_repeat_override=float(rng.uniform(1.0, 1.6)),
        )
        self.scene.add(legacy.pyrender.Mesh.from_trimesh(desk_leg, smooth=False), pose=legacy._tr(4.55, 2.0, 0.39))

        pipe_h = legacy.trimesh.creation.cylinder(radius=0.03, height=5.9, sections=18)
        _apply_texture_2d_pattern(
            pipe_h,
            materials["painted_metal_teal"],
            seed=int(rng.integers(0, 1_000_000)),
            uv_repeat_override=float(rng.uniform(1.6, 2.4)),
        )
        self.scene.add(legacy.pyrender.Mesh.from_trimesh(pipe_h, smooth=False), pose=legacy._tr(-6.1, 4.65, 3.1))

        pipe_v = legacy.trimesh.creation.cylinder(radius=0.03, height=2.6, sections=18)
        _apply_texture_2d_pattern(
            pipe_v,
            materials["painted_metal_teal"],
            seed=int(rng.integers(0, 1_000_000)),
            uv_repeat_override=float(rng.uniform(1.6, 2.4)),
        )
        self.scene.add(legacy.pyrender.Mesh.from_trimesh(pipe_v, smooth=False), pose=legacy._tr(6.1, 2.1, 2.0))

        clutter_box = legacy.trimesh.creation.box(extents=[0.28, 0.18, 0.16])
        _apply_texture_2d_pattern(
            clutter_box,
            materials["cardboard_kraft"],
            seed=int(rng.integers(0, 1_000_000)),
            uv_repeat_override=float(rng.uniform(1.0, 2.0)),
        )
        self.scene.add(legacy.pyrender.Mesh.from_trimesh(clutter_box, smooth=False), pose=legacy._tr(5.8, 1.45, 0.08))

        clutter_can = legacy.trimesh.creation.cylinder(radius=0.06, height=0.22, sections=20)
        _apply_texture_2d_pattern(
            clutter_can,
            materials["painted_metal_yellow"],
            seed=int(rng.integers(0, 1_000_000)),
            uv_repeat_override=float(rng.uniform(1.0, 1.8)),
        )
        self.scene.add(legacy.pyrender.Mesh.from_trimesh(clutter_can, smooth=False), pose=legacy._tr(5.55, 1.7, 0.11))

        wall_cabinet = legacy.trimesh.creation.box(extents=[1.4, 0.38, 1.05])
        _apply_texture_2d_pattern(
            wall_cabinet,
            materials["wall_cream"],
            seed=int(rng.integers(0, 1_000_000)),
            uv_repeat_override=float(rng.uniform(1.1, 1.9)),
        )
        self.scene.add(legacy.pyrender.Mesh.from_trimesh(wall_cabinet, smooth=False), pose=legacy._tr(-4.8, 3.05, 0.53))

        cabinet_handle = legacy.trimesh.creation.cylinder(radius=0.012, height=0.40, sections=12)
        _apply_texture_2d_pattern(
            cabinet_handle,
            materials["painted_metal_yellow"],
            seed=int(rng.integers(0, 1_000_000)),
            uv_repeat_override=float(rng.uniform(1.0, 1.8)),
        )
        self.scene.add(legacy.pyrender.Mesh.from_trimesh(cabinet_handle, smooth=False), pose=legacy._tr(-4.10, 3.02, 0.62))

        floor_clutter = legacy.trimesh.creation.box(extents=[0.34, 0.22, 0.08])
        _apply_texture_2d_pattern(
            floor_clutter,
            materials["cardboard_kraft"],
            seed=int(rng.integers(0, 1_000_000)),
            uv_repeat_override=float(rng.uniform(1.4, 2.6)),
        )
        self.scene.add(legacy.pyrender.Mesh.from_trimesh(floor_clutter, smooth=False), pose=legacy._tr(4.35, 1.2, 0.04))

        hanging_pipe = legacy.trimesh.creation.cylinder(radius=0.018, height=3.0, sections=16)
        _apply_texture_2d_pattern(
            hanging_pipe,
            materials["painted_metal_teal"],
            seed=int(rng.integers(0, 1_000_000)),
            uv_repeat_override=float(rng.uniform(1.4, 2.3)),
        )
        self.scene.add(legacy.pyrender.Mesh.from_trimesh(hanging_pipe, smooth=False), pose=legacy._tr(-6.0, 1.2, 2.95))

        warm_key = legacy.pyrender.SpotLight(
            color=[1.0, 0.95, 0.88],
            intensity=56.0 * lighting.key_light_intensity,
            innerConeAngle=0.42,
            outerConeAngle=1.05,
        )
        cool_fill = legacy.pyrender.SpotLight(
            color=[0.88, 0.94, 1.0],
            intensity=28.0 * lighting.fill_light_intensity,
            innerConeAngle=0.48,
            outerConeAngle=1.05,
        )
        window_day = legacy.pyrender.SpotLight(
            color=[0.98, 0.99, 1.0],
            intensity=72.0 * lighting.rim_light_intensity,
            innerConeAngle=0.30,
            outerConeAngle=0.75,
        )
        self.scene.add(
            warm_key,
            pose=legacy._look_at(np.array([-3.4, -2.0, 3.1]), np.array([0.1, 0.7, 0.55]), np.array([0.0, 0.0, 1.0])),
        )
        self.scene.add(
            cool_fill,
            pose=legacy._look_at(np.array([2.8, -1.7, 2.6]), np.array([0.1, 0.6, 0.42]), np.array([0.0, 0.0, 1.0])),
        )
        self.scene.add(
            window_day,
            pose=legacy._look_at(np.array([4.2, 4.7, 2.15]), np.array([3.8, 2.5, 2.0]), np.array([0.0, 0.0, 1.0])),
        )

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
    scene_style: str = "indoor_realistic",
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
                scene_style=scene_style,
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
        "asset_pack": build_indoor_asset_pack_manifest(),
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
    scene_style: str = "indoor_realistic",
) -> dict:
    blueprint = generate_scenario_blueprint(family_key=family_key, sample_key=sample_key, seed=seed)
    return render_blueprint_case(
        blueprint=blueprint,
        seed=seed,
        output_root=output_root,
        width=width,
        height=height,
        scene_style=scene_style,
    )
