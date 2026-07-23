from __future__ import annotations

import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    from .. import generate_sim_preview_gallery as legacy
except ImportError:  # pragma: no cover - keeps direct script execution working.
    import generate_sim_preview_gallery as legacy

from .common_specs import CameraSpec, MaterialSpec, ScenarioBlueprint
from .material_catalog_0705 import (
    build_hdri_catalog,
    build_lighting_catalog,
    build_indoor_asset_pack_manifest,
    build_material_catalog,
    build_surface_catalog,
)
from .object_catalog_0705 import build_object_family_catalog
from .scene_generators_0705 import (
    DEFAULT_CAMERA_DISTANCE_SCALE,
    build_scenario_family_catalog,
    generate_scenario_blueprint,
    validate_blueprint_physics,
)


DEFAULT_NEGATIVE_PROMPT = (
    "cartoon, anime, illustration, fantasy object behavior, impossible motion, floating objects, "
    "penetration, severe deformation, duplicated objects, extra objects, abrupt cut, camera shake, "
    "extreme motion blur, text, watermark, logo, subtitles"
)


MATERIAL_PROMPT_NAMES = {
    "rubber_red": "red rubber",
    "rubber_blue": "blue rubber",
    "painted_metal_teal": "teal painted metal",
    "painted_metal_yellow": "yellow painted metal",
    "plastic_orange": "orange plastic",
    "plastic_white": "white plastic",
    "wood_plywood": "plywood",
    "wood_dark": "dark wood",
    "cardboard_kraft": "kraft cardboard",
    "leather_brown": "brown leather",
    "fabric_curtain": "light fabric",
    "concrete_painted": "painted concrete",
    "concrete_clean_floor": "clean concrete",
    "concrete_clean_wall": "light concrete",
    "floor_wood": "wood floor",
    "floor_wood_weathered": "weathered wood floor",
    "wall_beige": "beige wall",
    "wall_cream": "cream wall",
}


SURFACE_PROMPT_NAMES = {
    "studio_wood_floor": "a studio-like wood floor",
    "residential_wood_floor": "a residential wood floor",
    "dark_wood_floor": "a dark wood floor",
    "painted_concrete_floor": "a painted concrete floor",
}


MOTION_TAG_PROMPT_NAMES = {
    "roll": "rolls across the floor",
    "slide": "slides across the floor",
    "bounce": "bounces and settles",
    "spin": "spins while moving forward",
    "glance": "glances across the scene with a shallow angle",
    "head_on": "moves straight into the target",
    "crossing": "crosses the scene before contact",
    "offset_push": "pushes the target with an offset hit",
    "domino": "triggers a domino-like interaction",
    "push_chain": "pushes through a two-step chain reaction",
    "rolling_chain": "rolls into a two-step chain reaction",
    "offset_chain": "starts an offset chain reaction",
    "left_pass": "passes behind the occluder and reappears",
    "right_pass": "passes behind the occluder and reappears",
    "cross": "crosses through the occlusion zone and reappears",
    "double_pass": "creates two visible occlusion phases before reappearing",
    "drop": "drops after losing support",
    "topple": "topples after support loss",
    "slide_off": "slides off the support and lands",
    "roll_off": "rolls off the support and lands",
    "shallow_slide": "slides down a shallow ramp",
    "steep_slide": "slides down a steep ramp",
    "rollout": "slides down the ramp and rolls out",
    "high_spin": "spins strongly with limited translation",
    "reverse_spin": "shows reverse spin while translating",
    "wobble_spin": "wobbles with strong angular motion",
    "vertical_drop": "drops vertically and rebounds",
    "oblique_drop": "drops at an oblique angle and rebounds",
    "multi_bounce": "bounces multiple times before settling",
    "crowded_slide": "slides into a cluttered local interaction",
    "offset_collision": "creates an offset collision in a cluttered setup",
    "spill": "spills through nearby clutter and support objects",
    "edge_roll": "rolls toward an edge",
    "fall_off": "falls off the edge after approaching it",
    "boundary_slide": "slides near a boundary with fall-off risk",
}


def _legacy_texture_asset(material: MaterialSpec) -> str:
    if material.texture_asset:
        return material.texture_asset
    if material.texture_path:
        return material.key
    return ""


def _human_join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _with_article(phrase: str) -> str:
    if not phrase:
        return phrase
    first = phrase[0].lower()
    article = "an" if first in {"a", "e", "i", "o", "u"} else "a"
    return f"{article} {phrase}"


def _capitalize_sentence(text: str) -> str:
    if not text:
        return text
    return text[0].upper() + text[1:]


def _object_prompt_phrase(obj) -> str:
    material_name = MATERIAL_PROMPT_NAMES.get(obj.material_key, obj.material_key.replace("_", " "))
    family_name = build_object_family_catalog()[obj.family_key].display_name.lower()
    return _with_article(f"{material_name} {family_name}")


def build_object_phrase_bundle(blueprint: ScenarioBlueprint) -> dict[str, object]:
    """Build machine-friendly noun phrase annotations for every scene object."""
    family_catalog = build_object_family_catalog()
    details: list[dict[str, object]] = []
    for obj in blueprint.objects:
        family_name = family_catalog[obj.family_key].display_name.lower()
        material_name = MATERIAL_PROMPT_NAMES.get(obj.material_key, obj.material_key.replace("_", " "))
        phrase = _with_article(f"{material_name} {family_name}")
        details.append(
            {
                "name": obj.name,
                "role": obj.role,
                "dynamic": bool(obj.dynamic),
                "family_key": obj.family_key,
                "object_noun": family_name,
                "material_key": obj.material_key,
                "material_phrase": material_name,
                "object_phrase": phrase,
            }
        )
    return {
        "object_nouns": [str(item["object_noun"]) for item in details],
        "object_phrases": [str(item["object_phrase"]) for item in details],
        "dynamic_object_phrases": [str(item["object_phrase"]) for item in details if item["dynamic"]],
        "static_object_phrases": [str(item["object_phrase"]) for item in details if not item["dynamic"]],
        "object_phrase_details": details,
    }


def _family_event_sentence(blueprint: ScenarioBlueprint) -> str:
    motion_tag = next((tag for tag in reversed(blueprint.tags) if tag in MOTION_TAG_PROMPT_NAMES), "")
    motion_text = MOTION_TAG_PROMPT_NAMES.get(motion_tag, "moves through the scene")
    role_map = {obj.name: obj for obj in blueprint.objects}

    if blueprint.family_key == "F1":
        mover = role_map.get("driver_0", blueprint.objects[0])
        return f"{_object_prompt_phrase(mover)} {motion_text}."
    if blueprint.family_key == "F2":
        driver = role_map.get("driver_0", blueprint.objects[0])
        target = role_map.get("target_0", blueprint.objects[-1])
        return f"{_object_prompt_phrase(driver)} {motion_text} and hits {_object_prompt_phrase(target)}."
    if blueprint.family_key == "F3":
        lead = role_map.get("lead_0", blueprint.objects[0])
        mid = role_map.get("mid_0", blueprint.objects[1])
        tail = role_map.get("tail_0", blueprint.objects[-1])
        return (
            f"{_object_prompt_phrase(lead)} hits {_object_prompt_phrase(mid)}, "
            f"which then drives {_object_prompt_phrase(tail)}."
        )
    if blueprint.family_key == "F4":
        movers = [obj for obj in blueprint.objects if obj.role == "dynamic"]
        occluders = [obj for obj in blueprint.objects if obj.role == "occluder"]
        mover_text = _human_join([_object_prompt_phrase(obj) for obj in movers])
        occ_count = len(occluders)
        return f"{mover_text} {motion_text} behind {occ_count} pillar occluder{'s' if occ_count != 1 else ''}."
    if blueprint.family_key == "F5":
        dynamic = role_map.get("drop_0", blueprint.objects[0])
        return f"{_object_prompt_phrase(dynamic)} {motion_text} after support loss."
    if blueprint.family_key == "F6":
        mover = role_map.get("slider_0", blueprint.objects[0])
        return f"{_object_prompt_phrase(mover)} {motion_text} on a visible ramp."
    if blueprint.family_key == "F7":
        mover = role_map.get("spinner_0", blueprint.objects[0])
        return f"{_object_prompt_phrase(mover)} {motion_text} with rotation dominating translation."
    if blueprint.family_key == "F8":
        mover = role_map.get("bouncer_0", blueprint.objects[0])
        return f"{_object_prompt_phrase(mover)} {motion_text} before coming to rest."
    if blueprint.family_key == "F9":
        mover_a = role_map.get("clutter_a", blueprint.objects[0])
        mover_b = role_map.get("clutter_b", blueprint.objects[1])
        return f"{_object_prompt_phrase(mover_a)} and {_object_prompt_phrase(mover_b)} collide in a cluttered setup."
    if blueprint.family_key == "F10":
        mover = role_map.get("edge_mover", blueprint.objects[0])
        return f"{_object_prompt_phrase(mover)} {motion_text} near a visible edge."
    return "Rigid objects move through a realistic indoor physics scene."


def _build_prompt_bundle(blueprint: ScenarioBlueprint, width: int, height: int) -> dict[str, object]:
    dynamic_objects = [obj for obj in blueprint.objects if obj.dynamic]
    visible_objects = [_object_prompt_phrase(obj) for obj in dynamic_objects[:3]]
    surface_text = SURFACE_PROMPT_NAMES.get(blueprint.surface_key, blueprint.surface_key.replace("_", " "))
    family_catalog = build_scenario_family_catalog()
    family_spec = family_catalog[blueprint.family_key]
    event_sentence = _capitalize_sentence(_family_event_sentence(blueprint))
    caption = event_sentence
    short_caption = event_sentence
    grounding_caption = _human_join(visible_objects) if visible_objects else family_spec.title.lower()
    phrase_bundle = build_object_phrase_bundle(blueprint)
    return {
        "caption": caption,
        "short_caption": short_caption,
        "grounding_caption": grounding_caption,
        "input_caption": caption,
        **phrase_bundle,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "prompt_metadata": {
            "version": "dataset_new_0705_prompt_v2",
            "family_title": family_spec.title,
            "family_description": family_spec.description,
            "surface_text": surface_text,
            "dynamic_object_count": len(dynamic_objects),
            "total_object_count": len(blueprint.objects),
            "tags": list(blueprint.tags),
            "resolution": [width, height],
        },
    }


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
    validate_blueprint_physics(blueprint)
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
        capture_instance_masks: bool = False,
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
        self.capture_instance_masks = capture_instance_masks
        self.mask_frames: list[np.ndarray] = []
        self.instance_ids: dict[str, int] = {}
        self._seg_node_map: dict[legacy.pyrender.Node, tuple[int, int, int]] = {}
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
        instance_id = len(self.instance_ids) + 1
        if instance_id > 255:
            raise ValueError("instance mask export supports at most 255 objects")
        self.instance_ids[obj.name] = instance_id
        self._seg_node_map[node] = (instance_id, 0, 0)

    def update_pose(self, name: str, pos: list[float], quat: list[float]) -> None:
        self.scene.set_pose(self.nodes[name], pose=legacy._pb_pose(pos, quat))

    def render(self):
        flags = legacy.RenderFlags.SHADOWS_SPOT if self.shadow_strength > 0.1 else 0
        color, _ = self.renderer.render(self.scene, flags=flags)
        if self.capture_instance_masks:
            segmentation, _ = self.renderer.render(
                self.scene,
                flags=legacy.RenderFlags.SEG,
                seg_node_map=self._seg_node_map,
            )
            self.mask_frames.append(np.asarray(segmentation[..., 0], dtype=np.uint8))
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


def _write_instance_mask_outputs(
    *,
    output_root: Path,
    sample_key: str,
    mask_frames: list[np.ndarray],
    instance_ids: dict[str, int],
) -> None:
    if not mask_frames:
        raise RuntimeError(f"no instance masks were captured for {sample_key}")

    mask_dir = output_root / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_ids = np.stack(mask_frames).astype(np.uint8, copy=False)
    np.savez_compressed(
        mask_dir / f"{sample_key}_instance_ids.npz",
        instance_ids=mask_ids,
        object_names=np.asarray(list(instance_ids), dtype=np.str_),
        object_ids=np.asarray(list(instance_ids.values()), dtype=np.uint8),
    )

    # BGR colors are used here because the shared video writer consumes OpenCV frames.
    palette_bgr = np.asarray(
        [
            [0, 0, 0],
            [76, 76, 230],
            [204, 153, 51],
            [102, 204, 76],
        ],
        dtype=np.uint8,
    )
    preview_frames = [palette_bgr[np.minimum(frame, len(palette_bgr) - 1)] for frame in mask_ids]
    legacy._write_video_h264(mask_dir / f"{sample_key}_instance_mask.mp4", preview_frames)


def render_blueprint_case(
    *,
    blueprint: ScenarioBlueprint,
    seed: int,
    output_root: Path,
    width: int = 1280,
    height: int = 720,
    scene_style: str = "indoor_realistic",
    export_instance_masks: bool = False,
    preserve_states: bool = False,
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
                capture_instance_masks=export_instance_masks,
            )
            try:
                meta = legacy.run_scenario(renderer, scenario, overlay_text=False)
                if export_instance_masks:
                    _write_instance_mask_outputs(
                        output_root=output_root,
                        sample_key=scenario.key,
                        mask_frames=renderer.mask_frames,
                        instance_ids=renderer.instance_ids,
                    )
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
    payload["size_scale"] = float(blueprint.metadata.get("size_scale", 1.0))
    payload["camera_distance_scale"] = float(blueprint.metadata.get("camera_distance_scale", 1.0))
    if not preserve_states:
        payload.pop("states", None)
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
    if export_instance_masks:
        payload["instance_masks"] = {
            "format": "uint8_instance_id",
            "background_id": 0,
            "instance_id_map": {obj.name: index + 1 for index, obj in enumerate(blueprint.objects)},
            "ids": str(output_root / "masks" / f"{scenario.key}_instance_ids.npz"),
            "preview_video": str(output_root / "masks" / f"{scenario.key}_instance_mask.mp4"),
        }
    payload.update(_build_prompt_bundle(blueprint, width=width, height=height))
    phrase_by_name = {
        str(item["name"]): item for item in payload.get("object_phrase_details", [])
        if isinstance(item, dict)
    }
    for obj_payload in payload.get("objects", []):
        if not isinstance(obj_payload, dict):
            continue
        phrase_detail = phrase_by_name.get(str(obj_payload.get("name", "")))
        if not phrase_detail:
            continue
        obj_payload["family_key"] = phrase_detail["family_key"]
        obj_payload["object_noun"] = phrase_detail["object_noun"]
        obj_payload["material_key"] = phrase_detail["material_key"]
        obj_payload["material_phrase"] = phrase_detail["material_phrase"]
        obj_payload["object_phrase"] = phrase_detail["object_phrase"]
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    object_phrase_path = output_root / "meta" / f"{scenario.key}_object_phrases.json"
    object_phrase_payload = {
        "case_id": blueprint.sample_key,
        "family_key": blueprint.family_key,
        "object_nouns": payload["object_nouns"],
        "object_phrases": payload["object_phrases"],
        "dynamic_object_phrases": payload["dynamic_object_phrases"],
        "static_object_phrases": payload["static_object_phrases"],
        "object_phrase_details": payload["object_phrase_details"],
    }
    object_phrase_path.write_text(json.dumps(object_phrase_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    states_path = output_root / "meta" / f"{scenario.key}_states.npz"
    if not preserve_states:
        states_path.unlink(missing_ok=True)

    manifest = {
        "sample_key": blueprint.sample_key,
        "family_key": blueprint.family_key,
        "seed": seed,
        "output_root": str(output_root),
        "video": str(output_root / "videos" / f"{scenario.key}.mp4"),
        "mask_video": (
            str(output_root / "masks" / f"{scenario.key}_instance_mask.mp4")
            if export_instance_masks
            else None
        ),
        "mask_ids": (
            str(output_root / "masks" / f"{scenario.key}_instance_ids.npz")
            if export_instance_masks
            else None
        ),
        "instance_id_map": (
            {obj.name: index + 1 for index, obj in enumerate(blueprint.objects)}
            if export_instance_masks
            else {}
        ),
        "states": str(states_path) if preserve_states else None,
        "meta": str(meta_path),
        "object_phrases_path": str(object_phrase_path),
        "width": width,
        "height": height,
        "size_scale": float(blueprint.metadata.get("size_scale", 1.0)),
        "camera_distance_scale": float(blueprint.metadata.get("camera_distance_scale", 1.0)),
        "caption": payload["caption"],
        "short_caption": payload["short_caption"],
        "object_nouns": payload["object_nouns"],
        "object_phrases": payload["object_phrases"],
        "dynamic_object_phrases": payload["dynamic_object_phrases"],
        "static_object_phrases": payload["static_object_phrases"],
        "object_phrase_details": payload["object_phrase_details"],
        "negative_prompt": payload["negative_prompt"],
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
    direction_mode: str = "auto",
    size_scale: float = 1.0,
    camera_distance_scale: float = DEFAULT_CAMERA_DISTANCE_SCALE,
) -> dict:
    blueprint = generate_scenario_blueprint(
        family_key=family_key,
        sample_key=sample_key,
        seed=seed,
        direction_mode=direction_mode,
        size_scale=size_scale,
        camera_distance_scale=camera_distance_scale,
    )
    return render_blueprint_case(
        blueprint=blueprint,
        seed=seed,
        output_root=output_root,
        width=width,
        height=height,
        scene_style=scene_style,
    )
