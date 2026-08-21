from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class RangeSpec:
    low: float
    high: float

    def midpoint(self) -> float:
        return 0.5 * (self.low + self.high)


@dataclass(frozen=True)
class MaterialSpec:
    key: str
    category: str
    texture_style: str
    texture_asset: str = ""
    base_color: Vec3 = (0.7, 0.7, 0.7)
    accent_color: Vec3 = (0.9, 0.9, 0.9)
    roughness: float = 0.5
    metallic: float = 0.0
    normal_strength: float = 1.0
    texture_path: str = ""
    normal_path: str = ""
    roughness_path: str = ""
    ao_path: str = ""
    texture_repeat_range: tuple[float, float] = (1.0, 1.0)
    texture_rotation_deg_range: tuple[float, float] = (0.0, 0.0)
    tone_jitter_range: tuple[float, float] = (0.0, 0.0)
    mix_variation_range: tuple[float, float] = (0.0, 0.0)
    notes: str = ""


@dataclass(frozen=True)
class CameraSpec:
    eye: Vec3
    target: Vec3
    up: Vec3 = (0.0, 0.0, 1.0)
    yfov_deg: float = 50.0
    jitter_eye_xyz: Vec3 = (0.0, 0.0, 0.0)
    jitter_target_xyz: Vec3 = (0.0, 0.0, 0.0)
    jitter_fov_deg: float = 0.0
    hdri_key: str = ""
    exposure_range: RangeSpec = field(default_factory=lambda: RangeSpec(0.95, 1.05))


@dataclass(frozen=True)
class ObjectFamilySpec:
    key: str
    display_name: str
    shape: str
    semantic_role: str
    dynamic_default: bool
    size_ranges: dict[str, RangeSpec]
    mass_range: RangeSpec
    friction_range: RangeSpec
    restitution_range: RangeSpec
    linear_damping_range: RangeSpec
    angular_damping_range: RangeSpec
    allowed_material_categories: tuple[str, ...]
    orientation_jitter_deg: Vec3 = (0.0, 0.0, 0.0)
    velocity_scale_range: RangeSpec = field(default_factory=lambda: RangeSpec(0.9, 1.1))
    angular_velocity_scale_range: RangeSpec = field(default_factory=lambda: RangeSpec(0.9, 1.1))
    notes: str = ""


@dataclass(frozen=True)
class ObjectInstanceSpec:
    name: str
    family_key: str
    shape: str
    semantic_role: str
    size: dict[str, float]
    mass: float
    friction: float
    restitution: float
    linear_damping: float
    angular_damping: float
    material_key: str
    color: Vec3
    dynamic: bool = True
    role: str = "dynamic"
    position: Vec3 = (0.0, 0.0, 0.0)
    orientation_euler_deg: Vec3 = (0.0, 0.0, 0.0)
    linear_velocity: Vec3 = (0.0, 0.0, 0.0)
    angular_velocity: Vec3 = (0.0, 0.0, 0.0)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_legacy_object_kwargs(self, material: MaterialSpec) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": self.shape,
            "color": list(self.color),
            "mass": float(self.mass),
            "position": list(self.position),
            "size": dict(self.size),
            "dynamic": bool(self.dynamic),
            "restitution": float(self.restitution),
            "friction": float(self.friction),
            "linear_damping": float(self.linear_damping),
            "angular_damping": float(self.angular_damping),
            "orientation_euler_deg": list(self.orientation_euler_deg),
            "linear_velocity": list(self.linear_velocity),
            "angular_velocity": list(self.angular_velocity),
            "role": self.role,
            "texture_style": material.texture_style,
            "texture_asset": material.texture_asset,
        }


@dataclass(frozen=True)
class LightingSpec:
    hdri_key: str
    key_light_intensity: float
    fill_light_intensity: float
    rim_light_intensity: float
    shadow_strength: float
    ambient_boost: float = 0.0


@dataclass(frozen=True)
class SurfaceThemeSpec:
    key: str
    floor_material_key: str
    wall_material_key: str
    floor_friction_range: RangeSpec
    floor_color_bias: Vec3 = (1.0, 1.0, 1.0)
    background_mode: str = "studio"
    notes: str = ""


@dataclass(frozen=True)
class ScenarioFamilySpec:
    key: str
    title: str
    description: str
    family_slug: str
    min_dynamic_objects: int
    max_dynamic_objects: int
    min_total_objects: int
    max_total_objects: int
    supports_occlusion: bool
    supports_support_objects: bool
    target_event_types: tuple[str, ...]
    preferred_surface_keys: tuple[str, ...]
    preferred_camera_keys: tuple[str, ...]
    motion_modes: tuple[str, ...] = ()
    speed_range: tuple[float, float] = (0.0, 0.0)
    spin_range: tuple[float, float] = (0.0, 0.0)
    angle_range_deg: tuple[float, float] = (0.0, 0.0)
    notes: str = ""


@dataclass(frozen=True)
class ScenarioBlueprint:
    family_key: str
    sample_key: str
    title: str
    description: str
    gravity: float
    pre_roll_s: float
    camera_key: str
    surface_key: str
    lighting_key: str
    camera: CameraSpec
    objects: tuple[ObjectInstanceSpec, ...]
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetLayoutSpec:
    raw_output_root: Path
    episode_output_root: Path
    preview_output_root: Path
    qa_output_root: Path
