from __future__ import annotations

from pathlib import Path

from .common_specs import LightingSpec, MaterialSpec, RangeSpec, SurfaceThemeSpec


TEXTURE_ROOT = Path("/data/gaoya/dataset/blender_render_assets/polyhaven_v1/textures")
HDRI_ROOT = Path("/data/gaoya/dataset/blender_render_assets/polyhaven_v1/hdris")
LEGACY_TEXTURE_ROOT = Path("/data/gaoya/dataset/textures/polyhaven_wood")


def build_material_catalog() -> dict[str, MaterialSpec]:
    materials = [
        MaterialSpec(
            key="rubber_red",
            category="rubber",
            texture_style="rubber",
            base_color=(0.82, 0.26, 0.18),
            accent_color=(0.93, 0.79, 0.73),
            roughness=0.86,
            notes="Primary high-contrast dynamic object material.",
        ),
        MaterialSpec(
            key="rubber_blue",
            category="rubber",
            texture_style="rubber",
            base_color=(0.20, 0.38, 0.72),
            accent_color=(0.80, 0.87, 0.94),
            roughness=0.84,
        ),
        MaterialSpec(
            key="painted_metal_teal",
            category="painted_metal",
            texture_style="painted",
            base_color=(0.20, 0.58, 0.62),
            accent_color=(0.90, 0.92, 0.92),
            roughness=0.44,
            metallic=0.35,
            notes="Works for cans, cylinders and support props.",
        ),
        MaterialSpec(
            key="painted_metal_yellow",
            category="painted_metal",
            texture_style="painted",
            base_color=(0.85, 0.66, 0.22),
            accent_color=(0.96, 0.95, 0.90),
            roughness=0.40,
            metallic=0.30,
        ),
        MaterialSpec(
            key="plastic_orange",
            category="plastic",
            texture_style="plastic",
            base_color=(0.87, 0.45, 0.22),
            accent_color=(0.99, 0.92, 0.84),
            roughness=0.56,
        ),
        MaterialSpec(
            key="plastic_white",
            category="plastic",
            texture_style="plastic",
            base_color=(0.86, 0.86, 0.84),
            accent_color=(0.98, 0.98, 0.97),
            roughness=0.42,
        ),
        MaterialSpec(
            key="wood_plywood",
            category="wood",
            texture_style="wood_real",
            texture_asset="plywood",
            base_color=(0.75, 0.60, 0.42),
            accent_color=(0.88, 0.82, 0.71),
            roughness=0.82,
            texture_path=str(LEGACY_TEXTURE_ROOT / "plywood_diff_4k.jpg"),
        ),
        MaterialSpec(
            key="wood_dark",
            category="wood",
            texture_style="wood_real",
            texture_asset="dark_wood",
            base_color=(0.42, 0.28, 0.16),
            accent_color=(0.68, 0.56, 0.42),
            roughness=0.78,
            texture_path=str(LEGACY_TEXTURE_ROOT / "dark_wood_diff_4k.jpg"),
        ),
        MaterialSpec(
            key="cardboard_kraft",
            category="cardboard",
            texture_style="painted",
            base_color=(0.68, 0.54, 0.37),
            accent_color=(0.84, 0.76, 0.62),
            roughness=0.88,
            notes="Approximation until a dedicated cardboard texture is added.",
        ),
        MaterialSpec(
            key="leather_brown",
            category="leather",
            texture_style="wood_real",
            base_color=(0.47, 0.28, 0.16),
            accent_color=(0.71, 0.56, 0.42),
            roughness=0.73,
            texture_path=str(TEXTURE_ROOT / "brown_leather" / "brown_leather_albedo_2k.jpg"),
            normal_path=str(TEXTURE_ROOT / "brown_leather" / "brown_leather_nor_gl_2k.jpg"),
            roughness_path=str(TEXTURE_ROOT / "brown_leather" / "brown_leather_rough_2k.jpg"),
            ao_path=str(TEXTURE_ROOT / "brown_leather" / "brown_leather_ao_2k.jpg"),
        ),
        MaterialSpec(
            key="concrete_painted",
            category="concrete",
            texture_style="wood_real",
            base_color=(0.64, 0.63, 0.62),
            accent_color=(0.78, 0.77, 0.75),
            roughness=0.93,
            texture_path=str(TEXTURE_ROOT / "painted_concrete" / "painted_concrete_diff_2k.jpg"),
            normal_path=str(TEXTURE_ROOT / "painted_concrete" / "painted_concrete_nor_gl_2k.jpg"),
            roughness_path=str(TEXTURE_ROOT / "painted_concrete" / "painted_concrete_rough_2k.jpg"),
            ao_path=str(TEXTURE_ROOT / "painted_concrete" / "painted_concrete_ao_2k.jpg"),
        ),
        MaterialSpec(
            key="floor_wood",
            category="floor",
            texture_style="wood_real",
            texture_asset="wood_floor",
            base_color=(0.61, 0.48, 0.34),
            accent_color=(0.86, 0.79, 0.68),
            roughness=0.78,
            texture_path=str(TEXTURE_ROOT / "wood_floor" / "wood_floor_diff_2k.jpg"),
            normal_path=str(TEXTURE_ROOT / "wood_floor" / "wood_floor_nor_gl_2k.jpg"),
            roughness_path=str(TEXTURE_ROOT / "wood_floor" / "wood_floor_rough_2k.jpg"),
            ao_path=str(TEXTURE_ROOT / "wood_floor" / "wood_floor_ao_2k.jpg"),
        ),
        MaterialSpec(
            key="wall_beige",
            category="wall",
            texture_style="wood_real",
            base_color=(0.81, 0.78, 0.70),
            accent_color=(0.92, 0.90, 0.86),
            roughness=0.90,
            texture_path=str(TEXTURE_ROOT / "beige_wall_001" / "beige_wall_001_diff_2k.jpg"),
            normal_path=str(TEXTURE_ROOT / "beige_wall_001" / "beige_wall_001_nor_gl_2k.jpg"),
            roughness_path=str(TEXTURE_ROOT / "beige_wall_001" / "beige_wall_001_rough_2k.jpg"),
            ao_path=str(TEXTURE_ROOT / "beige_wall_001" / "beige_wall_001_ao_2k.jpg"),
        ),
    ]
    return {material.key: material for material in materials}


def build_lighting_catalog() -> dict[str, LightingSpec]:
    lights = [
        LightingSpec(
            hdri_key="studio_soft",
            key_light_intensity=1.20,
            fill_light_intensity=0.82,
            rim_light_intensity=0.50,
            shadow_strength=0.35,
            ambient_boost=0.04,
        ),
        LightingSpec(
            hdri_key="hall_neutral",
            key_light_intensity=1.05,
            fill_light_intensity=0.72,
            rim_light_intensity=0.34,
            shadow_strength=0.46,
            ambient_boost=0.02,
        ),
        LightingSpec(
            hdri_key="studio_warm",
            key_light_intensity=1.12,
            fill_light_intensity=0.70,
            rim_light_intensity=0.58,
            shadow_strength=0.40,
            ambient_boost=0.03,
        ),
    ]
    return {light.hdri_key: light for light in lights}


def build_hdri_catalog() -> dict[str, str]:
    return {
        "studio_soft": str(HDRI_ROOT / "poly_haven_studio" / "poly_haven_studio_4k.hdr"),
        "studio_warm": str(HDRI_ROOT / "brown_photostudio_02" / "brown_photostudio_02_4k.hdr"),
        "hall_neutral": str(HDRI_ROOT / "old_hall" / "old_hall_4k.hdr"),
    }


def build_surface_catalog() -> dict[str, SurfaceThemeSpec]:
    surfaces = [
        SurfaceThemeSpec(
            key="studio_wood_floor",
            floor_material_key="floor_wood",
            wall_material_key="wall_beige",
            floor_friction_range=RangeSpec(0.58, 0.86),
            background_mode="studio",
            notes="Default clean training setup with higher visual realism.",
        ),
        SurfaceThemeSpec(
            key="painted_concrete_floor",
            floor_material_key="concrete_painted",
            wall_material_key="wall_beige",
            floor_friction_range=RangeSpec(0.52, 0.78),
            background_mode="industrial",
        ),
        SurfaceThemeSpec(
            key="dark_wood_floor",
            floor_material_key="wood_dark",
            wall_material_key="wall_beige",
            floor_friction_range=RangeSpec(0.62, 0.92),
            background_mode="warm_studio",
        ),
    ]
    return {surface.key: surface for surface in surfaces}

