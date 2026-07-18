#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .common_specs import ObjectInstanceSpec, RangeSpec, ScenarioBlueprint, SurfaceThemeSpec
from .material_catalog_0705 import build_material_catalog
from .object_catalog_0705 import build_object_family_catalog
from .scene_generators_0705 import EARTH_GRAVITY, build_camera_catalog, validate_blueprint_physics
from . import render_sim_0705 as render_sim


DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0718ToyDataset")
ATTRIBUTE_CHOICES = ("background_color", "object_color", "object_shape")
DEFAULT_CASE_KEY = "ball_throw_to_wood_block"
BASE_BUILD_SURFACE_CATALOG = render_sim.build_surface_catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate toy paired rigid-body samples where each pair changes exactly one visual attribute."
        )
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--case-key", default=DEFAULT_CASE_KEY)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--attributes",
        nargs="*",
        choices=ATTRIBUTE_CHOICES,
        default=(),
        help="Attributes to render as anchor/variant pairs. If omitted, only the base case is rendered.",
    )
    parser.add_argument(
        "--scene-style",
        default="toy_simple",
        help="Any non-indoor_realistic style uses the clean stage renderer for simpler toy samples.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete any existing output directory for a case before re-rendering.",
    )
    return parser.parse_args()


def _material_color(material_key: str) -> tuple[float, float, float]:
    material = build_material_catalog()[material_key]
    return tuple(float(channel) for channel in material.base_color)


def _driver_object(
    *,
    family_key: str,
    material_key: str,
) -> ObjectInstanceSpec:
    family_catalog = build_object_family_catalog()
    family = family_catalog[family_key]
    if family_key == "ball":
        size = {"radius": 0.16}
        position = (-1.65, 0.0, 0.70)
        orientation = (0.0, 0.0, 0.0)
        linear_velocity = (4.40, 0.0, 0.30)
        angular_velocity = (0.0, 9.0, 0.0)
        mass = 1.05
        friction = 0.54
        restitution = 0.76
    elif family_key == "capsule_can":
        size = {"radius": 0.10, "height": 0.28}
        position = (-1.65, 0.0, 0.72)
        orientation = (0.0, 90.0, 8.0)
        linear_velocity = (4.40, 0.0, 0.30)
        angular_velocity = (0.0, 5.5, 0.0)
        mass = 0.92
        friction = 0.58
        restitution = 0.14
    else:
        raise ValueError(f"unsupported driver family for toy case: {family_key}")

    return ObjectInstanceSpec(
        name="driver_0",
        family_key=family.key,
        shape=family.shape,
        semantic_role=family.semantic_role,
        size=size,
        mass=mass,
        friction=friction,
        restitution=restitution,
        linear_damping=0.02,
        angular_damping=0.02,
        material_key=material_key,
        color=_material_color(material_key),
        dynamic=True,
        role="dynamic",
        position=position,
        orientation_euler_deg=orientation,
        linear_velocity=linear_velocity,
        angular_velocity=angular_velocity,
        metadata={"toy_role": "driver"},
    )


def _target_object() -> ObjectInstanceSpec:
    family_catalog = build_object_family_catalog()
    family = family_catalog["crate_box"]
    material_key = "wood_plywood"
    return ObjectInstanceSpec(
        name="target_0",
        family_key=family.key,
        shape=family.shape,
        semantic_role=family.semantic_role,
        size={"hx": 0.18, "hy": 0.18, "hz": 0.18, "corner_radius": 0.03},
        mass=1.30,
        friction=0.76,
        restitution=0.05,
        linear_damping=0.03,
        angular_damping=0.03,
        material_key=material_key,
        color=_material_color(material_key),
        dynamic=True,
        role="dynamic",
        position=(0.10, 0.0, 0.18),
        orientation_euler_deg=(0.0, 0.0, 0.0),
        linear_velocity=(0.0, 0.0, 0.0),
        angular_velocity=(0.0, 0.0, 0.0),
        metadata={"toy_role": "target"},
    )


def build_ball_throw_blueprint(
    *,
    sample_key: str,
    surface_key: str,
    driver_material_key: str = "rubber_red",
    driver_family_key: str = "ball",
) -> ScenarioBlueprint:
    camera_key = "cam_00"
    camera = build_camera_catalog()[camera_key]
    driver = _driver_object(family_key=driver_family_key, material_key=driver_material_key)
    target = _target_object()
    driver_name = build_object_family_catalog()[driver_family_key].display_name
    blueprint = ScenarioBlueprint(
        family_key="F2",
        sample_key=sample_key,
        title=f"{driver_name} thrown into wood block",
        description="Toy two-object interaction with a single moving driver impacting a wooden target block.",
        gravity=EARTH_GRAVITY,
        pre_roll_s=0.05,
        camera_key=camera_key,
        surface_key=surface_key,
        lighting_key=camera.hdri_key,
        camera=camera,
        objects=(driver, target),
        tags=("toy_dataset_0718", "single_attribute_pair", "ball_block_throw"),
        metadata={
            "toy_case_key": DEFAULT_CASE_KEY,
            "driver_family_key": driver_family_key,
            "driver_material_key": driver_material_key,
            "target_material_key": target.material_key,
            "attribute_changes_supported": list(ATTRIBUTE_CHOICES),
        },
    )
    validate_blueprint_physics(blueprint)
    return blueprint


def _toy_surface_catalog() -> dict[str, SurfaceThemeSpec]:
    base = BASE_BUILD_SURFACE_CATALOG()
    base.update(
        {
            "toy_bg_beige": SurfaceThemeSpec(
                key="toy_bg_beige",
                floor_material_key="floor_wood",
                wall_material_key="wall_beige",
                floor_friction_range=RangeSpec(0.72, 0.72),
                background_mode="studio",
                notes="Toy clean wall backdrop in beige.",
            ),
            "toy_bg_cream": SurfaceThemeSpec(
                key="toy_bg_cream",
                floor_material_key="floor_wood",
                wall_material_key="wall_cream",
                floor_friction_range=RangeSpec(0.72, 0.72),
                background_mode="studio",
                notes="Toy clean wall backdrop in cream.",
            ),
            "toy_bg_concrete": SurfaceThemeSpec(
                key="toy_bg_concrete",
                floor_material_key="floor_wood",
                wall_material_key="concrete_clean_wall",
                floor_friction_range=RangeSpec(0.72, 0.72),
                background_mode="studio",
                notes="Toy clean wall backdrop in light concrete gray.",
            ),
        }
    )
    return base


@contextmanager
def _override_surface_catalog() -> Iterator[None]:
    original = render_sim.build_surface_catalog
    render_sim.build_surface_catalog = _toy_surface_catalog
    try:
        yield
    finally:
        render_sim.build_surface_catalog = original


def _render_case(
    *,
    output_root: Path,
    blueprint: ScenarioBlueprint,
    seed: int,
    width: int,
    height: int,
    scene_style: str,
    overwrite: bool,
) -> dict[str, object]:
    if overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    with _override_surface_catalog():
        return render_sim.render_blueprint_case(
            blueprint=blueprint,
            seed=seed,
            output_root=output_root,
            width=width,
            height=height,
            scene_style=scene_style,
        )


def _pair_variants(case_key: str) -> dict[str, dict[str, object]]:
    return {
        "background_color": {
            "anchor_surface_key": "toy_bg_beige",
            "variant_surface_key": "toy_bg_cream",
            "driver_material_key": "rubber_red",
            "driver_family_key": "ball",
            "anchor_label": "beige_backdrop",
            "variant_label": "cream_backdrop",
        },
        "object_color": {
            "surface_key": "toy_bg_beige",
            "anchor_driver_material_key": "rubber_red",
            "variant_driver_material_key": "rubber_blue",
            "driver_family_key": "ball",
            "anchor_label": "red_ball",
            "variant_label": "blue_ball",
        },
        "object_shape": {
            "surface_key": "toy_bg_beige",
            "driver_material_key": "rubber_red",
            "anchor_driver_family_key": "ball",
            "variant_driver_family_key": "capsule_can",
            "anchor_label": "ball_driver",
            "variant_label": "capsule_driver",
        },
    }


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    base_case_root = output_root / "cases" / args.case_key / "base"
    base_blueprint = build_ball_throw_blueprint(
        sample_key=f"{args.case_key}_base",
        surface_key="toy_bg_beige",
    )
    generated: dict[str, object] = {"base_case": None, "pairs": []}

    base_manifest = _render_case(
        output_root=base_case_root,
        blueprint=base_blueprint,
        seed=args.seed,
        width=args.width,
        height=args.height,
        scene_style=args.scene_style,
        overwrite=args.overwrite,
    )
    generated["base_case"] = base_manifest

    pair_specs = _pair_variants(args.case_key)
    for attribute in args.attributes:
        spec = pair_specs[attribute]
        pair_root = output_root / "pairs" / attribute / args.case_key
        anchor_root = pair_root / "anchor"
        variant_root = pair_root / "variant"

        if attribute == "background_color":
            anchor_blueprint = build_ball_throw_blueprint(
                sample_key=f"{args.case_key}_{attribute}_anchor",
                surface_key=str(spec["anchor_surface_key"]),
                driver_material_key=str(spec["driver_material_key"]),
                driver_family_key=str(spec["driver_family_key"]),
            )
            variant_blueprint = build_ball_throw_blueprint(
                sample_key=f"{args.case_key}_{attribute}_variant",
                surface_key=str(spec["variant_surface_key"]),
                driver_material_key=str(spec["driver_material_key"]),
                driver_family_key=str(spec["driver_family_key"]),
            )
        elif attribute == "object_color":
            anchor_blueprint = build_ball_throw_blueprint(
                sample_key=f"{args.case_key}_{attribute}_anchor",
                surface_key=str(spec["surface_key"]),
                driver_material_key=str(spec["anchor_driver_material_key"]),
                driver_family_key=str(spec["driver_family_key"]),
            )
            variant_blueprint = build_ball_throw_blueprint(
                sample_key=f"{args.case_key}_{attribute}_variant",
                surface_key=str(spec["surface_key"]),
                driver_material_key=str(spec["variant_driver_material_key"]),
                driver_family_key=str(spec["driver_family_key"]),
            )
        elif attribute == "object_shape":
            anchor_blueprint = build_ball_throw_blueprint(
                sample_key=f"{args.case_key}_{attribute}_anchor",
                surface_key=str(spec["surface_key"]),
                driver_material_key=str(spec["driver_material_key"]),
                driver_family_key=str(spec["anchor_driver_family_key"]),
            )
            variant_blueprint = build_ball_throw_blueprint(
                sample_key=f"{args.case_key}_{attribute}_variant",
                surface_key=str(spec["surface_key"]),
                driver_material_key=str(spec["driver_material_key"]),
                driver_family_key=str(spec["variant_driver_family_key"]),
            )
        else:
            raise ValueError(f"unsupported attribute {attribute}")

        anchor_manifest = _render_case(
            output_root=anchor_root,
            blueprint=anchor_blueprint,
            seed=args.seed,
            width=args.width,
            height=args.height,
            scene_style=args.scene_style,
            overwrite=args.overwrite,
        )
        variant_manifest = _render_case(
            output_root=variant_root,
            blueprint=variant_blueprint,
            seed=args.seed,
            width=args.width,
            height=args.height,
            scene_style=args.scene_style,
            overwrite=args.overwrite,
        )
        pair_manifest = {
            "attribute": attribute,
            "case_key": args.case_key,
            "anchor_label": spec["anchor_label"],
            "variant_label": spec["variant_label"],
            "anchor": anchor_manifest,
            "variant": variant_manifest,
        }
        (pair_root / "pair_manifest.json").write_text(
            json.dumps(pair_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        generated["pairs"].append(pair_manifest)

    dataset_manifest = {
        "dataset_root": str(output_root),
        "case_key": args.case_key,
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
        "scene_style": args.scene_style,
        "rendered_attributes": list(args.attributes),
        "available_attributes": list(ATTRIBUTE_CHOICES),
        **generated,
    }
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(dataset_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(dataset_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
