#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

from .common_specs import ObjectInstanceSpec, ScenarioBlueprint
from .material_catalog_0705 import build_material_catalog
from .object_catalog_0705 import build_object_family_catalog
from .scene_generators_0705 import EARTH_GRAVITY, build_camera_catalog, validate_blueprint_physics
from .generate_toy_0718_pairs import _override_surface_catalog
from . import render_sim_0705 as render_sim


DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0718ToyDataset")
ATTRIBUTES = ("background_color", "object_color", "object_shape")
BASE_SURFACE = "toy_bg_beige"
VARIANT_SURFACE = "toy_bg_cream"


@dataclass(frozen=True)
class ToyCaseSpec:
    case_id: str
    slug: str
    object_count: int
    primary_family: str
    secondary_family: str | None = None
    tertiary_family: str | None = None
    speed: float = 0.0
    lateral_offset: float = 0.0
    vertical_mode: str = "ground"
    spin: float = 0.0
    title: str = ""

    @property
    def key(self) -> str:
        return f"{self.case_id}_{self.slug}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 50 simple controlled-variable toy physics cases.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--scene-style", default="toy_simple")
    parser.add_argument("--case-ids", nargs="*", default=(), help="Optional case IDs, for example case_001 case_002.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--catalog-only", action="store_true")
    return parser.parse_args()


def build_case_catalog() -> tuple[ToyCaseSpec, ...]:
    single_rows = [
        ("ball_roll", "ball", 2.4, 8.0, "ground"),
        ("puck_slide", "flat_puck", 2.8, 0.5, "ground"),
        ("capsule_slide", "capsule_can", 2.5, 3.0, "ground"),
        ("cylinder_topple", "upright_cylinder", 1.2, 5.0, "ground"),
        ("box_slide", "crate_box", 2.2, 0.0, "ground"),
        ("ball_bounce", "ball", 1.4, 4.0, "drop"),
        ("ball_oblique_throw", "ball", 3.1, 7.0, "throw"),
        ("barrel_roll", "drum_barrel", 2.0, 8.0, "ground"),
        ("wheel_roll", "wheel", 2.7, 10.0, "ground"),
        ("spool_roll", "spool", 2.1, 8.0, "ground"),
        ("dumbbell_tumble", "dumbbell", 2.0, 7.0, "ground"),
        ("frustum_spin", "cone_frustum", 0.8, 12.0, "ground"),
        ("tall_box_tumble", "tall_box", 1.5, 6.0, "ground"),
        ("ball_vertical_drop", "ball", 0.0, 0.0, "drop"),
        ("puck_spin_slide", "flat_puck", 2.0, 14.0, "ground"),
    ]
    double_rows = [
        ("ball_hits_wood_block", "ball", "crate_box", 4.4, 0.00),
        ("puck_hits_tall_box", "flat_puck", "tall_box", 3.6, 0.00),
        ("capsule_hits_cylinder", "capsule_can", "upright_cylinder", 3.8, 0.00),
        ("wheel_hits_block", "wheel", "crate_box", 3.5, 0.00),
        ("ball_glances_box", "ball", "shipping_box", 4.2, 0.13),
        ("barrel_hits_case", "drum_barrel", "tool_case", 3.2, -0.08),
        ("spool_hits_cylinder", "spool", "upright_cylinder", 3.4, 0.06),
        ("puck_pushes_box", "flat_puck", "shipping_box", 4.1, 0.00),
        ("ball_topples_tall_box", "ball", "tall_box", 4.7, 0.00),
        ("capsule_glances_block", "capsule_can", "crate_box", 4.0, -0.12),
        ("dumbbell_hits_box", "dumbbell", "crate_box", 3.2, 0.05),
        ("ball_hits_frustum", "ball", "cone_frustum", 4.0, 0.00),
        ("puck_hits_tool_case", "flat_puck", "tool_case", 3.7, 0.09),
        ("wheel_hits_cylinder", "wheel", "upright_cylinder", 3.8, -0.05),
        ("barrel_hits_tall_box", "drum_barrel", "tall_box", 3.0, 0.00),
        ("spool_hits_shipping_box", "spool", "shipping_box", 3.6, 0.11),
        ("ball_hits_tool_case", "ball", "tool_case", 4.5, -0.07),
        ("capsule_hits_frustum", "capsule_can", "cone_frustum", 3.9, 0.04),
        ("puck_hits_cylinder", "flat_puck", "upright_cylinder", 4.1, -0.10),
        ("wheel_hits_shipping_box", "wheel", "shipping_box", 3.4, 0.08),
    ]
    triple_rows = [
        ("ball_box_box_chain", "ball", "crate_box", "crate_box", 4.3, 0.00),
        ("puck_cylinder_box_chain", "flat_puck", "upright_cylinder", "shipping_box", 3.9, 0.00),
        ("capsule_box_cylinder_chain", "capsule_can", "crate_box", "upright_cylinder", 4.0, 0.00),
        ("wheel_box_box_chain", "wheel", "shipping_box", "crate_box", 3.8, 0.00),
        ("ball_cylinder_frustum_chain", "ball", "upright_cylinder", "cone_frustum", 4.5, 0.03),
        ("puck_box_case_chain", "flat_puck", "crate_box", "tool_case", 4.0, -0.03),
        ("barrel_box_cylinder_chain", "drum_barrel", "shipping_box", "upright_cylinder", 3.5, 0.00),
        ("spool_cylinder_box_chain", "spool", "upright_cylinder", "crate_box", 3.8, 0.02),
        ("ball_box_frustum_chain", "ball", "crate_box", "cone_frustum", 4.6, -0.02),
        ("capsule_case_box_chain", "capsule_can", "tool_case", "shipping_box", 4.1, 0.00),
        ("wheel_cylinder_box_chain", "wheel", "upright_cylinder", "tall_box", 4.0, 0.02),
        ("puck_box_cylinder_offset", "flat_puck", "shipping_box", "upright_cylinder", 4.2, 0.06),
        ("ball_frustum_box_offset", "ball", "cone_frustum", "crate_box", 4.4, -0.06),
        ("dumbbell_box_box_chain", "dumbbell", "crate_box", "shipping_box", 3.7, 0.03),
        ("barrel_case_cylinder_chain", "drum_barrel", "tool_case", "upright_cylinder", 3.6, -0.03),
    ]

    cases: list[ToyCaseSpec] = []
    for index, (slug, family, speed, spin, vertical_mode) in enumerate(single_rows, 1):
        cases.append(ToyCaseSpec(f"case_{index:03d}", slug, 1, family, speed=speed, spin=spin, vertical_mode=vertical_mode, title=slug.replace("_", " ").title()))
    for offset, (slug, primary, secondary, speed, lateral) in enumerate(double_rows, 16):
        cases.append(ToyCaseSpec(f"case_{offset:03d}", slug, 2, primary, secondary, speed=speed, spin=7.0, lateral_offset=lateral, title=slug.replace("_", " ").title()))
    for offset, (slug, primary, secondary, tertiary, speed, lateral) in enumerate(triple_rows, 36):
        cases.append(ToyCaseSpec(f"case_{offset:03d}", slug, 3, primary, secondary, tertiary, speed=speed, spin=7.0, lateral_offset=lateral, title=slug.replace("_", " ").title()))
    if len(cases) != 50 or [sum(case.object_count == n for case in cases) for n in (1, 2, 3)] != [15, 20, 15]:
        raise AssertionError("toy case catalog must contain 15 single, 20 double, and 15 triple cases")
    return tuple(cases)


def _size(family_key: str) -> dict[str, float]:
    return {
        "ball": {"radius": 0.16},
        "flat_puck": {"radius": 0.19, "height": 0.07},
        "capsule_can": {"radius": 0.10, "height": 0.28},
        "upright_cylinder": {"radius": 0.11, "height": 0.34},
        "crate_box": {"hx": 0.18, "hy": 0.18, "hz": 0.18, "corner_radius": 0.03},
        "tall_box": {"hx": 0.13, "hy": 0.13, "hz": 0.28},
        "cone_frustum": {"r_base": 0.16, "r_top": 0.08, "height": 0.30},
        "tool_case": {"hx": 0.23, "hy": 0.13, "hz": 0.10, "corner_radius": 0.03},
        "shipping_box": {"hx": 0.20, "hy": 0.18, "hz": 0.16},
        "drum_barrel": {"radius": 0.18, "height": 0.36},
        "wheel": {"radius": 0.18, "width": 0.13},
        "spool": {"core_radius": 0.06, "flange_radius": 0.15, "width": 0.22, "flange_width": 0.04},
        "dumbbell": {"bar_radius": 0.04, "length": 0.30, "weight_radius": 0.09},
    }[family_key]


def _ground_extent(family_key: str, orientation: tuple[float, float, float]) -> float:
    size = _size(family_key)
    if family_key == "ball":
        return size["radius"]
    if family_key in {"flat_puck", "upright_cylinder", "cone_frustum"}:
        return size["height"] * 0.5
    if family_key == "capsule_can":
        return size["radius"] if abs(orientation[1]) > 45 else size["radius"] + size["height"] * 0.5
    if family_key in {"crate_box", "tall_box", "tool_case", "shipping_box"}:
        return size["hz"]
    if family_key == "drum_barrel":
        return size["radius"] if abs(orientation[1]) > 45 else size["height"] * 0.5
    if family_key == "wheel":
        return size["radius"]
    if family_key == "spool":
        return size["flange_radius"]
    if family_key == "dumbbell":
        return size["weight_radius"]
    raise KeyError(family_key)


def _orientation(family_key: str) -> tuple[float, float, float]:
    if family_key in {"drum_barrel", "wheel", "spool", "dumbbell"}:
        return (90.0, 0.0, 0.0)
    if family_key == "capsule_can":
        return (0.0, 90.0, 0.0)
    return (0.0, 0.0, 0.0)


def _horizontal_extent(family_key: str) -> float:
    size = _size(family_key)
    if family_key == "ball":
        return size["radius"]
    if family_key in {"flat_puck", "upright_cylinder", "drum_barrel", "wheel"}:
        return size["radius"]
    if family_key == "capsule_can":
        return size["radius"] + 0.5 * size["height"]
    if family_key in {"crate_box", "tall_box", "tool_case", "shipping_box"}:
        return size["hx"]
    if family_key == "cone_frustum":
        return max(size["r_base"], size["r_top"])
    if family_key == "spool":
        return size["flange_radius"]
    if family_key == "dumbbell":
        return size["weight_radius"]
    raise KeyError(family_key)


def _object(
    family_key: str,
    *,
    name: str,
    position: tuple[float, float, float],
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    spin: float = 0.0,
    material_key: str = "rubber_red",
) -> ObjectInstanceSpec:
    family = build_object_family_catalog()[family_key]
    orientation = _orientation(family_key)
    material = build_material_catalog()[material_key]
    return ObjectInstanceSpec(
        name=name,
        family_key=family_key,
        shape=family.shape,
        semantic_role=family.semantic_role,
        size=_size(family_key),
        mass=1.05 if name in {"driver_0", "lead_0"} else 1.30,
        friction=0.54 if name in {"driver_0", "lead_0"} else 0.72,
        restitution=0.42 if family_key == "ball" else 0.08,
        linear_damping=0.02,
        angular_damping=0.02,
        material_key=material_key,
        color=tuple(float(v) for v in material.base_color),
        dynamic=True,
        role="dynamic",
        position=position,
        orientation_euler_deg=orientation,
        linear_velocity=velocity,
        angular_velocity=(0.0, spin, 0.0),
        metadata={"toy_role": name},
    )


def build_base_blueprint(spec: ToyCaseSpec, sample_key: str) -> ScenarioBlueprint:
    objects: list[ObjectInstanceSpec] = []
    primary_orientation = _orientation(spec.primary_family)
    primary_z = _ground_extent(spec.primary_family, primary_orientation)
    velocity = (spec.speed, spec.lateral_offset * 0.35, 0.0)
    if spec.vertical_mode == "drop":
        primary_z = 1.18
        velocity = (spec.speed, 0.0, -0.15)
    elif spec.vertical_mode == "throw":
        primary_z = 0.72
        velocity = (spec.speed, 0.0, 0.35)
    primary_name = "driver_0" if spec.object_count < 3 else "lead_0"
    objects.append(_object(spec.primary_family, name=primary_name, position=(-1.65, spec.lateral_offset, primary_z), velocity=velocity, spin=spec.spin))

    if spec.secondary_family:
        orientation = _orientation(spec.secondary_family)
        objects.append(_object(spec.secondary_family, name="target_0" if spec.object_count == 2 else "mid_0", position=(-0.05, 0.0, _ground_extent(spec.secondary_family, orientation)), material_key="wood_plywood"))
    if spec.tertiary_family:
        orientation = _orientation(spec.tertiary_family)
        tail_x = (
            -0.05
            + _horizontal_extent(str(spec.secondary_family))
            + _horizontal_extent(spec.tertiary_family)
            + 0.025
        )
        objects.append(_object(spec.tertiary_family, name="tail_0", position=(tail_x, -spec.lateral_offset * 0.25, _ground_extent(spec.tertiary_family, orientation)), material_key="painted_metal_yellow"))

    camera_key = "cam_00" if spec.vertical_mode == "ground" else "cam_05"
    camera = build_camera_catalog()[camera_key]
    blueprint = ScenarioBlueprint(
        family_key=f"F{spec.object_count}",
        sample_key=sample_key,
        title=spec.title,
        description=f"Simple deterministic toy motion with {spec.object_count} visible object(s).",
        gravity=EARTH_GRAVITY,
        pre_roll_s=0.03,
        camera_key=camera_key,
        surface_key=BASE_SURFACE,
        lighting_key=camera.hdri_key,
        camera=camera,
        objects=tuple(objects),
        tags=("toy_dataset_0718", "controlled_variable", "simple_motion", "head_on" if spec.object_count > 1 else "roll"),
        metadata={"case_id": spec.case_id, "case_key": spec.key, "object_count": spec.object_count, "primary_object": primary_name},
    )
    validate_blueprint_physics(blueprint)
    return blueprint


def _shape_variant(obj: ObjectInstanceSpec) -> ObjectInstanceSpec:
    if obj.family_key == "ball":
        family_key, size = "capsule_can", {"radius": 0.112, "height": 0.096}
    elif obj.family_key in {"flat_puck", "wheel"}:
        radius = obj.size["radius"]
        half_width = 0.5 * obj.size.get("height", obj.size.get("width", 2.0 * radius))
        family_key, size = "crate_box", {"hx": radius, "hy": radius, "hz": half_width, "corner_radius": 0.02}
    elif obj.family_key == "capsule_can":
        family_key, size = "crate_box", {
            "hx": obj.size["radius"],
            "hy": obj.size["radius"],
            "hz": obj.size["radius"] + 0.5 * obj.size["height"],
            "corner_radius": 0.02,
        }
    elif obj.family_key == "dumbbell":
        family_key, size = "crate_box", {
            "hx": obj.size["weight_radius"],
            "hy": obj.size["weight_radius"],
            "hz": obj.size["weight_radius"] + 0.5 * obj.size["length"],
            "corner_radius": 0.02,
        }
    elif obj.family_key in {"upright_cylinder", "cone_frustum", "drum_barrel"}:
        if obj.family_key == "cone_frustum":
            radius, half_height = max(obj.size["r_base"], obj.size["r_top"]), 0.5 * obj.size["height"]
        else:
            radius, half_height = obj.size["radius"], 0.5 * obj.size["height"]
        family_key, size = "crate_box", {"hx": radius, "hy": radius, "hz": half_height, "corner_radius": 0.02}
    elif obj.family_key == "spool":
        family_key, size = "crate_box", {
            "hx": obj.size["flange_radius"],
            "hy": obj.size["flange_radius"],
            "hz": 0.5 * obj.size["width"],
            "corner_radius": 0.02,
        }
    else:
        extent = _ground_extent(obj.family_key, obj.orientation_euler_deg)
        family_key, size = "upright_cylinder", {"radius": extent, "height": 2.0 * extent}
    family = build_object_family_catalog()[family_key]
    return replace(obj, family_key=family_key, shape=family.shape, semantic_role=family.semantic_role, size=size)


def build_variant(base: ScenarioBlueprint, attribute: str) -> ScenarioBlueprint:
    primary_name = str(base.metadata["primary_object"])
    if attribute == "background_color":
        return replace(base, sample_key=f"{base.metadata['case_key']}_{attribute}", surface_key=VARIANT_SURFACE)
    objects = list(base.objects)
    primary_index = next(index for index, obj in enumerate(objects) if obj.name == primary_name)
    if attribute == "object_color":
        blue = build_material_catalog()["rubber_blue"]
        objects[primary_index] = replace(objects[primary_index], material_key="rubber_blue", color=tuple(float(v) for v in blue.base_color))
    elif attribute == "object_shape":
        objects[primary_index] = _shape_variant(objects[primary_index])
    else:
        raise ValueError(attribute)
    return replace(base, sample_key=f"{base.metadata['case_key']}_{attribute}", objects=tuple(objects))


def _render(blueprint: ScenarioBlueprint, root: Path, args: argparse.Namespace, seed: int) -> dict:
    if args.overwrite and root.exists():
        shutil.rmtree(root)
    manifest_path = root / "case_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    with _override_surface_catalog():
        return render_sim.render_blueprint_case(
            blueprint=blueprint,
            seed=seed,
            output_root=root,
            width=args.width,
            height=args.height,
            scene_style=args.scene_style,
            export_instance_masks=True,
            preserve_states=True,
        )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _selected_cases(cases: Iterable[ToyCaseSpec], case_ids: tuple[str, ...]) -> list[ToyCaseSpec]:
    selected = list(cases) if not case_ids else [case for case in cases if case.case_id in case_ids]
    missing = sorted(set(case_ids) - {case.case_id for case in selected})
    if missing:
        raise ValueError(f"unknown case IDs: {', '.join(missing)}")
    return selected


def main() -> None:
    args = parse_args()
    cases = build_case_catalog()
    selected = _selected_cases(cases, tuple(args.case_ids))
    config_root = args.output_root / "config"
    _write_json(config_root / "case_catalog.json", {"cases": [case.__dict__ for case in cases]})
    _write_json(config_root / "generation_config.json", {"seed": args.seed, "width": args.width, "height": args.height, "scene_style": args.scene_style, "attributes": list(ATTRIBUTES), "mask_format": "uint8 instance IDs; background=0"})
    if args.catalog_only:
        print(json.dumps({"case_count": len(cases), "catalog": str(config_root / "case_catalog.json")}, indent=2))
        return

    dataset_cases: list[dict[str, object]] = []
    for spec in selected:
        case_root = args.output_root / "cases" / spec.key
        case_seed = args.seed + int(spec.case_id.rsplit("_", 1)[1])
        base_blueprint = build_base_blueprint(spec, f"{spec.key}_base")
        base_manifest = _render(base_blueprint, case_root / "base", args, case_seed)
        pairs: list[dict[str, object]] = []
        for attribute in ATTRIBUTES:
            variant = build_variant(base_blueprint, attribute)
            variant_manifest = _render(variant, case_root / "variants" / attribute, args, case_seed)
            pair = {"attribute": attribute, "anchor": base_manifest, "variant": variant_manifest, "only_changed_attribute": attribute}
            _write_json(case_root / "pairs" / f"{attribute}.json", pair)
            pairs.append(pair)
        case_manifest = {"case_id": spec.case_id, "case_key": spec.key, "title": spec.title, "object_count": spec.object_count, "seed": case_seed, "base": base_manifest, "pairs": pairs}
        _write_json(case_root / "case_manifest.json", case_manifest)
        dataset_cases.append(case_manifest)
        print(json.dumps({"completed": spec.case_id, "case_key": spec.key}, ensure_ascii=False), flush=True)

    existing_manifests = sorted((args.output_root / "cases").glob("case_*/case_manifest.json"))
    all_generated = [json.loads(path.read_text(encoding="utf-8")) for path in existing_manifests]
    dataset_manifest = {
        "dataset_root": str(args.output_root),
        "schema_version": "toy_controlled_pairs_v2",
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
        "scene_style": args.scene_style,
        "requested_case_count": 50,
        "generated_case_count": len(all_generated),
        "case_distribution": {"one_object": 15, "two_objects": 20, "three_objects": 15},
        "videos_per_case": 4,
        "mask_outputs": ["color_preview_mp4", "lossless_instance_ids_npz"],
        "cases": all_generated,
    }
    _write_json(args.output_root / "dataset_manifest.json", dataset_manifest)
    print(json.dumps({"generated_this_run": len(dataset_cases), "generated_total": len(all_generated), "manifest": str(args.output_root / "dataset_manifest.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
