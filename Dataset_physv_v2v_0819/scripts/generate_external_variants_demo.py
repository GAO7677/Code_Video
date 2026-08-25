#!/usr/bin/env python3
"""Render a small, metadata-preserving external-condition variant demo."""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT.parent))

from Dataset_physv_v2v_0819.scripts.common_specs import (  # noqa: E402
    CameraSpec,
    ObjectInstanceSpec,
    ScenarioBlueprint,
)
from Dataset_physv_v2v_0819.scripts.render_sim_0705 import (  # noqa: E402
    render_blueprint_case,
)


DEFAULT_SOURCE_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0717pybullet_5000_vbenchtop5"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/datasets/pybullet0717_prompt_physics_consistency_v1"
) / "external_variants_demo"
DEFAULT_CASES = (
    "0717_f1_attempt000000",
    "0717_f7_attempt000007",
    "0717_f8_attempt000008",
)


def _tuple3(values: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    values = values if isinstance(values, (list, tuple)) and len(values) == 3 else default
    return tuple(float(value) for value in values)  # type: ignore[return-value]


def _load_source_index(source_root: Path) -> dict[str, dict[str, Any]]:
    manifest_path = source_root / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {str(item["case_id"]): item for item in raw}


def _load_meta(source_item: dict[str, Any]) -> dict[str, Any]:
    meta_path = Path(str(source_item["meta"]))
    if not meta_path.is_file():
        raise FileNotFoundError(meta_path)
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _blueprint_from_meta(meta: dict[str, Any], sample_key: str) -> ScenarioBlueprint:
    camera_payload = meta["camera"]
    camera = CameraSpec(
        eye=_tuple3(camera_payload.get("eye"), (0.0, -3.0, 1.5)),
        target=_tuple3(camera_payload.get("target"), (0.0, 0.2, 0.35)),
        up=_tuple3(camera_payload.get("up"), (0.0, 0.0, 1.0)),
        yfov_deg=float(camera_payload.get("yfov_deg", 48.0)),
        hdri_key=str(meta.get("lighting_key", "studio_soft")),
    )
    objects: list[ObjectInstanceSpec] = []
    for payload in meta.get("objects", []):
        material_key = str(payload.get("material_key", ""))
        family_key = str(payload.get("family_key", payload.get("object_noun", "object")))
        metadata = {
            "source_case_id": str(meta.get("case_id", meta.get("key", ""))),
            "source_object_name": str(payload.get("name", "")),
        }
        objects.append(
            ObjectInstanceSpec(
                name=str(payload["name"]),
                family_key=family_key,
                shape=str(payload["shape"]),
                semantic_role=str(payload.get("semantic_role", payload.get("role", "dynamic"))),
                size={str(k): float(v) for k, v in payload.get("size", {}).items()},
                mass=float(payload.get("mass", 1.0)),
                friction=float(payload.get("friction", 0.5)),
                restitution=float(payload.get("restitution", 0.1)),
                linear_damping=float(payload.get("linear_damping", 0.02)),
                angular_damping=float(payload.get("angular_damping", 0.04)),
                material_key=material_key,
                color=_tuple3(payload.get("color"), (0.7, 0.7, 0.7)),
                dynamic=bool(payload.get("dynamic", True)),
                role=str(payload.get("role", "dynamic")),
                position=_tuple3(payload.get("position"), (0.0, 0.0, 0.0)),
                orientation_euler_deg=_tuple3(payload.get("orientation_euler_deg"), (0.0, 0.0, 0.0)),
                linear_velocity=_tuple3(payload.get("linear_velocity"), (0.0, 0.0, 0.0)),
                angular_velocity=_tuple3(payload.get("angular_velocity"), (0.0, 0.0, 0.0)),
                metadata=metadata,
            )
        )

    blueprint_payload = meta.get("blueprint") or {}
    metadata = copy.deepcopy(blueprint_payload.get("metadata") or {})
    if meta.get("floor_friction") is not None:
        metadata["floor_friction"] = float(meta["floor_friction"])
    if meta.get("floor_restitution") is not None:
        metadata["floor_restitution"] = float(meta["floor_restitution"])
    metadata["source_case_id"] = str(meta.get("case_id", meta.get("key", "")))
    return ScenarioBlueprint(
        family_key=str(blueprint_payload.get("family_key", meta.get("family_key", ""))).split()[0],
        sample_key=sample_key,
        title=str(meta.get("title", "External-condition variant")),
        description=str(meta.get("description", "")),
        gravity=float(meta.get("gravity", 9.81)),
        pre_roll_s=float(meta.get("pre_roll_s", 0.0)),
        camera_key=str(blueprint_payload.get("camera_key", "cam_00")),
        surface_key=str(meta.get("surface_key", blueprint_payload.get("surface_key", "studio_wood_floor"))),
        lighting_key=str(meta.get("lighting_key", blueprint_payload.get("lighting_key", camera.hdri_key))),
        camera=camera,
        objects=tuple(objects),
        tags=tuple(str(tag) for tag in (meta.get("tags") or [])),
        metadata=metadata,
    )


def _rotate_xy(vector: np.ndarray, degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    c, s = math.cos(radians), math.sin(radians)
    return np.asarray((c * vector[0] - s * vector[1], s * vector[0] + c * vector[1], vector[2]))


def perturb_blueprint(
    blueprint: ScenarioBlueprint,
    *,
    source_case_id: str,
    variant_index: int,
    seed: int,
) -> tuple[ScenarioBlueprint, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    perturbations: list[dict[str, Any]] = []
    objects: list[ObjectInstanceSpec] = []
    for obj in blueprint.objects:
        if not obj.dynamic:
            objects.append(obj)
            continue

        base_position = np.asarray(obj.position, dtype=np.float64)
        base_velocity = np.asarray(obj.linear_velocity, dtype=np.float64)
        base_angular_velocity = np.asarray(obj.angular_velocity, dtype=np.float64)

        # Keep the contact height intact. Perturb only the horizontal spawn
        # location, with a smaller envelope for multi-object scenes.
        position_scale = 0.055 if len(blueprint.objects) > 1 else 0.085
        position_delta = np.asarray(
            [rng.uniform(-position_scale, position_scale), rng.uniform(-position_scale, position_scale), 0.0],
            dtype=np.float64,
        )
        new_position = base_position + position_delta

        speed_scale = float(rng.uniform(0.90, 1.10))
        heading_delta = float(rng.uniform(-8.0, 8.0))
        new_velocity = _rotate_xy(base_velocity, heading_delta) * speed_scale

        angular_scale = float(rng.uniform(0.90, 1.10))
        new_angular_velocity = base_angular_velocity * angular_scale
        # Yaw is an external orientation variable; keep roll/pitch unchanged
        # to avoid introducing contact penetration at the starting frame.
        orientation_delta = float(rng.uniform(-6.0, 6.0))
        new_orientation = list(obj.orientation_euler_deg)
        new_orientation[2] += orientation_delta

        objects.append(
            replace(
                obj,
                position=tuple(float(v) for v in new_position),
                linear_velocity=tuple(float(v) for v in new_velocity),
                angular_velocity=tuple(float(v) for v in new_angular_velocity),
                orientation_euler_deg=tuple(float(v) for v in new_orientation),
                metadata={
                    **obj.metadata,
                    "external_variant": True,
                    "source_case_id": source_case_id,
                    "variant_index": variant_index,
                },
            )
        )
        perturbations.append(
            {
                "object": obj.name,
                "position_delta_m": [round(float(v), 6) for v in position_delta],
                "velocity_scale": round(speed_scale, 6),
                "heading_delta_deg": round(heading_delta, 6),
                "angular_velocity_scale": round(angular_scale, 6),
                "yaw_delta_deg": round(orientation_delta, 6),
            }
        )

    variant = replace(
        blueprint,
        sample_key=f"{source_case_id}__extv{variant_index:02d}",
        objects=tuple(objects),
        metadata={
            **blueprint.metadata,
            "external_variant": True,
            "source_case_id": source_case_id,
            "variant_index": variant_index,
            "variant_seed": seed,
            "changed_variables": ["position_xy", "linear_velocity", "angular_velocity", "yaw"],
        },
    )
    return variant, {"variant_seed": seed, "objects": perturbations}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source-case", action="append", dest="source_cases")
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    source_index = _load_source_index(source_root)
    source_cases = tuple(args.source_cases or DEFAULT_CASES)
    output_root.mkdir(parents=True, exist_ok=True)
    generated: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []

    for source_offset, source_case_id in enumerate(source_cases):
        if source_case_id not in source_index:
            raise KeyError(f"source case not found: {source_case_id}")
        source_item = source_index[source_case_id]
        source_meta = _load_meta(source_item)
        blueprint = _blueprint_from_meta(source_meta, source_case_id)
        source_records.append(
            {
                "source_case_id": source_case_id,
                "family_key": source_item.get("family_key"),
                "source_video": source_item.get("video"),
                "source_meta": source_item.get("meta"),
            }
        )
        count_rng = np.random.default_rng(args.seed + source_offset * 1009)
        variant_count = int(count_rng.integers(2, 4))
        print(f"{source_case_id}: generating {variant_count} variants")
        for variant_index in range(1, variant_count + 1):
            variant_seed = int(args.seed + source_offset * 1009 + variant_index * 37)
            variant, perturbations = perturb_blueprint(
                blueprint,
                source_case_id=source_case_id,
                variant_index=variant_index,
                seed=variant_seed,
            )
            case_root = output_root / "cases" / str(source_item["family_key"]) / variant.sample_key
            existing_manifest = case_root / "case_manifest.json"
            if existing_manifest.is_file():
                manifest = json.loads(existing_manifest.read_text(encoding="utf-8"))
                generated.append(manifest)
                print(f"  reuse {variant.sample_key}")
                continue
            manifest = render_blueprint_case(
                blueprint=variant,
                seed=variant_seed,
                output_root=case_root,
                width=1280,
                height=720,
                scene_style="indoor_realistic",
            )
            manifest.update(
                {
                    "source_case_id": source_case_id,
                    "source_video": str(source_item["video"]),
                    "source_meta": str(source_item["meta"]),
                    "variant_index": variant_index,
                    "variant_seed": variant_seed,
                    "perturbations": perturbations,
                }
            )
            (case_root / "case_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            generated.append(manifest)
            print(f"  done {variant.sample_key}: {manifest['video']}")

    (output_root / "source_records.json").write_text(
        json.dumps(source_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "manifest.json").write_text(
        json.dumps(generated, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_root / "README.json").write_text(
        json.dumps(
            {
                "purpose": "demo external-condition variants; no VAE/prompt/Utonia cache generated",
                "source_root": str(source_root),
                "source_cases": list(source_cases),
                "num_generated": len(generated),
                "changed_variables": ["position_xy", "linear_velocity", "angular_velocity", "yaw"],
                "unchanged_variables": ["shape", "size", "mass", "friction", "restitution", "damping", "material", "static_scene", "camera"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"generated={len(generated)} output_root={output_root}")


if __name__ == "__main__":
    main()
