"""Render exact RGB0--RGB7 inputs for the 36-case phase-one pilot.

Saved simulator state is used only by this data-preparation step to render the
observed prefix.  The exported vision input tree contains no state arrays,
future frames, collision blueprint, or evaluation labels.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

from context_rgb_pybullet_common import (
    BALL_RADIUS_M,
    FPS,
    OBSERVED_FRAMES,
    camera_calibration,
    dump_json,
    pixel_sha256,
    sha256_file,
)


PROJECT = Path(__file__).resolve().parent
ENGINE_ROOT = Path("/home/gaoya/Code_Video/Dataset_physv_v2v_0819")
DATA_DEFAULT = Path("/data/gaoya/agent-data/outputs/physvideo_next_experiment_20260921_phase1_v5")
OUTPUT_DEFAULT = Path("/data/gaoya/agent-data/outputs/physvideo_context_rgb_to_pybullet_20260921_v1")
WIDTH = 640
HEIGHT = 360


def load_engine():
    sys.path[:0] = [str(PROJECT), str(ENGINE_ROOT)]
    import prepare_physvideo_phase1_pilot as pilot
    from scripts import render_sim_0705 as renderer

    generator, _ = pilot.load_engine()
    return pilot, generator, renderer


def reconstruct_case(pilot, generator, record: dict):
    family = str(record["family"])
    group = int(str(record["group_id"]).rsplit("g", 1)[1])
    value = float(record["geometry_value"])
    family_index = {"aperture": 0, "deflector": 1, "support_edge": 2}[family]
    seed = 2026092100 + family_index * 100 + group
    case = pilot.make_case(generator, family, group, value, seed)
    if case.case_id != record["key"]:
        raise ValueError(f"case reconstruction mismatch: {case.case_id} != {record['key']}")
    return case, seed


def renderable_blueprint(blueprint):
    family_for_shape = {
        "sphere": "ball",
        "puck": "flat_puck",
        "capsule": "capsule_can",
        "cylinder": "upright_cylinder",
        "rounded_box": "barrier_box",
        "box": "barrier_box",
    }
    objects = tuple(
        replace(obj, family_key=family_for_shape.get(obj.shape, "barrier_box"))
        for obj in blueprint.objects
    )
    return replace(blueprint, family_key="F1", objects=objects)


def load_observed_pose(sample_dir: Path, expected_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(sample_dir / "raw" / "states_xyzw.npz", allow_pickle=False) as archive:
        names = archive["object_names"].astype(str).tolist()
        positions = archive["positions"][:OBSERVED_FRAMES].astype(np.float64).copy()
        quaternions = archive["quats"][:OBSERVED_FRAMES].astype(np.float64).copy()
        times = archive["frame_times"][:OBSERVED_FRAMES].astype(np.float64).copy()
    if names != expected_names:
        raise ValueError(f"render object order mismatch: {names} != {expected_names}")
    if positions.shape != (OBSERVED_FRAMES, len(names), 3):
        raise ValueError("invalid observed position shape")
    if quaternions.shape != (OBSERVED_FRAMES, len(names), 4):
        raise ValueError("invalid observed quaternion shape")
    if not np.allclose(times, np.arange(OBSERVED_FRAMES) / FPS, atol=2e-7, rtol=0.0):
        raise ValueError(f"unexpected context times: {times.tolist()}")
    return positions, quaternions, times


def render_observed_prefix(renderer_module, blueprint, seed: int, positions: np.ndarray, quaternions: np.ndarray,
                           output_dir: Path, width: int, height: int, scene_style: str) -> np.ndarray:
    render_blueprint = renderable_blueprint(blueprint)
    scenario = renderer_module.blueprint_to_legacy_scenario(render_blueprint, seed=seed)
    materials = renderer_module.build_material_catalog()
    renderer_module.register_material_assets(materials)
    output_dir.mkdir(parents=True, exist_ok=False)
    frames: list[np.ndarray] = []
    with renderer_module.override_legacy_runtime(
        output_root=output_dir.parent / "renderer_runtime",
        camera=blueprint.camera,
        width=width,
        height=height,
    ):
        renderer = renderer_module.RealismPreviewRenderer(
            camera=blueprint.camera,
            surface_key=blueprint.surface_key,
            lighting_key=blueprint.lighting_key,
            width=width,
            height=height,
            scene_style=scene_style,
            capture_instance_masks=False,
            capture_rgb_frames_dir=None,
            capture_depth_frames=False,
        )
        renderer.object_materials = {
            obj.name: materials[obj.material_key] for obj in render_blueprint.objects
        }
        renderer.object_texture_seeds = {
            obj.name: renderer_module._stable_name_seed(
                str(obj.metadata.get("appearance_group", obj.name))
            )
            for obj in render_blueprint.objects
        }
        try:
            for obj in scenario.objects:
                renderer.add_object(obj)
            for frame_index in range(OBSERVED_FRAMES):
                for object_index, name in enumerate([obj.name for obj in scenario.objects]):
                    renderer.update_pose(
                        name,
                        positions[frame_index, object_index].tolist(),
                        quaternions[frame_index, object_index].tolist(),
                    )
                rgb = np.asarray(renderer.render(), dtype=np.uint8)
                if rgb.shape != (height, width, 3):
                    raise ValueError(f"renderer returned {rgb.shape}, expected {(height, width, 3)}")
                Image.fromarray(rgb, mode="RGB").save(output_dir / f"rgb_{frame_index:02d}.png")
                frames.append(rgb.copy())
        finally:
            renderer.cleanup()
    return np.stack(frames)


def prepare_case(pilot, generator, renderer_module, record: dict, data_root: Path, output_root: Path,
                 width: int, height: int, scene_style: str) -> dict:
    key = str(record["key"])
    case, seed = reconstruct_case(pilot, generator, record)
    sample_dir = data_root / "samples" / key
    expected_names = [obj.name for obj in case.blueprint.objects if not obj.metadata.get("visual_only")]
    positions, quaternions, times = load_observed_pose(sample_dir, expected_names)
    case_root = output_root / "vision_inputs" / key
    started = time.perf_counter()
    rgb = render_observed_prefix(
        renderer_module,
        case.blueprint,
        seed,
        positions,
        quaternions,
        case_root,
        width,
        height,
        scene_style,
    )
    calibration = camera_calibration(case.blueprint.camera, width, height)
    calibration["known_object_prior"] = {
        "shape": "sphere",
        "radius_m": BALL_RADIUS_M,
        "scope": "fixed protocol constant shared by all 36 cases",
    }
    dump_json(case_root / "calibration.json", calibration)
    input_payload = {
        "schema": "context_rgb8_pybullet_input_v1",
        "case_id": key,
        "family": str(record["family"]),
        "frames": [f"rgb_{index:02d}.png" for index in range(OBSERVED_FRAMES)],
        "observed_indices": list(range(OBSERVED_FRAMES)),
        "time_s": times.tolist(),
        "shape": list(rgb.shape),
        "pixels_sha256": pixel_sha256(rgb),
        "calibration": "calibration.json",
        "allowed_estimator_inputs": [
            "RGB0-RGB7 pixels",
            "observed timestamps",
            "fixed camera calibration",
            "family label",
            "fixed sphere radius prior",
        ],
    }
    dump_json(case_root / "input.json", input_payload)
    files = [case_root / name for name in input_payload["frames"]] + [
        case_root / "input.json",
        case_root / "calibration.json",
    ]
    return {
        "case_id": key,
        "family": str(record["family"]),
        "group_id": str(record["group_id"]),
        "geometry_value": float(record["geometry_value"]),
        "geometry_units": str(record["geometry_units"]),
        "input_dir": key,
        "input_json": f"{key}/input.json",
        "render_seconds": time.perf_counter() - started,
        "output_sha256": {path.name: sha256_file(path) for path in files},
        "source_state_scope": "positions/quaternions RGB0-RGB7 only, render preparation only",
        "future_rgb_rendered": False,
        "future_state_exported_to_vision_tree": False,
        "blueprint_exported_to_vision_tree": False,
    }


def run(args: argparse.Namespace) -> dict:
    data_root = args.data_root.resolve()
    output_root = args.output.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_root}")
    output_root.mkdir(parents=True)
    manifest = json.loads((data_root / "pilot_manifest.json").read_text(encoding="utf-8"))
    records = sorted(manifest["records"], key=lambda row: row["key"])
    if len(records) != 36:
        raise ValueError(f"expected the fixed 36-case pilot, got {len(records)}")
    counts = {family: sum(row["family"] == family for row in records)
              for family in ("aperture", "deflector", "support_edge")}
    if counts != {"aperture": 12, "deflector": 12, "support_edge": 12}:
        raise ValueError(f"family coverage mismatch: {counts}")

    pilot, generator, renderer_module = load_engine()
    rows = []
    for index, record in enumerate(records, start=1):
        row = prepare_case(
            pilot,
            generator,
            renderer_module,
            record,
            data_root,
            output_root,
            args.width,
            args.height,
            args.scene_style,
        )
        rows.append(row)
        print(f"CONTEXT_READY {index:02d}/36 {row['case_id']} {row['render_seconds']:.3f}s", flush=True)

    vision_manifest = {
        "schema": "context_rgb8_pybullet_vision_manifest_v1",
        "case_count": len(rows),
        "family_counts": counts,
        "records": [
            {key: row[key] for key in ("case_id", "family", "group_id", "input_dir", "input_json")}
            for row in rows
        ],
        "input_root": str((output_root / "vision_inputs").resolve()),
        "future_fields_present": False,
        "gt_state_fields_present": False,
        "blueprint_fields_present": False,
    }
    dump_json(output_root / "vision_inputs" / "manifest.json", vision_manifest)
    dump_json(
        output_root / "evaluation_manifest.json",
        {
            "schema": "context_rgb8_pybullet_evaluation_manifest_v1",
            "data_root": str(data_root),
            "vision_input_root": str((output_root / "vision_inputs").resolve()),
            "records": rows,
            "policy": "GT state, future, and blueprint are evaluation-only and are not arguments to GPU inference",
        },
    )
    dump_json(
        output_root / "input_contract.json",
        {
            "status": "EXECUTED",
            "estimator_allowed": vision_manifest["records"][0].keys() if False else [
                "manifest case_id/family/input_dir/input_json",
                "per-case RGB0-RGB7",
                "timestamps",
                "fixed camera calibration",
                "fixed 0.11 m sphere-radius prior",
            ],
            "estimator_forbidden": [
                "RGB8-RGB48",
                "positions/velocities/quaternions",
                "future trajectory",
                "interaction/contact labels",
                "blueprint objects or dimensions",
            ],
            "render_preparation": "reads only positions/quaternions/times slices [0:8] to make observed pixels",
            "vision_tree": str((output_root / "vision_inputs").resolve()),
        },
    )
    summary = {
        "schema": "context_rgb8_pybullet_prepare_report_v1",
        "status": "EXECUTED",
        "case_count": len(rows),
        "family_counts": counts,
        "resolution_wh": [args.width, args.height],
        "scene_style": args.scene_style,
        "future_rgb_rendered": False,
        "vision_manifest": str(output_root / "vision_inputs" / "manifest.json"),
        "evaluation_manifest": str(output_root / "evaluation_manifest.json"),
        "total_render_seconds": float(sum(row["render_seconds"] for row in rows)),
    }
    dump_json(output_root / "prepare_report.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument(
        "--scene-style",
        default="indoor_natural",
        choices=("simple", "indoor_natural", "indoor_realistic"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
