#!/usr/bin/env python3
"""Add the formal incline-to-horizontal-platform control family to strict CYCLES.

The four cases keep the block, incline, table height, camera, materials, and
physics parameters fixed.  Only the horizontal platform length and the
position of its right-side legs change.  Staging uses the existing PyBullet
exporter; the final reference and aligned truth use the existing CYCLES
pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .build_physv_v2v_0819_strict import (
    regenerate_contexts,
    synchronize_cycle_config,
    validate_case,
    write_first_frame,
)
from .caption_observations_0819 import derive_caption_observations
from .caption_templates_0819 import attach_caption_metadata
from .common_specs import CameraSpec
from .export_physv_v2v_0819_dataset import (
    ExportCase,
    _package_case,
    _validate_sample,
    _write_json,
)
from .generate_v2v_context_demos import (
    SCENE_STYLE,
    V2V_QUESTION,
    _blueprint,
    _object,
)
from .render_sim_0705 import render_blueprint_case


DATASET_ROOT = Path("/data/gaoya/AAA_test_video/physv_v2v_0819_strict")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BLENDER = Path("/data/gaoya/agent-data/tools/blender-3.6.23-linux-x64/blender")
FFMPEG = Path(shutil.which("ffmpeg") or "/usr/bin/ffmpeg")
CYCLES_CACHE = Path("/data/gaoya/agent-data/cache/physv_ramp_platform_cycles")
SMOKE_OUTPUT = Path("/data/gaoya/agent-data/outputs/physv_ramp_platform_smoke")
FAMILY_KEY = "V2V_RAMP_PLATFORM"
PLATFORM_LENGTHS_M = (0.40, 0.80, 1.20, 1.60)
BASE_SEED = 2026082800


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--blender", type=Path, default=BLENDER)
    parser.add_argument("--ffmpeg", type=Path, default=FFMPEG)
    parser.add_argument("--cycles-gpu", default="5", help="Physical GPU for sequential CYCLES rendering.")
    parser.add_argument("--smoke", action="store_true", help="Run only one low-resolution physics/render smoke test.")
    parser.add_argument("--smoke-output", type=Path, default=SMOKE_OUTPUT)
    parser.add_argument("--force", action="store_true", help="Replace only the four new sample directories.")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _camera() -> CameraSpec:
    return CameraSpec(
        eye=(0.75, -5.80, 2.35),
        target=(0.40, 0.0, 0.78),
        up=(0.0, 0.0, 1.0),
        yfov_deg=45.0,
        jitter_eye_xyz=(0.0, 0.0, 0.0),
        jitter_target_xyz=(0.0, 0.0, 0.0),
        jitter_fov_deg=0.0,
        hdri_key="studio_soft",
    )


def build_export_case(platform_length_m: float, index: int) -> ExportCase:
    """Build one deterministic case with a single controlled geometry value."""
    platform_length_m = float(platform_length_m)
    if platform_length_m not in PLATFORM_LENGTHS_M:
        raise ValueError(f"unsupported platform length {platform_length_m}")

    table_height = 1.0
    table_top_hz = 0.04
    table_top_z = table_height - table_top_hz
    table_top_hx = 0.75
    table_top_hy = 0.55
    ramp_length = 1.20
    ramp_angle_deg = 25.0
    theta = math.radians(ramp_angle_deg)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    ramp_hx = ramp_length * 0.5
    ramp_hy = 0.55
    ramp_hz = 0.04

    # The lower underside corner of the incline is exactly at x=0,z=1.0.
    ramp_center_x = -ramp_hx * cos_theta + ramp_hz * sin_theta
    ramp_center_z = table_height + ramp_hx * sin_theta + ramp_hz * cos_theta

    support_hx = 0.08
    support_local_x = -0.45
    support_center_x = ramp_center_x + support_local_x * cos_theta - ramp_hz * sin_theta
    board_underside_at_support = (
        ramp_center_z - support_local_x * sin_theta - ramp_hz * cos_theta
    )
    support_top_z = board_underside_at_support - support_hx * math.tan(theta)
    support_height = support_top_z - table_height
    if support_height <= 0:
        raise ValueError(f"invalid ramp support height: {support_height}")

    block_hx, block_hy, block_hz = 0.16, 0.14, 0.12
    block_local_x = -0.42
    block_local_z = ramp_hz + block_hz
    block_position = (
        ramp_center_x + block_local_x * cos_theta + block_local_z * sin_theta,
        0.0,
        ramp_center_z - block_local_x * sin_theta + block_local_z * cos_theta,
    )

    objects = [
        _object(
            name="block_0",
            family_key="wood_block",
            shape="box",
            size={"hx": block_hx, "hy": block_hy, "hz": block_hz},
            material_key="wood_red",
            position=block_position,
            dynamic=True,
            mass=2.5,
            friction=0.10,
            restitution=0.08,
            role="dynamic",
            orientation=(0.0, ramp_angle_deg, 0.0),
            linear_damping=0.02,
            angular_damping=0.04,
            metadata={"appearance_group": "ramp_platform_block_v1"},
        ),
        _object(
            name="table_top_0",
            family_key="table_top",
            shape="box",
            size={"hx": table_top_hx, "hy": table_top_hy, "hz": table_top_hz},
            material_key="wood_dark",
            position=(-0.75, 0.0, table_top_z),
            dynamic=False,
            mass=0.0,
            friction=0.82,
            restitution=0.02,
            role="anchored_fixture",
            metadata={"appearance_group": "ramp_platform_table_surface_v1"},
        ),
        _object(
            name="horizontal_platform_0",
            family_key="table_top",
            shape="box",
            size={"hx": platform_length_m * 0.5, "hy": table_top_hy, "hz": table_top_hz},
            material_key="wood_dark",
            position=(platform_length_m * 0.5, 0.0, table_top_z),
            dynamic=False,
            mass=0.0,
            friction=0.82,
            restitution=0.02,
            role="anchored_fixture",
            metadata={"appearance_group": "ramp_platform_table_surface_v1"},
        ),
        _object(
            name="incline_board_0",
            family_key="incline_board",
            shape="box",
            size={"hx": ramp_hx, "hy": ramp_hy, "hz": ramp_hz},
            material_key="wood_plywood",
            position=(ramp_center_x, 0.0, ramp_center_z),
            dynamic=False,
            mass=0.0,
            friction=0.82,
            restitution=0.02,
            role="anchored_fixture",
            orientation=(0.0, ramp_angle_deg, 0.0),
            metadata={"appearance_group": "ramp_platform_incline_v1"},
        ),
    ]

    # Four fixed legs make the elevated table physically explicit.  The right
    # pair moves only because the controlled platform becomes longer.
    leg_height = table_height - 2.0 * table_top_hz
    for side_name, x in (("left", -1.32), ("right", platform_length_m - 0.12)):
        for y_index, y in enumerate((-0.42, 0.42)):
            objects.append(
                _object(
                    name=f"table_leg_{side_name}_{y_index}",
                    family_key="table_leg",
                    shape="box",
                    size={"hx": 0.08, "hy": 0.08, "hz": leg_height * 0.5},
                    material_key="concrete_painted",
                    position=(x, y, leg_height * 0.5),
                    dynamic=False,
                    mass=0.0,
                    friction=0.80,
                    restitution=0.02,
                    role="anchored_fixture",
                    metadata={"appearance_group": "ramp_platform_table_legs_v1"},
                )
            )

    for index_side, y in enumerate((-0.35, 0.35)):
        objects.append(
            _object(
                name=f"ramp_support_{index_side}",
                family_key="incline_riser",
                shape="box",
                size={"hx": support_hx, "hy": 0.10, "hz": support_height * 0.5},
                material_key="concrete_painted",
                position=(support_center_x, y, table_height + support_height * 0.5),
                dynamic=False,
                mass=0.0,
                friction=0.82,
                restitution=0.02,
                role="anchored_fixture",
                metadata={"appearance_group": "ramp_platform_ramp_support_v1"},
            )
        )

    case_id = f"v2v_ramp_platform_l{int(round(platform_length_m * 100)):03d}"
    metadata = {
        "controlled_variable": "horizontal_platform_length_m",
        "horizontal_platform_length_m": platform_length_m,
        "ramp_angle_deg": ramp_angle_deg,
        "ramp_length_m": ramp_length,
        "table_height_m": table_height,
        "table_top_z_m": table_height,
        "ramp_low_x_m": 0.0,
        "ramp_exit_x_m": 0.0,
        "platform_edge_x_m": platform_length_m,
        "block_half_x_m": block_hx,
        "released_from_rest": True,
        "landing_surface": "ground_floor",
        "gt_responses": ["ramp_exit_time_s", "platform_departure_time_s", "landing_point_m"],
        "ground_truth_event_definitions": {
            "ramp_exit": "first video frame where block_0 center x >= 0.0 m, the lower edge of the incline",
            "platform_departure": "first video frame where block_0 center x > platform edge + half block length",
            "landing": "first block_0 contact with the ground after platform departure",
            "landing_point": "block_0 world-coordinate center at the landing frame",
        },
        "scene_control_contract": (
            "block, incline, table height, camera, materials, friction, restitution and timestep stay fixed; "
            "only horizontal platform length and the right leg x position change"
        ),
        "physics_sub_steps": 8,
    }
    blueprint = _blueprint(
        family_key=FAMILY_KEY,
        sample_key=case_id,
        title=f"Incline to horizontal platform ({platform_length_m:.2f} m)",
        description=(
            f"A wooden block is released from rest on a {ramp_angle_deg:.0f} degree incline, "
            f"crosses a {platform_length_m:.2f} m horizontal platform, and falls from its edge."
        ),
        objects=objects,
        camera=_camera(),
        surface_key="residential_wood_floor",
        tags=("ramp_platform", "gravity_release", "fixed_ramp", f"platform_length_{platform_length_m:.2f}"),
        metadata=metadata,
    )
    return ExportCase(
        case_id=case_id,
        source_group="v2v_ramp_platform",
        family_key=FAMILY_KEY,
        task_type="incline_to_platform",
        title=blueprint.title,
        description=blueprint.description,
        analysis_question=(
            "How does the horizontal platform length affect the time at which the block leaves its support "
            "and its subsequent landing point?"
        ),
        controlled_variable="horizontal_platform_length_m",
        controlled_value=platform_length_m,
        controlled_value_label=f"{platform_length_m:.2f} m",
        units="m",
        event_rule="block center crosses the lower incline edge at x=0.0 m",
        blueprint=blueprint,
        seed=BASE_SEED + index,
        scene_style=SCENE_STYLE,
        v2v_case=None,
        taxonomy="Scene",
    )


def _load_state_for_smoke(render_root: Path, contact_root: Path, case_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    state_path = render_root / "meta" / f"{case_id}_states.npz"
    arrays = np.load(state_path, allow_pickle=False)
    names = [str(value) for value in arrays["object_names"]]
    contacts = json.loads((contact_root / "contacts.json").read_text(encoding="utf-8"))
    state = {
        "names": names,
        "frame_times": np.asarray(arrays["frame_times"]),
        "positions": np.asarray(arrays["positions"]),
        "velocities": np.asarray(arrays["linear_velocities"]),
        "angular_velocities": np.asarray(arrays["angular_velocities"]),
        "quats": np.asarray(arrays["quats"]),
    }
    return state, {"contacts": contacts}


def run_smoke(args: argparse.Namespace) -> None:
    output = args.smoke_output.expanduser().resolve()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    case = build_export_case(PLATFORM_LENGTHS_M[1], 1)
    manifest = render_blueprint_case(
        blueprint=case.blueprint,
        seed=case.seed,
        output_root=output / "render",
        width=320,
        height=180,
        scene_style=case.scene_style,
        export_instance_masks=True,
        preserve_states=True,
        ground_truth_output_dir=output,
    )
    state, extra = _load_state_for_smoke(output / "render", output, case.case_id)
    positions = state["positions"]
    names = state["names"]
    block_index = names.index("block_0")
    contacts = extra["contacts"]
    contact_pairs = sorted({f"{item.get('obj_a')}×{item.get('obj_b')}" for item in contacts if item.get("contacts")})
    x = positions[:, block_index, 0]
    exit_hits = np.flatnonzero(x >= 0.0)
    depart_hits = np.flatnonzero(x > case.controlled_value + case.blueprint.metadata["block_half_x_m"])
    ground_hits = [
        int(item["frame"])
        for item in contacts
        if {item.get("obj_a"), item.get("obj_b")} == {"block_0", "ground"}
        and item.get("contacts")
    ]
    print(json.dumps({
        "status": "smoke_passed",
        "case_id": case.case_id,
        "frames": int(len(x)),
        "resolution": [320, 180],
        "initialization_qa": manifest.get("initialization_qa"),
        "contact_pairs": contact_pairs,
        "ramp_exit_frame": int(exit_hits[0]) if len(exit_hits) else None,
        "platform_departure_frame": int(depart_hits[0]) if len(depart_hits) else None,
        "ground_contact_frame": min(ground_hits) if ground_hits else None,
        "final_block_position_m": positions[-1, block_index].round(5).tolist(),
        "output": str(output),
    }, ensure_ascii=False, indent=2))


def finalize_metadata(sample_dir: Path) -> dict[str, Any]:
    metadata_path = sample_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["render_variant"] = "strict_cycles_pbr_896x512"
    metadata["conditioning"]["target_video"] = "videos/rgb_cycles.mp4"
    metadata["conditioning"]["first_event_rule"] = metadata["scenario_spec"]["ground_truth_event_definitions"]["ramp_exit"]
    observations = derive_caption_observations(sample_dir, metadata)
    details = observations.get("details", {})
    event_payload = {
        "ramp_exit": {
            "frame": details.get("ramp_exit_frame"),
            "time_s": details.get("ramp_exit_time_s"),
        },
        "platform_departure": {
            "frame": details.get("platform_departure_frame"),
            "time_s": details.get("platform_departure_time_s"),
        },
        "landing": {
            "frame": details.get("landing_frame"),
            "time_s": details.get("landing_time_s"),
            "point_m": details.get("landing_point_m"),
        },
    }
    metadata["ground_truth_events"] = event_payload
    metadata["scenario_spec"]["ground_truth_events"] = event_payload
    metadata["gt_response"] = {
        "support_departure_time_s": details.get("platform_departure_time_s"),
        "ramp_exit_time_s": details.get("ramp_exit_time_s"),
        "landing_point_m": details.get("landing_point_m"),
        "definitions": metadata["scenario_spec"]["ground_truth_event_definitions"],
    }
    attach_caption_metadata(metadata, observations)
    _write_json(metadata_path, metadata)

    manifest_path = sample_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "schema_version": "physv_v2v_rigidbench_style_v2",
            "caption_observations": observations,
            "video_spec": {"width": 896, "height": 512, "fps": 30.0, "source_frame_count": 90},
            "render_variant": "strict_cycles_pbr_896x512",
            "files": {
                "reference_video": "videos/rgb_cycles.mp4",
                "context8": "context/context8_cycles.mp4",
                "context16": "context/context16_cycles.mp4",
                "first_frame": "frames/00000.png",
                "simulator_trajectory": "raw/trajectories.npz",
                "contacts": "contacts.json",
                "physics_supervision": "physics_supervision.npz",
                "aligned_truth": "../../truth/cases/" + sample_dir.name,
            },
            "ground_truth_events": event_payload,
        }
    )
    _write_json(manifest_path, manifest)
    captions = sample_dir / "captions"
    (captions / "caption_specific.txt").write_text(metadata["captions"]["specific"]["text"] + "\n", encoding="utf-8")
    (captions / "caption_abstract.txt").write_text(metadata["captions"]["abstract"]["text"] + "\n", encoding="utf-8")
    _write_json(
        captions / "captions.json",
        {"schema_version": metadata["caption_schema_version"], "source": "metadata.json", "specific": metadata["captions"]["specific"]["text"], "abstract": metadata["captions"]["abstract"]["text"]},
    )
    _write_json(
        sample_dir / "export_summary.json",
        {
            "sample_id": sample_dir.name,
            "frame_count": metadata["simulation"]["frame_count"],
            "event_first_frame": observations.get("event_frame"),
            "ground_truth_events": event_payload,
        },
    )
    return metadata


def render_cycles(args: argparse.Namespace, cases: list[ExportCase]) -> None:
    case_ids = [case.case_id for case in cases]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.cycles_gpu)
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/run_physv_cycles_previews.py"),
        *case_ids,
        "--dataset-root", str(args.dataset_root),
        "--cache-root", str(CYCLES_CACHE),
        "--blender", str(args.blender),
        "--ffmpeg", str(args.ffmpeg),
        "--gpu", str(args.cycles_gpu),
        "--width", "896",
        "--height", "512",
        "--samples", "32",
        "--engine", "CYCLES",
    ]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env, cwd=PROJECT_ROOT)


def generate_aligned_truth(args: argparse.Namespace, cases: list[ExportCase]) -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/generate_physv_cycles_aligned_truth.py"),
        "--dataset-root", str(args.dataset_root),
        "--output-root", str(args.dataset_root / "truth"),
        "--gpus", str(args.cycles_gpu),
    ]
    for case in cases:
        command.extend(["--sample-id", case.case_id])
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def materialize_selected_adapter_videos(root: Path, cases: list[ExportCase]) -> None:
    """Match the strict package convention: adapter video is a real file, not a link."""
    for case in cases:
        video = root / "truth" / "cases" / case.case_id / "rigidbench" / "video.mp4"
        if not video.is_symlink():
            continue
        source = video.resolve()
        video.unlink()
        shutil.copy2(source, video)


def write_test_json(sample_dir: Path, metadata: dict[str, Any]) -> None:
    root = sample_dir.parents[1]
    truth = root / "truth" / "cases" / sample_dir.name
    captions = metadata["captions"]
    payload = {
        "source_video": str(sample_dir / "videos/rgb_cycles.mp4"),
        "input_caption": captions["abstract"]["text"],
        "input_caption_specific": captions["specific"]["text"],
        "input_caption_abstract": captions["abstract"]["text"],
        "input_video": str(sample_dir / "context/context8_cycles.mp4"),
        "input_video_8f": str(sample_dir / "context/context8_cycles.mp4"),
        "input_video_16f": str(sample_dir / "context/context16_cycles.mp4"),
        "input_image": str(sample_dir / "frames/00000.png"),
        "sample_id": sample_dir.name,
        "dataset": metadata["dataset"],
        "schema_version": metadata["schema_version"],
        "task_type": metadata["task_type"],
        "split": metadata["split"],
        "source_group": metadata["source_group"],
        "family_key": metadata["family_key"],
        "taxonomy": metadata["taxonomy"],
        "taxonomy_definition": metadata["taxonomy_definition"],
        "title": metadata["title"],
        "control": metadata["control"],
        "conditioning": metadata["conditioning"],
        "metadata_json": str(sample_dir / "metadata.json"),
        "manifest_json": str(sample_dir / "manifest.json"),
        "captions_json": str(sample_dir / "captions/captions.json"),
        "physics_supervision_npz": str(sample_dir / "physics_supervision.npz"),
        "physics_supervision_summary": str(sample_dir / "physics_supervision.json"),
        "contacts_json": str(sample_dir / "contacts.json"),
        "trajectories_npz": str(sample_dir / "raw/trajectories.npz"),
        "masks_npz": str(truth / "rigidbench/masks.npz"),
        "depth_npz": str(truth / "rigidbench/depth.npz"),
        "frame_counts": {"source_video": 90, "input_video_context8": 8, "input_video_16f": 16},
        "video_spec": {"width": 896, "height": 512, "fps": 30.0, "source_frame_count": 90},
        "caption_variant": "observed_metadata_caption",
        "render_variant": "strict_cycles_pbr_896x512",
        "conditioning_note": "The strict benchmark reads context8_cycles.mp4; the continuation reference is rgb_cycles.mp4.",
        "caption_schema_version": metadata["caption_schema_version"],
        "caption_observations": metadata["caption_observations"],
        "ground_truth_events": metadata["ground_truth_events"],
    }
    _write_json(root / "testjsons/v2v_jsons/physv_v2v_0819_all_cycles" / f"{sample_dir.name}.json", payload)


def update_test_lists(root: Path) -> None:
    json_dir = root / "testjsons/v2v_jsons/physv_v2v_0819_all_cycles"
    paths = [str(path) for path in sorted(json_dir.glob("*.json"))]
    if len(paths) != 74:
        raise RuntimeError(f"expected 74 strict JSON cases after addition, got {len(paths)}")
    for suffix in ("ctx8", "ctx8_description_no_event_timing"):
        name = f"physv_v2v_0819_all_cycles_test74_{suffix}.txt"
        (root / "testjsons" / name).write_text("\n".join(paths) + "\n", encoding="utf-8")


def update_indexes(root: Path) -> None:
    samples = sorted(path for path in (root / "samples").iterdir() if path.is_dir() and (path / "metadata.json").is_file())
    records = [validate_case(root, path.name) for path in samples]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["sample_count"] = len(records)
    manifest["samples"] = records
    _write_json(root / "manifest.json", manifest)

    dataset_meta = json.loads((root / "dataset_meta.json").read_text(encoding="utf-8"))
    dataset_meta["records"] = records
    source_selection = str(dataset_meta.get("source_selection", ""))
    addition = "4 V2V_RAMP_PLATFORM incline-to-horizontal-platform cases controlling horizontal platform length"
    if addition not in source_selection:
        dataset_meta["source_selection"] = source_selection.rstrip("; ") + "; " + addition + "."
    _write_json(root / "dataset_meta.json", dataset_meta)
    (root / "benchmark.yaml").write_text(
        "name: physv_v2v_0819_strict_cycles\nversion: 1\nreference_video: rgb_cycles.mp4\nwidth: 896\nheight: 512\nfps: 30\nframes: 90\ncases: 74\nprotocol: rigidbench-style-native-cycles\nofficial_rigidbench: false\n",
        encoding="utf-8",
    )

    readme = root / "README.md"
    text = readme.read_text(encoding="utf-8")
    text = text.replace("- cases: 70", "- cases: 74")
    marker = "## V2V_RAMP_PLATFORM · 斜面—水平平台长度"
    if marker not in text:
        text += f"""

{marker}

该正式 active family 由四个 case 组成，保持木块、1.20 m / 25° 斜面、桌面高度 1.00 m、相机、材质和 PyBullet 参数不变，仅改变水平平台长度：0.40、0.80、1.20、1.60 m。木块从静止释放，沿斜面滑下并穿过水平平台后离开平台边缘。

GT 响应定义：

| 响应 | 定义 | 文件位置 |
| --- | --- | --- |
| `ramp_exit_time_s` | 木块中心首次满足 `x >= 0.0 m`，即斜面低端参考线 | `samples/*/metadata.json` → `ground_truth_events.ramp_exit` |
| `platform_departure_time_s` | 木块中心首次满足 `x > platform_edge_x_m + block_half_x_m`，即完全越过平台边缘 | `samples/*/metadata.json` → `ground_truth_events.platform_departure` |
| `landing_point_m` | 平台离开后首次与 PyBullet ground 接触时的木块世界坐标中心 | `samples/*/metadata.json` → `ground_truth_events.landing.point_m` |

正式测试集合使用 `testjsons/physv_v2v_0819_all_cycles_test74_ctx8.txt`；历史 `test70` 列表保持不变，便于复现实验。
"""
    readme.write_text(text, encoding="utf-8")


def update_truth_manifest(root: Path) -> None:
    truth_root = root / "truth"
    path = truth_root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records = []
    for sample in sorted((root / "samples").iterdir()):
        case_dir = truth_root / "cases" / sample.name
        meta_path = case_dir / "truth_metadata.json"
        if not meta_path.is_file():
            raise RuntimeError(f"aligned truth missing for {sample.name}")
        truth_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        records.append(
            {
                "sample_id": sample.name,
                "status": "complete",
                "gpu": str(manifest.get("gpus", ["5"])[0]),
                "resolution": [896, 512],
                "frame_count": 90,
                "dynamic_objects": truth_meta.get("dynamic_objects", []),
                "output_dir": str(case_dir),
                "log": str(truth_root / "logs" / f"{sample.name}.log"),
            }
        )
    manifest["records"] = records
    manifest["selected_sample_count"] = 4
    manifest["total_sample_count"] = len(records)
    manifest["status"] = "complete"
    manifest["status_counts"] = {"complete": len(records)}
    _write_json(path, manifest)


def cleanup_staging(sample_dir: Path) -> None:
    """Keep only the strict active package; targets are explicit generated files."""
    files = (
        "raw/source_video.mp4",
        "raw/states_xyzw.npz",
        "raw/instance_ids.npz",
        "raw/depth.npz",
        "raw/simulator_render_metadata.json",
        "videos/rgb.mp4",
        "videos/masks.mp4",
        "videos/depth.mp4",
        "videos/trajectory.mp4",
        "videos/contacts.mp4",
        "context/context8.mp4",
        "context/context16.mp4",
        "meta.json",
    )
    for relative in files:
        path = sample_dir / relative
        if path.is_symlink() or path.is_file():
            path.unlink()
    for relative in ("raw/frames", ".render"):
        path = sample_dir / relative
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)


def update_registry(root: Path) -> None:
    path = root / "refine" / "CASE_REGISTRY.md"
    text = path.read_text(encoding="utf-8") if path.is_file() else "# strict CYCLES refine：新增/修改 Case 登记表\n"
    marker = "`V2V_RAMP_PLATFORM`"
    if marker not in text:
        text += """

## 正式 active 新增场景（2026-08-28）

虽然该 family 不是 refine 变体，但按登记规则在这里保留来源和 GT 入口，避免后续新增场景失去可追溯性。

| ID | Case family | 类型 | 变更范围 | 来源 / 控制变量 | GT 响应 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| `A001` | `V2V_RAMP_PLATFORM` | 新增 | 新增桌面、25°斜面、水平平台和木块释放场景；四个长度条件，其他参数固定 | `scripts/add_ramp_platform_cases.py`；`horizontal_platform_length_m ∈ {0.40, 0.80, 1.20, 1.60}` | CYCLES 对齐 mask/depth/trajectory；PyBullet `contacts.json`；斜面离开时间、平台离开时间、落点 | `completed` |
"""
        path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.dataset_root.expanduser().resolve()
    if "4" == str(args.cycles_gpu):
        raise ValueError("GPU 4 is reserved and cannot be used")
    if not args.blender.is_file():
        raise FileNotFoundError(args.blender)
    if not args.ffmpeg.is_file():
        raise FileNotFoundError(args.ffmpeg)
    if args.smoke:
        run_smoke(args)
        return

    cases = [build_export_case(length, index) for index, length in enumerate(PLATFORM_LENGTHS_M)]
    for case in cases:
        sample_dir = root / "samples" / case.case_id
        if sample_dir.exists():
            if not args.force:
                raise FileExistsError(f"{sample_dir} already exists; use --force to replace this new case only")
            shutil.rmtree(sample_dir)
        sample_dir.mkdir(parents=True, exist_ok=True)
        print(f"[stage] {case.case_id}", flush=True)
        render_manifest = render_blueprint_case(
            blueprint=case.blueprint,
            seed=case.seed,
            output_root=sample_dir / ".render",
            width=1280,
            height=720,
            scene_style=case.scene_style,
            export_instance_masks=True,
            preserve_states=True,
            ground_truth_output_dir=sample_dir,
        )
        _package_case(case=case, sample_dir=sample_dir, render_manifest=render_manifest, width=1280, height=720)
        _validate_sample(sample_dir)
        finalize_metadata(sample_dir)

    print("[cycles] rendering four cases sequentially", flush=True)
    render_cycles(args, cases)
    for case in cases:
        sample_dir = root / "samples" / case.case_id
        regenerate_contexts(sample_dir, args.ffmpeg)
        write_first_frame(sample_dir)
        synchronize_cycle_config(sample_dir)
        finalize_metadata(sample_dir)

    print("[truth] generating CYCLES-aligned GT", flush=True)
    had_adapter_dataset = (root / "truth/rigidbench_dataset").exists()
    generate_aligned_truth(args, cases)
    materialize_selected_adapter_videos(root, cases)
    update_truth_manifest(root)
    if not had_adapter_dataset and (root / "truth/rigidbench_dataset").exists():
        shutil.rmtree(root / "truth/rigidbench_dataset")

    for case in cases:
        sample_dir = root / "samples" / case.case_id
        metadata = finalize_metadata(sample_dir)
        write_test_json(sample_dir, metadata)
        cleanup_staging(sample_dir)
    update_test_lists(root)
    update_indexes(root)
    update_registry(root)
    visualization_script = PROJECT_ROOT / "scripts" / "build_strict_dataset_visualization.py"
    visualization_command = [
        sys.executable,
        str(visualization_script),
        "--dataset-root", str(root),
    ]
    print("+", " ".join(visualization_command), flush=True)
    subprocess.run(visualization_command, check=True, cwd=PROJECT_ROOT)
    print(json.dumps({"status": "complete", "cases_added": [case.case_id for case in cases], "dataset_root": str(root)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
