"""Generate concise English descriptions from PhysV V2V physics truth.

The generator re-derives caption observations from trajectories and contacts,
then combines the existing observed-outcome templates with a motion-relevant
scene description.  It does not call a language model and never modifies the
exported dataset files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from .caption_observations_0819 import derive_caption_observations
from .caption_templates_0819 import build_caption_bundle


DATASET_ROOT = Path("/data/gaoya/AAA_test_video/physv_v2v_0819")
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/physv_v2v_0819_truth_descriptions"
)
SCHEMA_VERSION = "physv_truth_description_en_v1"


SCENE_DESCRIPTIONS = {
    "F11": "A horizontal table provides a supporting surface for a rolling object, with a free edge above the floor.",
    "F12": "An inclined ramp supports a sliding block and leads toward a floor at its lower end.",
    "F12_RAMP_LENGTH": "An inclined ramp with a supported high end leads toward a floor at its lower end.",
    "V2V_GAP": "Two horizontal platforms are separated by a visible gap, with the floor below the platforms.",
    "V2V_OBSTACLE": "A horizontal floor guides a rolling object toward a fixed barrier.",
    "V2V_OBSTACLE_SIZE": "A horizontal floor guides a rolling object toward a fixed barrier.",
    "V2V_BOWL": "A curved bowl surface supports an object and guides its motion along the inner wall.",
    "V2V_PENDULUM": "A suspended bob moves through open space below a fixed suspension point.",
    "V2V_PENDULUM_CABINET": "A suspended bob moves through open space toward a tall cabinet that can obstruct its path.",
    "V2V_SEESAW": "A hinged board supports a load and can rotate about its central support.",
    "V2V_DOMINO": "A floor supports a row of upright dominoes arranged along the path of a moving trigger object.",
    "SCENE_PUCK_BARRIER": "A horizontal floor guides a sliding puck toward a fixed barrier.",
    "SCENE_DOOR_FRAME": "A floor guides a moving crate toward a wall opening bounded by side posts and a top frame.",
    "SCENE_DOOR_FRAME_BALL": "A floor guides a moving ball toward a wall opening bounded by side posts and a top frame.",
}


def _scene_description(metadata: Mapping[str, object]) -> str:
    family_key = str(metadata.get("family_key", ""))
    task_type = str(metadata.get("task_type", ""))
    if family_key in SCENE_DESCRIPTIONS:
        return SCENE_DESCRIPTIONS[family_key]
    if task_type == "table_rolloff":
        return SCENE_DESCRIPTIONS["F11"]
    if task_type in {"incline_release", "incline_length_release"}:
        return SCENE_DESCRIPTIONS["F12"]
    return "The scene contains rigid objects and supporting structures that constrain the moving object's path."


def _neutralize_visual_adjectives(text: str) -> str:
    """Align template wording with the motion-only description policy."""

    replacements = (
        ("red trigger ball", "trigger ball"),
        ("red wooden block", "wooden block"),
        ("blue rubber ball", "ball"),
        ("red ball", "ball"),
        ("fixed blue barrier", "fixed barrier"),
        ("blue barrier", "barrier"),
        ("An ice puck", "A puck"),
        ("ice puck", "puck"),
        ("larger red bob", "bob"),
        ("red pendulum bob", "pendulum bob"),
        ("thick, high-contrast rope", "rope"),
        ("low-friction floor", "floor"),
        ("ground", "floor"),
        ("Ground", "Floor"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _format_description(
    metadata: Mapping[str, object],
    observations: Mapping[str, object],
    *,
    specific: bool,
) -> str:
    caption_metadata = dict(metadata)
    caption_metadata["caption_observations"] = dict(observations)
    bundle = build_caption_bundle(caption_metadata)
    motion = _neutralize_visual_adjectives(bundle["specific" if specific else "abstract"])

    event_frame = observations.get("event_frame")
    event_time = observations.get("event_time_s")
    if isinstance(event_frame, int) and event_frame >= 0:
        if isinstance(event_time, (int, float)):
            motion += f" The main recorded transition occurs around frame {event_frame} ({float(event_time):.2f} s)."
        else:
            motion += f" The main recorded transition occurs around frame {event_frame}."

    return f"Scene:\n{_scene_description(metadata)}\n\nPhysical motion:\n{motion}"


def _sample_dirs(dataset_root: Path, case_ids: list[str] | None) -> list[Path]:
    available = {path.name: path for path in (dataset_root / "samples").iterdir() if path.is_dir()}
    selected = sorted(available) if not case_ids else case_ids
    missing = [case_id for case_id in selected if case_id not in available]
    if missing:
        raise FileNotFoundError(f"Unknown case IDs: {', '.join(missing)}")
    return [available[case_id] for case_id in selected]


def generate_case(sample_dir: Path, *, specific: bool) -> dict[str, object]:
    metadata_path = sample_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    observations = derive_caption_observations(sample_dir, metadata)
    description = _format_description(metadata, observations, specific=specific)
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": str(metadata.get("sample_id", sample_dir.name)),
        "title": metadata.get("title"),
        "control": metadata.get("control", {}),
        "video": str(sample_dir / "videos" / "rgb_cycles.mp4"),
        "context8_video": str(sample_dir / "context" / "context8_cycles.mp4"),
        "truth_sources": {
            "metadata": str(metadata_path),
            "physics_supervision": str(sample_dir / "physics_supervision.npz"),
            "contacts": str(sample_dir / "contacts.json"),
        },
        "observations": observations,
        "description": description,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument(
        "--specific",
        action="store_true",
        help="Expose the controlled variable in the motion sentence.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sample_dirs = _sample_dirs(args.dataset_root.resolve(), args.case_ids)
    output_root = args.output_root.resolve()
    cases_root = output_root / "cases"
    cases_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for sample_dir in sample_dirs:
        record = generate_case(sample_dir, specific=args.specific)
        records.append(record)
        case_path = cases_root / f"{record['case_id']}.txt"
        case_path.write_text(str(record["description"]) + "\n", encoding="utf-8")

    jsonl_path = output_root / "truth_descriptions.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_root": str(args.dataset_root.resolve()),
        "output_root": str(output_root),
        "caption_variant": "specific" if args.specific else "abstract",
        "case_count": len(records),
        "jsonl": str(jsonl_path),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
