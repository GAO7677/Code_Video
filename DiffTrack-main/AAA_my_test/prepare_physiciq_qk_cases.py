#!/usr/bin/env python3
"""Build lightweight manifests for the selected PhysicIQ Q/K experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_JSONS = (
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed-ball-and-block-fall_motion_to_end.json",
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed.json",
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/physicIQ_026_Solid_Mechanics_0005_perspective-center_trimmed-ball-behind-rotating-paper.json",
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed_crop_top60px.json",
)
DEFAULT_OUTPUT = Path("/data/gaoya/agent-data/datasets/physiciq_selected_qk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsons", nargs="+", default=list(DEFAULT_JSONS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def object_phrases(json_path: Path) -> list[str]:
    name = json_path.stem.lower()
    if "rotating-paper" in name:
        return ["tennis ball", "piece of cardstock"]
    return ["brown tennis ball", "orange block"]


def main() -> None:
    args = parse_args()
    root = args.output_dir.expanduser().resolve()
    cases = []
    for index, raw_path in enumerate(args.input_jsons, start=1):
        json_path = Path(raw_path).expanduser().resolve()
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        for key in ("input_video", "source_video", "input_caption"):
            if not payload.get(key):
                raise ValueError(f"{json_path}: missing {key}")
        input_video = Path(payload["input_video"]).resolve()
        source_video = Path(payload["source_video"]).resolve()
        if not input_video.is_file() or not source_video.is_file():
            raise FileNotFoundError(f"{json_path}: missing input/source video")
        case_key = f"case_physiciq_{index:02d}_{json_path.stem}"
        phrases = object_phrases(json_path)
        manifest = {
            "case_key": case_key,
            "object_count": len(phrases),
            "base": {
                "video": str(input_video),
                "source_video": str(source_video),
                "caption": str(payload["input_caption"]),
                "object_phrases": phrases,
                "input_json": str(json_path),
            },
        }
        case_dir = root / "cases" / case_key
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "case_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        cases.append(manifest)
    (root / "manifest.json").write_text(
        json.dumps({"cases": cases}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Prepared {len(cases)} cases under {root}")


if __name__ == "__main__":
    main()
