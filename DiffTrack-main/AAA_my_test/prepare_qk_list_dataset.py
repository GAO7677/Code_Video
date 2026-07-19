#!/usr/bin/env python3
"""Convert a newline-delimited list of generation JSONs into Q/K case manifests."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-list", type=Path, required=True)
    parser.add_argument("--dataset-tag", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def expected_object_count(caption: str) -> int:
    normalized = caption.strip().lower()
    if re.match(r"^f1\s+sample\s+\d+\s+industrial rigid body simulation\b", normalized):
        return 1
    return 2


def object_phrases(json_path: Path, caption: str) -> list[str]:
    normalized = caption.strip().lower()
    match = re.match(
        r"^f[1-5]\s+sample\s+\d+\s+industrial rigid body simulation\s+(.+)$",
        normalized,
    )
    if match:
        shapes = match.group(1).split()
        distinct = []
        for shape in shapes:
            if shape not in distinct:
                distinct.append(shape)
        return distinct[: expected_object_count(caption)]
    stem = json_path.stem.lower()
    for marker, phrases in PHYSICIQ_OBJECTS.items():
        if marker in stem:
            return list(phrases)
    if "physiciq_025_solid_mechanics_0002_perspective-center_trimmed" in stem:
        return ["brown tennis ball", "orange block"]
    raise ValueError(f"no audited object phrases for {json_path.name}")


def safe_stem(value: str, limit: int = 84) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return cleaned[:limit] or "sample"


def main() -> None:
    args = parse_args()
    input_list = args.input_list.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    raw_paths = [line.strip() for line in input_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    paths = list(dict.fromkeys(raw_paths))
    cases = []
    failures = []
    for index, raw_path in enumerate(paths):
        try:
            json_path = Path(raw_path).expanduser().resolve()
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            context_video = Path(payload["input_video"]).expanduser().resolve()
            source_video = Path(payload["source_video"]).expanduser().resolve()
            caption = str(payload["input_caption"]).strip()
            if not context_video.is_file() or not source_video.is_file() or not caption:
                raise ValueError("missing context video, source video, or caption")
            case_key = (
                f"case_{safe_stem(args.dataset_tag, 24)}_{index:03d}_"
                f"{safe_stem(json_path.stem)}"
            )
            phrases = object_phrases(json_path, caption)
            manifest = {
                "case_key": case_key,
                "object_count": len(phrases),
                "base": {
                    "video": str(context_video),
                    "source_video": str(source_video),
                    "caption": caption,
                    "input_json": str(json_path),
                    "object_phrases": phrases,
                },
            }
            case_dir = output / "cases" / case_key
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "case_manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            cases.append(manifest)
        except Exception as error:
            failures.append({"input": raw_path, "error": f"{type(error).__name__}: {error}"})
    output.mkdir(parents=True, exist_ok=True)
    audit = {
        "input_list": str(input_list),
        "raw_lines": len(raw_paths),
        "unique_inputs": len(paths),
        "prepared_cases": len(cases),
        "failures": failures,
        "cases": cases,
    }
    (output / "manifest.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(cases)}/{len(paths)} cases under {output}", flush=True)
    if failures:
        raise SystemExit(f"failed to prepare {len(failures)} inputs")


PHYSICIQ_OBJECTS = {
    "ball-behind-rotating-paper": ["tennis ball", "piece of cardstock"],
    "napkin-soak": ["paper towel", "shallow dish"],
    "paint-on-glass": ["paintbrush", "clear acrylic sheet"],
    "siphon": ["lit matchsticks", "glass jar"],
    "ball-and-block-fall": ["brown tennis ball", "orange block"],
    "ball-in-basket": ["orange basketball", "black plastic crate"],
    "ball-rolls-on-glass": ["blue tennis ball", "piece of clear glass"],
    "block-domino": ["wooden stick", "colorful wooden blocks"],
    "cut-orange": ["tangerine", "knife"],
    "silk-cover": ["teapot", "silk fabric"],
    "solid-ball-peakaboo": ["woven basket", "orange tennis ball"],
    "two-balls-pass": ["blue tennis ball", "yellow tennis ball"],
    "unstable-block-stack": ["blue wooden block", "yellow block"],
    "weight-on-ceramic": ["kettlebell", "yellow ceramic mug"],
    "weight-on-paper": ["kettlebell", "white styrofoam cup"],
    "weight-on-pillow": ["kettlebell", "green piece of paper"],
    "weight-protects-duck": ["grey tennis ball", "yellow rubber duck"],
    "lit-candle": ["left red candle", "right red candle"],
    "match-blows-balloon": ["black balloon", "lit matchstick"],
    "paper-smoke": ["folded paper", "glass cutting board"],
    "blow-balloon": ["black balloon", "air pump hose"],
    "domino-in-juice": ["white domino", "blue mug"],
    "fill-glass-red-drink": ["beverage dispenser", "clear glass"],
    "glass-stays-same": ["beverage dispenser", "glass"],
    "juice-in-water": ["beverage dispenser", "glass"],
    "liquid-on-duck": ["yellow rubber duck", "beverage dispenser"],
    "liquid-overfill": ["dispenser", "glass"],
    "perspective-center_trimmed-match": ["lit match", "glass of water"],
    "paper-fall-water": ["crumpled paper", "bowl"],
    "paper-in-water": ["crumpled white paper", "tall glass"],
    "potato-in-water": ["potato", "tall glass"],
    "water-in-juice": ["beverage dispenser", "glass"],
    "magnet-domino": ["magnet", "white domino"],
    "magnet-wrench": ["magnet", "metal wrench"],
    "light-on-block": ["blue wooden block", "black turntable"],
    "light-on-mug-block": ["yellow mug", "blue wooden block"],
    "light-on-mug": ["yellow mug", "rotating turntable"],
    "light-on-statue": ["porcelain statue", "rotating base"],
    "mirror-ball-fall": ["tennis ball", "mirror"],
    "mirror-ball-rotate": ["tennis ball", "mirror"],
    "mirror-teapot-rotate": ["teapot", "mirror"],
    "rolling-reflection": ["tennis ball", "kettlebell"],
    "ball-hits-duck": ["brown tennis ball", "yellow rubber duck"],
    "ball-hits-nothing": ["orange ball", "smaller red ball"],
    "ball-in-sand": ["tennis ball", "green kinetic sand"],
    "ball-ramp": ["yellow tennis ball", "cardboard ramp"],
    "ball-rolls-off": ["grey tennis ball", "wooden table"],
    "ball-train": ["grey tennis ball", "orange tennis ball"],
    "balls-collide": ["blue tennis ball", "yellow tennis ball"],
    "cut-paper": ["green paper", "gripping tool"],
    "dominos-with-space": ["dominoes", "wooden stick"],
    "double-cradle": ["Newtons cradle", "metal ball"],
    "duck-and-dominos": ["yellow rubber duck", "dominoes"],
    "duck-falls-in-box": ["yellow rubber duck", "dark green fabric box"],
    "duck-static": ["yellow rubber duck", "wooden table"],
    "magnet-transparent-peakaboo": ["clear acrylic box", "tennis ball"],
    "marble-run-x": ["yellow marble", "magnetic ramp"],
    "marble-run-y": ["yellow marble", "magnetic ramp"],
    "mug-breaks": ["yellow mug", "concrete brick"],
    "roll-behind-box": ["grey tennis ball", "white lampshade"],
    "roll-front-box": ["grey tennis ball", "white lampshade"],
    "roll-in-box": ["brown tennis ball", "olive green fabric box"],
    "single-cradle": ["Newtons cradle", "metal ball"],
    "smiley-ball-rotates": ["tennis ball", "black platform"],
    "stable-blocks": ["pink block", "colorful blocks"],
    "teapot-rotates": ["teapot", "rotating display"],
}


if __name__ == "__main__":
    main()
