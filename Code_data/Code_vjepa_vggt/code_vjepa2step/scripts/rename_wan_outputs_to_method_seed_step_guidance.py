#!/usr/bin/env python3
"""Rename Wan output files to method_seed_step_guidance format.

Renames completed .mp4/.json files under the configured output roots while
skipping files that are currently being written by active inference processes.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


ROOTS = [
    Path("/data/gaoya/AAA_test_video/0623/test/t2v"),
    Path("/data/gaoya/AAA_test_video/0623/test/t2v_guidance_sweep"),
]

METHODS = ("wan22_base", "lora_step000500", "lora_step001000")

PATTERN_WITH_GUIDANCE = re.compile(
    r"^(?P<prefix>.+?)_(?P<method>wan22_base|lora_step000500|lora_step001000)"
    r"_step(?P<step>\d+)_guidance(?P<guidance>[0-9]+p[0-9]+)_seed(?P<seed>\d+)$"
)

PATTERN_NO_GUIDANCE = re.compile(
    r"^(?P<prefix>.+?)_(?P<method>wan22_base|lora_step000500|lora_step001000)"
    r"_step(?P<step>\d+)_seed(?P<seed>\d+)$"
)

PATTERN_NEW = re.compile(
    r"^(?P<method>wan22_base|lora_step000500|lora_step001000)"
    r"_seed(?P<seed>\d+)_step(?P<step>\d+)_guidance(?P<guidance>[0-9]+p[0-9]+)$"
)


def get_active_output_paths() -> set[Path]:
    cmd = [
        "ps",
        "-ef",
    ]
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    active_paths: set[Path] = set()
    for line in result.stdout.splitlines():
        if "generate.py" not in line and "infer_t2v_lora.py" not in line:
            continue
        for flag in ("--save_file", "--output_video_path"):
            marker = f"{flag} "
            if marker in line:
                tail = line.split(marker, 1)[1]
                value = tail.split(" --", 1)[0].strip()
                active_paths.add(Path(value))
    return active_paths


def compute_new_stem(stem: str) -> str | None:
    if PATTERN_NEW.match(stem):
        return None
    match = PATTERN_WITH_GUIDANCE.match(stem)
    if match:
        return (
            f"{match.group('method')}_seed{match.group('seed')}"
            f"_step{match.group('step')}_guidance{match.group('guidance')}"
        )
    match = PATTERN_NO_GUIDANCE.match(stem)
    if match:
        return (
            f"{match.group('method')}_seed{match.group('seed')}"
            f"_step{match.group('step')}_guidance5p0"
        )
    return None


def update_json_payload(json_path: Path, mp4_path: Path) -> None:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    payload["output_path"] = str(mp4_path)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rename_group(mp4_path: Path, json_path: Path, new_stem: str) -> bool:
    new_mp4 = mp4_path.with_name(f"{new_stem}.mp4")
    new_json = json_path.with_name(f"{new_stem}.json")
    if mp4_path == new_mp4 and json_path == new_json:
        return False
    mp4_path.rename(new_mp4)
    json_path.rename(new_json)
    update_json_payload(new_json, new_mp4)
    print(f"[renamed] {mp4_path.name} -> {new_mp4.name}")
    return True


def main() -> None:
    active_outputs = get_active_output_paths()
    renamed = 0
    for root in ROOTS:
        if not root.exists():
            continue
        for mp4_path in sorted(root.rglob("*.mp4")):
            if mp4_path in active_outputs:
                continue
            json_path = mp4_path.with_suffix(".json")
            if not json_path.exists():
                continue
            new_stem = compute_new_stem(mp4_path.stem)
            if not new_stem:
                continue
            renamed += int(rename_group(mp4_path, json_path, new_stem))
    print(f"[done] renamed_groups={renamed}")


if __name__ == "__main__":
    main()
