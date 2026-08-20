#!/usr/bin/env python3
"""Register a completed PhysRVG result root in the existing 8844 pages."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path(
    "/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_hub/"
    "physrvg-physiciq-lora-ablation/reference_models.json"
)
DEFAULT_INPUT_LIST = Path(
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-key", required=True)
    parser.add_argument("--method-label", required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--video-subdir", required=True)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--input-json-list", type=Path, default=DEFAULT_INPUT_LIST)
    parser.add_argument("--inference-steps", type=int, default=40)
    return parser.parse_args()


def read_cases(path: Path) -> list[Path]:
    return [
        Path(line.strip()).expanduser().resolve()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    result_root = args.result_root.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    input_list = args.input_json_list.expanduser().resolve()
    if not result_root.is_dir():
        raise FileNotFoundError(f"result root not found: {result_root}")
    cases = read_cases(input_list)
    if len(cases) != 67:
        raise ValueError(f"expected the 67-case PhysicIQ list, got {len(cases)}")

    video_dir = manifest_path.parent / "videos" / args.video_subdir
    video_dir.mkdir(parents=True, exist_ok=True)
    for input_path in cases:
        stem = input_path.stem
        source = result_root / f"{stem}.mp4"
        link = video_dir / f"{stem}.mp4"
        if not source.is_file():
            raise FileNotFoundError(f"missing generated video: {source}")
        if link.is_symlink() or link.exists():
            if link.resolve() != source.resolve():
                raise RuntimeError(f"existing page link points elsewhere: {link}")
        else:
            link.symlink_to(source)

    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        payload = {"models": []}
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise ValueError(f"invalid reference manifest: {manifest_path}")
    model = {
        "method_key": args.method_key,
        "method_label": args.method_label,
        "inference_steps": int(args.inference_steps),
        "checkpoint_label": args.checkpoint_label,
        "result_root": str(result_root),
        "video_prefix": f"../physrvg-physiciq-lora-ablation/videos/{args.video_subdir}",
        "registration": {
            "input_json_list": str(input_list),
            "num_cases": len(cases),
            "result_profile": "8844 PhysicIQ strict profile",
        },
    }
    models = [item for item in payload["models"] if item.get("method_key") != args.method_key]
    models.append(model)
    payload["models"] = models
    atomic_write(manifest_path, payload)
    print(json.dumps(model, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

