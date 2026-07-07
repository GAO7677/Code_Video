#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_ROOTS = (
    Path("/data/gaoya/AAA_test_video/0623/test/ti2v"),
    Path("/data/gaoya/AAA_test_video/0623/test/t2v"),
)
DEFAULT_INPUT_JSON_ROOT = Path("/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons")
EXCLUDED_JSON_NAMES = {"summary.json", "result.json", "batch_manifest.json", "eval_summary.json", "state.json"}


@dataclass
class UpdateStat:
    path: Path
    changed_keys: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize ti2v/t2v per-case result jsons toward the v2v-style field names. "
            "This is an in-place migration that only adds/fills normalized fields and "
            "does not delete legacy fields."
        )
    )
    parser.add_argument("--root", dest="roots", action="append", type=Path, default=[])
    parser.add_argument("--input-json-root", type=Path, default=DEFAULT_INPUT_JSON_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def is_result_json(path: Path) -> bool:
    if path.name in EXCLUDED_JSON_NAMES:
        return False
    if path.name.startswith("eval_summary_"):
        return False
    if "/.watch" in str(path):
        return False
    return True


def resolve_existing_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = candidate.resolve()
    else:
        candidate = candidate.resolve()
    if candidate.exists():
        return candidate
    return None


def infer_mode(result_json_path: Path, payload: dict[str, Any]) -> str | None:
    mode = payload.get("mode")
    if isinstance(mode, str) and mode.strip():
        return mode.strip()
    parts = {part.lower() for part in result_json_path.parts}
    if "ti2v" in parts:
        return "ti2v"
    if "t2v" in parts:
        return "t2v"
    return None


def infer_method(result_json_path: Path, payload: dict[str, Any]) -> str | None:
    method = payload.get("method")
    if isinstance(method, str) and method.strip():
        return method.strip()
    model_preset = payload.get("model_preset")
    if isinstance(model_preset, str) and model_preset.strip():
        return model_preset.strip()
    return result_json_path.parent.name


def infer_case_json_path(result_json_path: Path, payload: dict[str, Any], input_json_root: Path) -> Path | None:
    for key in ("case_json", "input_json"):
        current = resolve_existing_path(payload.get(key))
        if current is not None and current.is_file():
            return current
    candidate = (input_json_root / result_json_path.name).expanduser().resolve()
    if candidate.is_file():
        return candidate
    return None


def infer_output_video_path(result_json_path: Path, payload: dict[str, Any]) -> Path | None:
    for key in ("output_video", "video_path"):
        candidate = resolve_existing_path(payload.get(key))
        if candidate is not None and candidate.is_file():
            return candidate
    sibling_mp4 = result_json_path.with_suffix(".mp4").resolve()
    if sibling_mp4.is_file():
        return sibling_mp4
    return None


def infer_text(payload: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def maybe_set(payload: dict[str, Any], key: str, value: Any, changed_keys: list[str]) -> None:
    if value is None:
        return
    current = payload.get(key)
    if current == value:
        return
    payload[key] = value
    changed_keys.append(key)


def maybe_remove(payload: dict[str, Any], key: str, changed_keys: list[str]) -> None:
    if key not in payload:
        return
    payload.pop(key, None)
    changed_keys.append(key)


def normalize_one(result_json_path: Path, input_json_root: Path, *, write_back: bool) -> UpdateStat | None:
    payload = load_json(result_json_path)
    changed_keys: list[str] = []

    mode = infer_mode(result_json_path, payload)
    case_json_path = infer_case_json_path(result_json_path, payload, input_json_root)
    input_payload = load_json(case_json_path) if case_json_path is not None else None
    output_video_path = infer_output_video_path(result_json_path, payload)
    method = infer_method(result_json_path, payload)

    input_caption = infer_text(payload, ("input_caption", "prompt", "caption", "input_prompt"))
    if input_caption is None and input_payload is not None:
        input_caption = infer_text(input_payload, ("input_caption", "prompt", "caption", "input_prompt"))

    guidance = payload.get("guidance")
    if guidance is None:
        guidance = payload.get("cfg_scale")

    step = payload.get("step")
    if step is None:
        step = payload.get("num_inference_steps")

    negative_prompt = payload.get("negative_prompt")
    if negative_prompt is None:
        negative_prompt = ""

    input_video_path = resolve_existing_path(payload.get("input_video"))
    if input_video_path is None:
        input_video_path = resolve_existing_path(payload.get("context_path"))

    input_image_path = resolve_existing_path(payload.get("input_image"))
    if input_image_path is None:
        input_image_path = resolve_existing_path(payload.get("first_frame_path"))

    has_explicit_first_frame = any(
        isinstance(payload.get(key), str) and str(payload.get(key)).strip()
        for key in ("first_frame_path", "input_image")
    )
    conditioning_mode = payload.get("conditioning_mode")
    uses_input_image = isinstance(conditioning_mode, str) and conditioning_mode == "input_image_only"

    source_video_path = resolve_existing_path(payload.get("source_video"))
    if source_video_path is None and input_payload is not None:
        source_video_path = resolve_existing_path(input_payload.get("source_video"))

    maybe_set(payload, "mode", mode, changed_keys)
    maybe_set(payload, "input_json", str(case_json_path) if case_json_path is not None else None, changed_keys)
    maybe_set(payload, "case_json", str(case_json_path) if case_json_path is not None else None, changed_keys)
    maybe_set(payload, "input_caption", input_caption, changed_keys)
    maybe_set(payload, "output_video", str(output_video_path) if output_video_path is not None else None, changed_keys)
    maybe_set(payload, "video_path", str(output_video_path) if output_video_path is not None else None, changed_keys)
    maybe_set(payload, "method", method, changed_keys)
    maybe_set(payload, "guidance", guidance, changed_keys)
    maybe_set(payload, "step", step, changed_keys)
    maybe_set(payload, "negative_prompt", negative_prompt, changed_keys)
    if mode == "ti2v" and input_video_path is not None and not uses_input_image:
        maybe_set(payload, "input_video", str(input_video_path), changed_keys)
    else:
        maybe_remove(payload, "input_video", changed_keys)
    if mode == "ti2v" and uses_input_image and input_image_path is not None and has_explicit_first_frame:
        maybe_set(payload, "input_image", str(input_image_path), changed_keys)
    else:
        maybe_remove(payload, "input_image", changed_keys)
    maybe_set(payload, "source_video", str(source_video_path) if source_video_path is not None else None, changed_keys)

    if not changed_keys:
        return None
    if write_back:
        write_json(result_json_path, payload)
    return UpdateStat(path=result_json_path, changed_keys=changed_keys)


def main() -> None:
    args = parse_args()
    roots = [path.expanduser().resolve() for path in (args.roots or list(DEFAULT_ROOTS))]
    input_json_root = args.input_json_root.expanduser().resolve()

    stats: list[UpdateStat] = []
    scanned = 0
    for root in roots:
        if not root.is_dir():
            continue
        for result_json_path in sorted(root.rglob("*.json")):
            if not is_result_json(result_json_path):
                continue
            scanned += 1
            stat = normalize_one(result_json_path, input_json_root, write_back=not args.dry_run)
            if stat is None:
                continue
            stats.append(stat)

    if args.dry_run:
        print(json.dumps(
            {
                "roots": [str(path) for path in roots],
                "input_json_root": str(input_json_root),
                "scanned": scanned,
                "changed": len(stats),
                "examples": [
                    {"path": str(item.path), "changed_keys": item.changed_keys}
                    for item in stats[:20]
                ],
            },
            ensure_ascii=False,
            indent=2,
        ))
        return

    print(json.dumps(
        {
            "roots": [str(path) for path in roots],
            "input_json_root": str(input_json_root),
            "scanned": scanned,
            "changed": len(stats),
            "examples": [
                {"path": str(item.path), "changed_keys": item.changed_keys}
                for item in stats[:20]
            ],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
