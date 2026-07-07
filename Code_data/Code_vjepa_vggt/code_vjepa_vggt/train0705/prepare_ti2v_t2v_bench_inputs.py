#!/usr/bin/env python3
"""
Usage:
  /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/prepare_ti2v_t2v_bench_inputs.py \
    --source-root /data/gaoya/AAA_test_video/0623/test/ti2v \
    --source-root /data/gaoya/AAA_test_video/0623/test/t2v \
    --output-root /data/gaoya/agent-data/outputs/train0705_ti2v_t2v_bench_inputs \
    --limit-per-folder 1
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_ROOTS = (
    Path("/data/gaoya/AAA_test_video/0623/test/ti2v"),
    Path("/data/gaoya/AAA_test_video/0623/test/t2v"),
)
DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/agent-data/outputs/train0705_ti2v_t2v_bench_inputs")
DEFAULT_INPUT_JSON_ROOT = Path("/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons")
EXCLUDED_JSON_NAMES = {"summary.json", "result.json", "batch_manifest.json", "eval_summary.json"}
METRIC_FIELDS = (
    "wmreward",
    "physics_iq",
    "physics_iq_with_context",
    "physics_iq_without_context",
    "pmf_with_context",
    "pmf_without_context",
    "videophy2",
    "cosmos_reason1",
)


@dataclass(frozen=True)
class AdaptedCase:
    mode_root: str
    source_dir: str
    smoke_dir: str
    sample_json: str
    adapted_input_json: bool
    source_input_json: str | None
    method: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a bench.py-compatible mirror tree for ti2v/t2v results. "
            "Old-format result jsons without input_json/output_video are adapted into "
            "temporary bench inputs under the output root."
        )
    )
    parser.add_argument("--source-root", dest="source_roots", action="append", type=Path, default=[])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--input-json-root", type=Path, default=DEFAULT_INPUT_JSON_ROOT)
    parser.add_argument("--limit-per-folder", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def list_result_jsons(folder: Path) -> list[Path]:
    result = []
    for path in sorted(folder.glob("*.json")):
        if path.name in EXCLUDED_JSON_NAMES or path.name.startswith("eval_summary_"):
            continue
        result.append(path)
    return result


def find_leaf_result_dirs(source_root: Path) -> list[Path]:
    result_dirs: list[Path] = []
    for path in sorted(source_root.rglob("*")):
        if not path.is_dir():
            continue
        if list_result_jsons(path):
            result_dirs.append(path)
    return result_dirs


def choose_method(payload: dict[str, Any], source_dir: Path) -> str | None:
    value = payload.get("method")
    if isinstance(value, str) and value.strip():
        return value.strip()

    mode = payload.get("mode")
    model_preset = payload.get("model_preset")
    if isinstance(mode, str) and mode.strip() and isinstance(model_preset, str) and model_preset.strip():
        return f"{mode.strip()}_{model_preset.strip()}"
    if isinstance(model_preset, str) and model_preset.strip():
        return model_preset.strip()
    if source_dir.name:
        return source_dir.name
    return None


def resolve_source_case_json(
    payload: dict[str, Any],
    *,
    sample_name: str,
    input_json_root: Path,
) -> Path | None:
    for key in ("input_json", "case_json"):
        raw_input_json = payload.get(key)
        if isinstance(raw_input_json, str) and raw_input_json.strip():
            candidate = Path(raw_input_json).expanduser().resolve()
            if candidate.is_file():
                return candidate

    candidate = (input_json_root / sample_name).expanduser().resolve()
    if candidate.is_file():
        return candidate
    return None


def adapt_result_json(
    src_json_path: Path,
    *,
    source_root: Path,
    output_root: Path,
    input_json_root: Path,
) -> AdaptedCase:
    source_dir = src_json_path.parent.resolve()
    rel_dir = source_dir.relative_to(source_root.resolve())
    out_dir = (output_root / source_root.name / rel_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    dst_json_path = out_dir / src_json_path.name
    payload = load_json(src_json_path)
    for key in METRIC_FIELDS:
        payload.pop(key, None)

    source_case_json = resolve_source_case_json(payload, sample_name=src_json_path.name, input_json_root=input_json_root)
    adapted_input_json = False
    source_case_json_str: str | None = None
    source_payload: dict[str, Any] | None = None
    if source_case_json is not None:
        source_case_json_str = str(source_case_json)
        source_payload = load_json(source_case_json)

    raw_input_json = payload.get("input_json")
    if not isinstance(raw_input_json, str) or not raw_input_json.strip():
        raw_input_json = payload.get("case_json")
    if not isinstance(raw_input_json, str) or not raw_input_json.strip() or not Path(raw_input_json).expanduser().resolve().is_file():
        if source_payload is None:
            raise FileNotFoundError(
                f"Cannot derive source input json for {src_json_path}. "
                f"Expected {input_json_root / src_json_path.name}"
            )
        source_video = source_payload.get("source_video")
        if not isinstance(source_video, str) or not source_video.strip():
            raise ValueError(f"Missing source_video in derived source json: {source_case_json}")
        bench_input_json_path = out_dir / f"{src_json_path.stem}__bench_input.json"
        write_json(bench_input_json_path, {"source_video": source_video})
        payload["input_json"] = str(bench_input_json_path.resolve())
        adapted_input_json = True
    else:
        payload["input_json"] = str(Path(raw_input_json).expanduser().resolve())

    source_mp4 = src_json_path.with_suffix(".mp4")
    if not source_mp4.is_file():
        source_video_value = payload.get("output_video") or payload.get("video_path")
        if isinstance(source_video_value, str) and source_video_value.strip():
            candidate_mp4 = Path(source_video_value).expanduser().resolve()
            if candidate_mp4.is_file():
                source_mp4 = candidate_mp4
    if not source_mp4.is_file():
        raise FileNotFoundError(f"Cannot resolve source mp4 for {src_json_path}")

    target_mp4 = out_dir / source_mp4.name
    if not target_mp4.exists():
        os.symlink(source_mp4, target_mp4)
    payload["output_video"] = str(target_mp4.resolve())
    payload["video_path"] = str(target_mp4.resolve())

    if not (isinstance(payload.get("input_caption"), str) and payload["input_caption"].strip()):
        for key in ("prompt", "caption", "input_prompt"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                payload["input_caption"] = value.strip()
                break
        else:
            if source_payload is not None:
                for key in ("prompt", "caption", "input_caption"):
                    value = source_payload.get(key)
                    if isinstance(value, str) and value.strip():
                        payload["input_caption"] = value.strip()
                        break

    method = choose_method(payload, source_dir)
    if method is not None:
        payload["method"] = method

    write_json(dst_json_path, payload)
    return AdaptedCase(
        mode_root=source_root.name,
        source_dir=str(source_dir),
        smoke_dir=str(out_dir),
        sample_json=src_json_path.name,
        adapted_input_json=adapted_input_json,
        source_input_json=source_case_json_str,
        method=method,
    )


def main() -> None:
    args = parse_args()
    source_roots = [path.expanduser().resolve() for path in (args.source_roots or list(DEFAULT_SOURCE_ROOTS))]
    output_root = args.output_root.expanduser().resolve()
    input_json_root = args.input_json_root.expanduser().resolve()
    if args.overwrite and output_root.exists():
        for path in sorted(output_root.rglob("*"), reverse=True):
            if path.is_symlink() or path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    output_root.mkdir(parents=True, exist_ok=True)

    adapted_cases: list[AdaptedCase] = []
    skipped_dirs: list[str] = []
    for source_root in source_roots:
        if not source_root.is_dir():
            skipped_dirs.append(str(source_root))
            continue
        for result_dir in find_leaf_result_dirs(source_root):
            result_jsons = list_result_jsons(result_dir)
            if args.limit_per_folder is not None:
                result_jsons = result_jsons[: max(0, int(args.limit_per_folder))]
            for result_json in result_jsons:
                adapted_cases.append(
                    adapt_result_json(
                        result_json,
                        source_root=source_root,
                        output_root=output_root,
                        input_json_root=input_json_root,
                    )
                )

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps([asdict(item) for item in adapted_cases], ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "source_roots": [str(path) for path in source_roots],
        "output_root": str(output_root),
        "input_json_root": str(input_json_root),
        "limit_per_folder": args.limit_per_folder,
        "num_cases": len(adapted_cases),
        "num_dirs": len({item.smoke_dir for item in adapted_cases}),
        "num_adapted_input_json": sum(1 for item in adapted_cases if item.adapted_input_json),
        "skipped_source_roots": skipped_dirs,
        "manifest_path": str(manifest_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
