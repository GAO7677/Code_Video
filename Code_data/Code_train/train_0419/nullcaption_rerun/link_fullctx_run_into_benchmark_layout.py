#!/usr/bin/env python3
"""Expose the ongoing full-context run under the benchmark's standard directory layout."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


NULL_BENCH_ROOT = Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V_nullcaption")
CAPTION_BENCH_ROOT = Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V")
LIVE_ROOT = NULL_BENCH_ROOT / "tools" / "fullctx_runs" / "physicsiq_fullvideo"

CAPTION_OUTPUT_DIR = CAPTION_BENCH_ROOT / "output" / "VACE_1_3B_V2V" / "context_fullctx_fullvideo"
NULL_OUTPUT_DIR = NULL_BENCH_ROOT / "output" / "VACE_1_3B_V2V" / "context_fullctx_fullvideo"
RUNTIME_DIR = NULL_BENCH_ROOT / "tools" / "runtime" / "vace_v2v_fullctx_fullvideo"
META_DIR = NULL_BENCH_ROOT / "tools" / "meta" / "physicsiq_fullctx_fullvideo"
LOG_DIR = NULL_BENCH_ROOT / "tools" / "logs" / "physicsiq_fullctx_fullvideo"


def ensure_clean_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        path.unlink()


def ensure_symlink(target: Path, link_path: Path) -> None:
    ensure_clean_parent(link_path)
    link_path.symlink_to(target)


def copy_file(target: Path, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target, dst_path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clear_generated_links(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("*"):
        if path.is_symlink() or path.is_file():
            path.unlink()


def compact_input_payload(sidecar: dict) -> dict:
    payload: dict[str, str | None] = {
        "input_caption": "",
        "input_image": None,
        "input_video": "",
    }
    caption = sidecar.get("caption")
    if isinstance(caption, str) and caption:
        payload["input_caption"] = caption

    paths = sidecar.get("paths", {})
    if not isinstance(paths, dict):
        return payload

    input_roles = paths.get("input_roles", [])
    if isinstance(input_roles, list):
        for item in input_roles:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            path = item.get("path")
            if not isinstance(path, str) or not path:
                continue
            if role in {"input_image", "vace_video_known_frame"} and not payload["input_image"]:
                payload["input_image"] = path
            elif role in {"context_video", "vace_video_known_frames"} and not payload["input_video"]:
                payload["input_video"] = path

    first_frame_path = paths.get("first_frame_path")
    if not payload["input_image"] and isinstance(first_frame_path, str) and first_frame_path:
        conditioning_mode = ((sidecar.get("generation_params") or {}).get("conditioning_mode"))
        if conditioning_mode in {"input_image_only", "ti2v_firstframe", "context_aware"}:
            payload["input_image"] = first_frame_path

    context_video_path = paths.get("context_video_path")
    if not payload["input_video"] and isinstance(context_video_path, str) and context_video_path:
        payload["input_video"] = context_video_path

    return payload


def compact_setting_payload(sidecar: dict) -> dict:
    params = sidecar.get("generation_params", {})
    if not isinstance(params, dict):
        params = {}
    return {
        "model": str(sidecar.get("model_name") or ""),
        "conditioning_mode": params.get("conditioning_mode"),
        "height": params.get("height"),
        "width": params.get("width"),
        "fps": params.get("fps"),
        "context_frames": params.get("context_frames"),
        "output_frames": params.get("requested_output_frames", params.get("num_frames")),
        "num_inference_steps": params.get("num_inference_steps"),
        "cfg_scale": params.get("cfg_scale"),
        "seed": sidecar.get("seed"),
    }


def build_compact_sidecar(*, live_json_path: Path, output_video_path: Path, output_json_path: Path) -> dict:
    sidecar = read_json(live_json_path)
    paths = sidecar.get("paths", {})
    if not isinstance(paths, dict):
        paths = {}
    benchmark_path = paths.get("sample_dir")
    payload = {
        "benchmark_path": benchmark_path,
        "input": compact_input_payload(sidecar),
        "output": {
            "output_video": str(output_video_path),
            "output_json": str(output_json_path),
        },
        "setting": compact_setting_payload(sidecar),
    }
    return payload


def main() -> None:
    if not LIVE_ROOT.exists():
        raise SystemExit(f"Live root missing: {LIVE_ROOT}")

    output_generated_root = LIVE_ROOT / "generated"
    runtime_generated_root = LIVE_ROOT / "runtime"
    meta_root = LIVE_ROOT / "meta"
    log_root = LIVE_ROOT / "logs"

    clear_generated_links(CAPTION_OUTPUT_DIR)
    clear_generated_links(NULL_OUTPUT_DIR)
    RUNTIME_DIR.parent.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Runtime summary symlink target.
    ensure_symlink(runtime_generated_root, RUNTIME_DIR)

    # Meta and logs are linked case-by-case / file-by-file to stay inspectable.
    for meta_file in sorted(meta_root.rglob("*")):
        if meta_file.is_file():
            ensure_symlink(meta_file, META_DIR / meta_file.relative_to(meta_root))

    for log_file in sorted(log_root.glob("*")):
        if log_file.is_file():
            ensure_symlink(log_file, LOG_DIR / log_file.name)

    # Flatten generated caption/nullcaption outputs into the benchmark-style output directory.
    # Videos are copied directly. Json sidecars are rewritten into a compact benchmark schema.
    for variant_dir in sorted(output_generated_root.iterdir()):
        if not variant_dir.is_dir():
            continue
        for path in sorted(variant_dir.glob("*")):
            if path.suffix not in {".mp4", ".json"}:
                continue
            stem = path.stem
            suffix = path.suffix
            if variant_dir.name.startswith("caption_"):
                output_dir = CAPTION_OUTPUT_DIR
                output_name = f"{stem}__caption_fullctx_fullvideo{suffix}"
            elif variant_dir.name.startswith("nullcaption_"):
                output_dir = NULL_OUTPUT_DIR
                output_name = f"{stem}__nullcaption_fullctx_fullvideo{suffix}"
            else:
                output_dir = NULL_OUTPUT_DIR
                output_name = path.name

            dst_path = output_dir / output_name
            if path.suffix == ".mp4":
                copy_file(path, dst_path)
                continue

            output_video_path = output_dir / output_name.replace(".json", ".mp4")
            compact_payload = build_compact_sidecar(
                live_json_path=path,
                output_video_path=output_video_path,
                output_json_path=dst_path,
            )
            write_json(dst_path, compact_payload)

    manifest = {
        "live_root": str(LIVE_ROOT),
        "caption_output_dir": str(CAPTION_OUTPUT_DIR),
        "null_output_dir": str(NULL_OUTPUT_DIR),
        "runtime_dir": str(RUNTIME_DIR),
        "meta_dir": str(META_DIR),
        "log_dir": str(LOG_DIR),
    }
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
