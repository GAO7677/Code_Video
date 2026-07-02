#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import imageio.v3 as iio
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WMREWARD_ROOT = Path(__file__).resolve().parents[1]
PROBE_ROOT = Path(__file__).resolve().parent
WMREWARD_DATA_ROOT = Path("/data/gaoya/AAA_test_video/0626vjepa_free/wmreward")
DEFAULT_INPUT_JSON_DIR = Path("/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons")
DEFAULT_PIPELINE_ROOT = WMREWARD_DATA_ROOT / "probe_wan22/datasets/generated"
DEFAULT_SMOKE_PIPELINE_ROOT = WMREWARD_DATA_ROOT / "tmp/smoke/pipeline_runs"
DEFAULT_WAN_PYTHON = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")
DEFAULT_WMREWARD_CHECKPOINT = Path("/data/gaoya/ckpt/Sylvest-vjepa2-vit-g/vitg-384.pt")
DEFAULT_WMREWARD_MODEL_NAME = "vitg384"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    script_path: Path
    model_name: str
    output_subdir: str
    kind: str
    wan_root: Path | None = None
    weights_root: Path | None = None


MODEL_SPECS: dict[str, ModelSpec] = {
    "base": ModelSpec(
        key="base",
        script_path=PROJECT_ROOT / "code_vjepa_vggt/AAAinfer/wanti2v.py",
        model_name="wan2p2_ti2v5B",
        output_subdir="generations/base",
        kind="ti2v",
        wan_root=Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B"),
    ),
    "openvid_lora": ModelSpec(
        key="openvid_lora",
        script_path=PROJECT_ROOT / "code_vjepa_vggt/AAAinfer/wan_openvid_lorav2v.py",
        model_name="wan_openvid_lorav2v_step10000",
        output_subdir="generations/openvid_lora",
        kind="lora",
        wan_root=Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B"),
        weights_root=Path(
            "/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora/checkpoints/step-010000"
        ),
    ),
    "pybullet_lora": ModelSpec(
        key="pybullet_lora",
        script_path=PROJECT_ROOT / "code_vjepa_vggt/AAAinfer/wan_openvid_0613pybullet_lorav2v.py",
        model_name="wan_openvid_0613pybullet_lorav2v_step000500",
        output_subdir="generations/pybullet_lora",
        kind="lora",
        wan_root=Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B"),
        weights_root=Path(
            "/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500"
        ),
    ),
}


def parse_model_keys(raw_value: str) -> list[str]:
    keys = [item.strip() for item in raw_value.split(",") if item.strip()]
    unknown = [key for key in keys if key not in MODEL_SPECS]
    if unknown:
        raise ValueError(f"Unknown model keys: {unknown}. Known keys: {sorted(MODEL_SPECS)}")
    return keys


def resolve_python_bin(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    if DEFAULT_WAN_PYTHON.is_file():
        return DEFAULT_WAN_PYTHON
    return Path(sys.executable).resolve()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def discover_input_jsons(input_json_dir: Path, limit: int | None = None) -> list[Path]:
    json_paths = sorted(path.resolve() for path in input_json_dir.glob("*.json"))
    if limit is not None:
        json_paths = json_paths[:limit]
    return json_paths


def write_list_file(path: Path, values: Iterable[Path]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(f"{value}\n")


def bool_str(value: bool) -> str:
    return "True" if value else "False"


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ensure_non_empty_str(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def derive_firstframe_png_path(source_video: Path) -> Path:
    return source_video.parent / f"{source_video.stem}_firstframe.png"


def extract_first_frame_png(video_path: Path, output_png: Path) -> Path:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    frame = iio.imread(video_path, index=0)
    if frame.ndim != 3:
        raise ValueError(f"unexpected first-frame shape from {video_path}: {frame.shape}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    Image.fromarray(frame).save(output_png)
    return output_png


def build_normalized_input_json(
    *,
    source_json_path: Path,
    normalized_root: Path,
) -> Path:
    payload = load_json(source_json_path)
    source_video = ensure_non_empty_str(payload, "source_video")
    if source_video is None:
        raise ValueError(f"missing or empty 'source_video' in {source_json_path}")
    source_video_path = Path(source_video).expanduser().resolve()
    payload["source_video"] = str(source_video_path)

    input_video = ensure_non_empty_str(payload, "input_video")
    if input_video is None:
        payload["input_video"] = str(source_video_path)
    else:
        payload["input_video"] = str(Path(input_video).expanduser().resolve())

    input_image = ensure_non_empty_str(payload, "input_image")
    if input_image is None:
        firstframe_path = derive_firstframe_png_path(source_video_path)
        if not firstframe_path.exists():
            extract_first_frame_png(source_video_path, firstframe_path)
        payload["input_image"] = str(firstframe_path.resolve())
    else:
        payload["input_image"] = str(Path(input_image).expanduser().resolve())

    output_path = normalized_root / source_json_path.name
    write_json(output_path, payload)
    return output_path


def make_generation_record(
    *,
    model_spec: ModelSpec,
    input_json_path: Path,
    output_json_path: Path,
    output_video_path: Path,
    pipeline_root: Path,
    generated_json: dict[str, Any] | None,
) -> dict[str, Any]:
    source_json = load_json(input_json_path)
    wmreward_payload = generated_json.get("wmreward") if generated_json else None
    wmreward_status = "missing"
    surprise_score = None
    similarity_score = None
    if isinstance(wmreward_payload, dict):
        surprise_score = safe_float(wmreward_payload.get("surprise"))
        similarity_score = safe_float(wmreward_payload.get("similarity"))
        if surprise_score is not None:
            wmreward_status = "ok"
        elif wmreward_payload.get("status"):
            wmreward_status = str(wmreward_payload["status"])

    return {
        "model_key": model_spec.key,
        "model_name": model_spec.model_name,
        "model_kind": model_spec.kind,
        "script_path": str(model_spec.script_path),
        "wan_root": str(model_spec.wan_root) if model_spec.wan_root is not None else "",
        "weights_root": str(model_spec.weights_root) if model_spec.weights_root is not None else "",
        "sample_stem": input_json_path.stem,
        "basename": f"{input_json_path.stem}.mp4",
        "input_json_path": str(input_json_path),
        "input_caption": source_json.get("input_caption", ""),
        "input_video_path": source_json.get("input_video", ""),
        "input_image_path": source_json.get("input_image", ""),
        "gt_video_path": source_json.get("source_video", ""),
        "output_json_path": str(output_json_path),
        "output_video_path": str(output_video_path),
        "relative_path": os.path.relpath(output_video_path, pipeline_root),
        "output_json_exists": bool_str(output_json_path.is_file()),
        "output_video_exists": bool_str(output_video_path.is_file()),
        "wmreward_status": wmreward_status,
        "surprise_score": f"{surprise_score:.8f}" if surprise_score is not None else "",
        "similarity_score": f"{similarity_score:.8f}" if similarity_score is not None else "",
    }


def scan_generated_records(
    *,
    model_spec: ModelSpec,
    input_json_paths: list[Path],
    pipeline_root: Path,
) -> list[dict[str, Any]]:
    model_output_root = pipeline_root / model_spec.output_subdir
    rows: list[dict[str, Any]] = []
    for input_json_path in input_json_paths:
        output_json_path = model_output_root / f"{input_json_path.stem}.json"
        output_video_path = model_output_root / f"{input_json_path.stem}.mp4"
        generated_json = load_json(output_json_path) if output_json_path.is_file() else None
        rows.append(
            make_generation_record(
                model_spec=model_spec,
                input_json_path=input_json_path,
                output_json_path=output_json_path,
                output_video_path=output_video_path,
                pipeline_root=pipeline_root,
                generated_json=generated_json,
            )
        )
    return rows


def generation_registry_fieldnames() -> list[str]:
    return [
        "model_key",
        "model_name",
        "model_kind",
        "script_path",
        "wan_root",
        "weights_root",
        "sample_stem",
        "basename",
        "input_json_path",
        "input_caption",
        "input_video_path",
        "input_image_path",
        "gt_video_path",
        "output_json_path",
        "output_video_path",
        "relative_path",
        "output_json_exists",
        "output_video_exists",
        "wmreward_status",
        "surprise_score",
        "similarity_score",
    ]


def pipeline_run_summary(
    *,
    input_json_dir: Path,
    pipeline_root: Path,
    selected_model_keys: list[str],
    input_json_count: int,
) -> dict[str, Any]:
    model_specs = {}
    for key in selected_model_keys:
        payload = asdict(MODEL_SPECS[key])
        for field_name in ("script_path", "wan_root", "weights_root"):
            if payload.get(field_name) is not None:
                payload[field_name] = str(payload[field_name])
        model_specs[key] = payload
    return {
        "input_json_dir": str(input_json_dir),
        "pipeline_root": str(pipeline_root),
        "selected_models": selected_model_keys,
        "input_json_count": input_json_count,
        "model_specs": model_specs,
    }
