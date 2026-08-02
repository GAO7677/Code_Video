#!/usr/bin/env python3
"""Validate a T-head + slot-dedup smoke checkpoint and its inference output."""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import decord
import numpy as np
import torch
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, default=49)
    parser.add_argument("--expected-height", type=int, default=512)
    parser.add_argument("--expected-width", type=int, default=896)
    parser.add_argument("--expected-heads", type=int, default=70)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_nonfinite(value: Any, path: str = "root") -> list[str]:
    failures: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            failures.extend(find_nonfinite(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(find_nonfinite(child, f"{path}[{index}]"))
    elif isinstance(value, float) and not math.isfinite(value):
        failures.append(path)
    return failures


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> int:
    args = parse_args()
    failures: list[str] = []
    report: dict[str, Any] = {
        "run_root": str(args.run_root),
        "checkpoint_dir": str(args.checkpoint_dir),
        "inference_dir": str(args.inference_dir),
    }

    checkpoint_path = args.checkpoint_dir / "checkpoint.safetensors"
    training_state_path = args.checkpoint_dir / "training_state.pt"
    resolved_path = args.run_root / "resolved_experiment_config.json"
    head_config_path = args.run_root / "head_selection_config.json"
    for path in (checkpoint_path, training_state_path, resolved_path, head_config_path):
        require(path.is_file() and path.stat().st_size > 0, f"missing or empty: {path}", failures)
    if failures:
        report["failures"] = failures
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise SystemExit("; ".join(failures))

    resolved = load_json(resolved_path)
    config = resolved["resolved_config"]
    snapshot = resolved["head_selection_snapshot"]
    head_config = load_json(head_config_path)
    require(config["adaptation"]["mode"] == "t_head", "adaptation mode is not t_head", failures)
    require(config["adaptation"]["enable_object_branch"] is True, "object branch is disabled", failures)
    require(
        config["conditioning"]["slot_dedup"]["mode"] == "merge",
        "slot dedup mode is not merge",
        failures,
    )
    require(snapshot["num_heads"] == args.expected_heads, "snapshot head count mismatch", failures)
    require(head_config["num_heads"] == args.expected_heads, "head config count mismatch", failures)
    require(head_config["role"] == "T", "head config role is not T", failures)
    require(len(head_config["targets"]) == args.expected_heads, "head target list length mismatch", failures)
    require(len({(item["block"], item["head"]) for item in head_config["targets"]}) == args.expected_heads,
            "head target list contains duplicates", failures)

    training_state = torch.load(training_state_path, map_location="cpu", weights_only=False)
    require(training_state.get("global_step") == 1, "training state global_step is not 1", failures)

    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
        checkpoint_keys = list(handle.keys())
    object_lora_keys = [key for key in checkpoint_keys if ".object_cross_attn." in key and ".lora_" in key]
    head_lora_keys = [key for key in checkpoint_keys if ".self_attn." in key and ".head_lora_" in key]
    require(bool(object_lora_keys), "checkpoint has no object cross-attention LoRA", failures)
    require(bool(head_lora_keys), "checkpoint has no head-specific self-attention LoRA", failures)

    videos = sorted(args.inference_dir.glob("*.mp4"))
    require(len(videos) == 1, f"expected one inference video, found {len(videos)}", failures)
    video_reports: list[dict[str, Any]] = []
    for video_path in videos:
        metadata_path = video_path.with_suffix(".json")
        require(metadata_path.is_file(), f"missing inference metadata: {metadata_path}", failures)
        if not metadata_path.is_file():
            continue
        metadata = load_json(metadata_path)
        nonfinite_paths = find_nonfinite(metadata)
        require(not nonfinite_paths, f"non-finite metadata values: {nonfinite_paths}", failures)
        dedup = metadata.get("object_debug", {}).get("xssc_slot_dedup", {})
        require(dedup.get("mode") == "merge", "inference did not use merge dedup", failures)
        require(float(dedup.get("enabled", 0.0)) == 1.0, "inference dedup is not enabled", failures)
        require(metadata.get("model_args", {}).get("enable_object_branch") is True,
                "inference object branch is disabled", failures)

        reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
        frames = reader.get_batch(range(len(reader))).asnumpy()
        expected_shape = (
            args.expected_frames,
            args.expected_height,
            args.expected_width,
            3,
        )
        require(tuple(frames.shape) == expected_shape,
                f"video shape {tuple(frames.shape)} != {expected_shape}", failures)
        require(np.isfinite(frames).all(), "video contains non-finite pixels", failures)
        pixel_std = float(frames.std())
        pixel_range = int(frames.max()) - int(frames.min())
        require(pixel_std > 1.0, f"video pixel std is too small: {pixel_std}", failures)
        require(pixel_range > 4, f"video pixel range is too small: {pixel_range}", failures)
        video_reports.append(
            {
                "path": str(video_path),
                "shape": list(frames.shape),
                "pixel_std": pixel_std,
                "pixel_range": pixel_range,
                "slot_dedup": dedup,
            }
        )

    report.update(
        {
            "passed": not failures,
            "failures": failures,
            "global_step": training_state.get("global_step"),
            "head_selection": {
                "subset_id": head_config["subset_id"],
                "role": head_config["role"],
                "num_heads": head_config["num_heads"],
                "sha256": snapshot["sha256"],
            },
            "checkpoint": {
                "tensor_count": len(checkpoint_keys),
                "object_lora_tensor_count": len(object_lora_keys),
                "head_lora_tensor_count": len(head_lora_keys),
            },
            "videos": video_reports,
        }
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if failures:
        raise SystemExit("; ".join(failures))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
