from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


_SPLIT_NAMES = {"train", "val", "test", "all"}


def _stable_unit_interval(text: str) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def _sample_split_name(key: str, *, train_ratio: float, val_ratio: float) -> str:
    u = _stable_unit_interval(key)
    if u < train_ratio:
        return "train"
    if u < train_ratio + val_ratio:
        return "val"
    return "test"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _caption_from_metadata(metadata: dict[str, Any], default: str) -> str:
    for key in ("input_caption", "caption", "prompt"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    simulation_type = metadata.get("simulation_type")
    if isinstance(simulation_type, str) and simulation_type.strip():
        return simulation_type.strip().replace("_", " ")
    return default


def _build_caption(sample_dir: Path, metadata: dict[str, Any] | None, scenario_default: str) -> str:
    candidates = [
        sample_dir / "caption.txt",
        sample_dir.parent.parent / "common_caption_cosmos.txt",
    ]
    candidates.extend(sorted(sample_dir.parent.parent.glob("common_caption_cosmos*.txt")))
    for candidate in candidates:
        if candidate.is_file():
            text = _read_text(candidate)
            if text:
                return text
    if metadata is not None:
        return _caption_from_metadata(metadata, scenario_default)
    return scenario_default


def _frame_count_hint_from_metadata(metadata: dict[str, Any] | None) -> int | None:
    if metadata is None:
        return None
    rendering_efficiency = metadata.get("rendering_efficiency", {})
    if not isinstance(rendering_efficiency, dict):
        return None
    for key in ("total_frames", "frames_rendered"):
        value = rendering_efficiency.get(key)
        if isinstance(value, int) and value > 0:
            return int(value)
        if isinstance(value, float) and value > 0:
            return int(value)
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sample Kubric/PhyCo rgba.mp4 cases and emit input jsons compatible "
            "with inspect_stage1b_prepipe_overlay.py."
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/gaoya/dataset/nnsriram97-phyco_kubric"),
    )
    parser.add_argument("--split", choices=sorted(_SPLIT_NAMES), default="train")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--split-train-ratio", type=float, default=0.9)
    parser.add_argument("--split-val-ratio", type=float, default=0.05)
    parser.add_argument("--scenario", action="append", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/gaoya/agent-data/outputs/kubric_prepipe_overlay/input_jsons"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {dataset_root}")
    if args.count <= 0:
        raise ValueError(f"--count must be positive, got {args.count}")
    if not 0.0 < float(args.split_train_ratio) < 1.0:
        raise ValueError("--split-train-ratio must be in (0,1)")
    if not 0.0 <= float(args.split_val_ratio) < 1.0:
        raise ValueError("--split-val-ratio must be in [0,1)")
    if float(args.split_train_ratio) + float(args.split_val_ratio) >= 1.0:
        raise ValueError("split ratios must sum to < 1.0")

    scenario_filter = {item.strip() for item in args.scenario or [] if item and item.strip()}
    candidates: list[dict[str, Any]] = []
    for rgba_path in dataset_root.glob("*/*/*/rgba.mp4"):
        sample_dir = rgba_path.parent
        date_dir = sample_dir.parent
        scenario_dir = date_dir.parent
        scenario = scenario_dir.name
        if scenario_filter and scenario not in scenario_filter:
            continue
        key = f"{scenario}/{date_dir.name}/{sample_dir.name}"
        if args.split != "all":
            split_name = _sample_split_name(
                key,
                train_ratio=float(args.split_train_ratio),
                val_ratio=float(args.split_val_ratio),
            )
            if split_name != str(args.split):
                continue
        metadata_path = sample_dir / "metadata.json"
        metadata = _read_json(metadata_path) if metadata_path.is_file() else None
        frame_count_hint = _frame_count_hint_from_metadata(metadata)
        if frame_count_hint is not None and int(frame_count_hint) < int(args.num_frames):
            continue
        prompt = _build_caption(sample_dir, metadata, scenario.replace("_", " "))
        candidates.append(
            {
                "sample_key": key,
                "scenario": scenario,
                "date": date_dir.name,
                "sample_id": sample_dir.name,
                "input_video": str(rgba_path.resolve()),
                "input_caption": prompt,
                "frame_count_hint": frame_count_hint,
            }
        )

    if len(candidates) < int(args.count):
        raise RuntimeError(
            f"only found {len(candidates)} candidate samples for split={args.split}, "
            f"smaller than requested count={args.count}"
        )

    rng = random.Random(int(args.seed))
    rng.shuffle(candidates)
    selected = candidates[: int(args.count)]

    json_paths: list[str] = []
    for index, item in enumerate(selected):
        stem = f"{index:02d}_{item['scenario']}_{item['date']}_{item['sample_id']}"
        json_path = output_dir / f"{stem}.json"
        payload = {
            "input_video": item["input_video"],
            "input_caption": item["input_caption"],
            "sample_key": item["sample_key"],
            "scenario": item["scenario"],
            "date": item["date"],
            "sample_id": item["sample_id"],
        }
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        json_paths.append(str(json_path))

    manifest = {
        "dataset_root": str(dataset_root),
        "split": str(args.split),
        "count": int(args.count),
        "seed": int(args.seed),
        "num_frames": int(args.num_frames),
        "split_train_ratio": float(args.split_train_ratio),
        "split_val_ratio": float(args.split_val_ratio),
        "json_paths": json_paths,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
