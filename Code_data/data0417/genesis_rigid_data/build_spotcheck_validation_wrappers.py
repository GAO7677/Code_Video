"""Build heldout-style wrapper folders so validate_saved_dataset_states.py can browse train-format spot-check samples."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wrap train-format Genesis samples with meta.json/full_video.mp4 aliases for the state validator."
    )
    parser.add_argument("--dataset_root", type=Path, required=True, help="Spot-check dataset root.")
    parser.add_argument(
        "--wrapper_root",
        type=Path,
        default=None,
        help="Where wrapper sample folders are written. Defaults to <dataset_root>/validator_wrappers.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Remove an existing wrapper root before rebuilding.")
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def copy_asset(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    wrapper_root = (args.wrapper_root or (dataset_root / "validator_wrappers")).resolve()

    if args.overwrite and wrapper_root.exists():
        shutil.rmtree(wrapper_root)
    wrapper_root.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted(
        path.parent
        for path in dataset_root.rglob("metadata.json")
        if (path.parent / "videos" / "rgb.mp4").exists()
    )
    if not sample_dirs:
        raise FileNotFoundError(f"No metadata.json files found under {dataset_root}")

    for sample_dir in sample_dirs:
        source_meta = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
        sample_id = str(source_meta.get("scene_id", sample_dir.name))
        wrapper_dir = wrapper_root / sample_id
        wrapper_dir.mkdir(parents=True, exist_ok=True)

        object_id = str(source_meta.get("object_id", "")).strip()
        if not object_id:
            objects = source_meta.get("objects", [])
            if objects:
                object_id = str(objects[0].get("source_object_id", objects[0].get("object_id", ""))).strip()

        fps = int(source_meta.get("simulation", {}).get("fps", source_meta.get("fps", 12) or 12))
        meta_payload = {
            "sample_id": sample_id,
            "scene_id": sample_id,
            "object_id": object_id,
            "fps": fps,
            "scene_composition": str(source_meta.get("scene_composition", "")),
            "interaction_pattern": str(source_meta.get("interaction_pattern", "")),
            "source_paths": {
                "source_sample_dir": str(sample_dir.resolve()),
            },
        }
        write_json(wrapper_dir / "meta.json", meta_payload)

        rgb_video = sample_dir / "videos" / "rgb.mp4"
        if not rgb_video.exists():
            raise FileNotFoundError(f"Missing rgb video for {sample_dir}")
        copy_asset(rgb_video.resolve(), wrapper_dir / "full_video.mp4")
        copy_asset(rgb_video.resolve(), wrapper_dir / "future_gt_video.mp4")

    print(f"[DONE] wrappers={len(sample_dirs)} root={wrapper_root}")


if __name__ == "__main__":
    main()
