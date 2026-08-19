"""Refresh metadata-driven captions in an exported PhysV V2V dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .caption_templates_0819 import (
    CAPTION_FILES,
    CAPTION_SCHEMA_VERSION,
    attach_caption_metadata,
)


DATASET_SCHEMA_VERSION = "physv_v2v_rigidbench_style_v2"


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def refresh_sample(sample_dir: Path) -> None:
    metadata_path = sample_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    bundle = attach_caption_metadata(metadata)
    metadata["schema_version"] = DATASET_SCHEMA_VERSION

    caption_dir = sample_dir / "captions"
    caption_dir.mkdir(parents=True, exist_ok=True)
    (caption_dir / "caption_specific.txt").write_text(bundle["specific"] + "\n", encoding="utf-8")
    (caption_dir / "caption_abstract.txt").write_text(bundle["abstract"] + "\n", encoding="utf-8")
    _write_json(
        caption_dir / "captions.json",
        {
            "schema_version": CAPTION_SCHEMA_VERSION,
            "source": "metadata.json",
            "specific": bundle["specific"],
            "abstract": bundle["abstract"],
        },
    )
    _write_json(metadata_path, metadata)

    meta_path = sample_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["schema_version"] = DATASET_SCHEMA_VERSION
    content = meta.setdefault("content", {})
    content.pop("prompt", None)
    content["caption_specific"] = CAPTION_FILES["specific"]
    content["caption_abstract"] = CAPTION_FILES["abstract"]
    content["captions"] = CAPTION_FILES["bundle"]
    status = meta.setdefault("status", {})
    status.pop("prompt", None)
    status["captions"] = True
    _write_json(meta_path, meta)

    manifest_path = sample_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = DATASET_SCHEMA_VERSION
    manifest["captions"] = {
        "specific": CAPTION_FILES["specific"],
        "abstract": CAPTION_FILES["abstract"],
        "bundle": CAPTION_FILES["bundle"],
    }
    _write_json(manifest_path, manifest)
    (sample_dir / "prompt.txt").unlink(missing_ok=True)


def refresh_dataset(output_root: Path) -> int:
    sample_dirs = sorted(path for path in (output_root / "samples").iterdir() if path.is_dir())
    for sample_dir in sample_dirs:
        refresh_sample(sample_dir)

    for filename in ("manifest.json", "dataset_meta.json"):
        path = output_root / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["schema_version"] = DATASET_SCHEMA_VERSION
        if filename == "dataset_meta.json":
            payload["captions"] = {
                "specific": CAPTION_FILES["specific"],
                "abstract": CAPTION_FILES["abstract"],
                "bundle": CAPTION_FILES["bundle"],
                "caption_schema_version": CAPTION_SCHEMA_VERSION,
            }
        _write_json(path, payload)

    readme_path = output_root / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    readme = readme.replace(
        "- `metadata.json`, `meta.json`, `manifest.json`, `prompt.txt`: sample metadata and continuation conditioning metadata.",
        "- `captions/caption_specific.txt`: caption with the controlled variable and value exposed.\n"
        "- `captions/caption_abstract.txt`: caption with the controlled variable and value hidden.\n"
        "- `captions/captions.json`: structured copy of both caption versions.\n"
        "- `metadata.json`, `meta.json`, `manifest.json`: sample metadata and caption references.",
    )
    readme_path.write_text(readme, encoding="utf-8")
    return len(sample_dirs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/physv_v2v_0819"),
    )
    args = parser.parse_args()
    count = refresh_dataset(args.output_root)
    print(json.dumps({"output_root": str(args.output_root), "samples_refreshed": count}))


if __name__ == "__main__":
    main()
