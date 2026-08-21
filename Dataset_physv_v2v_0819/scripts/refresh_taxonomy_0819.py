"""Backfill taxonomy metadata after adding controlled Scene families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .taxonomy_0819 import TAXONOMY_DEFINITIONS, taxonomy_for_family


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def refresh_dataset(output_root: Path) -> int:
    sample_root = output_root / "samples"
    sample_dirs = sorted(path for path in sample_root.iterdir() if path.is_dir())
    rows_by_id: dict[str, dict[str, object]] = {}
    for sample_dir in sample_dirs:
        metadata_path = sample_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        family_key = str(metadata.get("family_key", ""))
        taxonomy = taxonomy_for_family(family_key)
        metadata["taxonomy"] = taxonomy
        metadata["taxonomy_definition"] = TAXONOMY_DEFINITIONS[taxonomy]
        _write_json(metadata_path, metadata)

        for relative_path in (
            "manifest.json",
            "meta.json",
            "raw/simulator_render_metadata.json",
        ):
            path = sample_dir / relative_path
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["taxonomy"] = taxonomy
            payload["taxonomy_definition"] = TAXONOMY_DEFINITIONS[taxonomy]
            _write_json(path, payload)

        rows_by_id[sample_dir.name] = {
            "taxonomy": taxonomy,
            "taxonomy_definition": TAXONOMY_DEFINITIONS[taxonomy],
        }

    manifest_path = output_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in manifest.get("samples", []):
        sample_id = str(row.get("sample_id", ""))
        if sample_id in rows_by_id:
            row.update(rows_by_id[sample_id])
    manifest["taxonomy"] = TAXONOMY_DEFINITIONS
    manifest["taxonomy_counts"] = {
        taxonomy: sum(
            1 for row in manifest.get("samples", []) if row.get("taxonomy") == taxonomy
        )
        for taxonomy in TAXONOMY_DEFINITIONS
    }
    _write_json(manifest_path, manifest)

    dataset_meta_path = output_root / "dataset_meta.json"
    dataset_meta = json.loads(dataset_meta_path.read_text(encoding="utf-8"))
    dataset_meta["taxonomy"] = TAXONOMY_DEFINITIONS
    _write_json(dataset_meta_path, dataset_meta)
    return len(sample_dirs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/physv_v2v_0819"),
    )
    args = parser.parse_args()
    print(json.dumps({"samples_refreshed": refresh_dataset(args.output_root)}))


if __name__ == "__main__":  # pragma: no cover
    main()
