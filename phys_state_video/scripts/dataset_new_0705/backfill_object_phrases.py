#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .render_sim_0705 import build_object_phrase_bundle
from .scene_generators_0705 import generate_scenario_blueprint


DEFAULT_DATASET_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0713pybullet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill per-object noun phrase annotations for generated rigid datasets."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--include-direction-check",
        action="store_true",
        help="Also process dataset-root/direction_check when its manifest exists.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _case_id_from_record(record: dict[str, Any]) -> str:
    return str(record.get("case_id") or record.get("sample_key") or Path(str(record["output_root"])).name)


def _direction_mode(record: dict[str, Any], meta_payload: dict[str, Any] | None) -> str:
    if record.get("direction_mode"):
        return str(record["direction_mode"])
    if meta_payload:
        blueprint = meta_payload.get("blueprint", {})
        if isinstance(blueprint, dict):
            metadata = blueprint.get("metadata", {})
            if isinstance(metadata, dict) and metadata.get("direction_mode"):
                return str(metadata["direction_mode"])
    return "auto"


def _apply_object_phrases_to_meta(meta_path: Path, phrase_payload: dict[str, Any]) -> None:
    if not meta_path.exists():
        return
    meta_payload = _load_json(meta_path)
    meta_payload["object_nouns"] = phrase_payload["object_nouns"]
    meta_payload["object_phrases"] = phrase_payload["object_phrases"]
    meta_payload["dynamic_object_phrases"] = phrase_payload["dynamic_object_phrases"]
    meta_payload["static_object_phrases"] = phrase_payload["static_object_phrases"]
    meta_payload["object_phrase_details"] = phrase_payload["object_phrase_details"]

    detail_by_name = {
        str(item["name"]): item
        for item in phrase_payload["object_phrase_details"]
        if isinstance(item, dict)
    }
    for obj_payload in meta_payload.get("objects", []):
        if not isinstance(obj_payload, dict):
            continue
        detail = detail_by_name.get(str(obj_payload.get("name", "")))
        if not detail:
            continue
        obj_payload["family_key"] = detail["family_key"]
        obj_payload["object_noun"] = detail["object_noun"]
        obj_payload["material_key"] = detail["material_key"]
        obj_payload["material_phrase"] = detail["material_phrase"]
        obj_payload["object_phrase"] = detail["object_phrase"]

    _write_json(meta_path, meta_payload)


def _apply_object_phrases_to_case_manifest(case_manifest_path: Path, phrase_payload: dict[str, Any], sidecar_path: Path) -> None:
    if not case_manifest_path.exists():
        return
    case_manifest = _load_json(case_manifest_path)
    case_manifest["object_phrases_path"] = str(sidecar_path)
    case_manifest["object_nouns"] = phrase_payload["object_nouns"]
    case_manifest["object_phrases"] = phrase_payload["object_phrases"]
    case_manifest["dynamic_object_phrases"] = phrase_payload["dynamic_object_phrases"]
    case_manifest["static_object_phrases"] = phrase_payload["static_object_phrases"]
    _write_json(case_manifest_path, case_manifest)


def _process_manifest_root(manifest_root: Path) -> dict[str, Any]:
    manifest_path = manifest_root / "manifest.json"
    if not manifest_path.exists():
        return {"manifest_root": str(manifest_root), "cases": 0, "objects": 0, "missing_manifest": True}

    records = _load_json(manifest_path)
    if not isinstance(records, list):
        raise ValueError(f"manifest must be a list: {manifest_path}")

    aggregate_cases: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    updated_records: list[dict[str, Any]] = []

    for record in records:
        if not isinstance(record, dict):
            continue
        case_id = _case_id_from_record(record)
        family_key = str(record["family_key"])
        seed = int(record["seed"])
        meta_path = Path(str(record.get("meta", ""))) if record.get("meta") else None
        meta_payload = _load_json(meta_path) if meta_path and meta_path.exists() else None
        direction_mode = _direction_mode(record, meta_payload)
        blueprint = generate_scenario_blueprint(
            family_key=family_key,
            sample_key=case_id,
            seed=seed,
            direction_mode=direction_mode,
        )
        phrase_bundle = build_object_phrase_bundle(blueprint)
        sidecar_path = (
            meta_path.parent / f"{case_id}_object_phrases.json"
            if meta_path is not None
            else Path(str(record["output_root"])) / "meta" / f"{case_id}_object_phrases.json"
        )
        phrase_payload = {
            "case_id": case_id,
            "family_key": family_key,
            "seed": seed,
            "direction_mode": direction_mode,
            **phrase_bundle,
        }

        _write_json(sidecar_path, phrase_payload)
        if meta_path is not None:
            _apply_object_phrases_to_meta(meta_path, phrase_payload)
        _apply_object_phrases_to_case_manifest(
            Path(str(record["output_root"])) / "case_manifest.json",
            phrase_payload,
            sidecar_path,
        )

        record["object_phrases_path"] = str(sidecar_path)
        record["object_nouns"] = phrase_payload["object_nouns"]
        record["object_phrases"] = phrase_payload["object_phrases"]
        record["dynamic_object_phrases"] = phrase_payload["dynamic_object_phrases"]
        record["static_object_phrases"] = phrase_payload["static_object_phrases"]
        updated_records.append(record)

        aggregate_cases.append(phrase_payload)
        for detail in phrase_payload["object_phrase_details"]:
            row = {
                "case_id": case_id,
                "scenario_family_key": family_key,
                "seed": seed,
                "direction_mode": direction_mode,
                "name": detail["name"],
                "role": detail["role"],
                "dynamic": detail["dynamic"],
                "object_family_key": detail["family_key"],
                "object_noun": detail["object_noun"],
                "material_key": detail["material_key"],
                "material_phrase": detail["material_phrase"],
                "object_phrase": detail["object_phrase"],
            }
            aggregate_rows.append(row)

    _write_json(manifest_path, updated_records)
    _write_json(manifest_root / "object_phrases.json", aggregate_cases)
    csv_path = manifest_root / "object_phrases.csv"
    if aggregate_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0].keys()))
            writer.writeheader()
            writer.writerows(aggregate_rows)

    return {
        "manifest_root": str(manifest_root),
        "cases": len(aggregate_cases),
        "objects": len(aggregate_rows),
        "object_phrases_json": str(manifest_root / "object_phrases.json"),
        "object_phrases_csv": str(csv_path),
    }


def main() -> None:
    args = parse_args()
    manifest_roots = [args.dataset_root]
    direction_root = args.dataset_root / "direction_check"
    if args.include_direction_check and direction_root.exists():
        manifest_roots.append(direction_root)
    summaries = [_process_manifest_root(path) for path in manifest_roots]
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
