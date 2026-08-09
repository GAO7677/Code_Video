#!/usr/bin/env python3
"""Build a stable VBench index for completed Legacy PhysicIQ67 ablations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


EXPERIMENT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50"
)
VISUAL_ROOT = EXPERIMENT_ROOT / "visual_samples"
SAMPLE_MANIFESTS = (
    VISUAL_ROOT / "samples.json",
    VISUAL_ROOT / "attention_zero_seed47326" / "cases.json",
)
FIXED_ROOTS = (
    VISUAL_ROOT / "attention_matrix_ablations_v2",
    VISUAL_ROOT / "attention_zero_seed47326" / "attention_matrix_ablations_v2",
)
TUBE_ROOT = (
    VISUAL_ROOT
    / "attention_zero_seed47326"
    / "attention_matrix_ablations_temporal_tube_v1"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/legacy_physiciq67_attention_ablation_vbench"
)


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def samples() -> list[dict]:
    merged: dict[tuple[str, int], dict] = {}
    for path in SAMPLE_MANIFESTS:
        if not path.is_file():
            continue
        for row in load_json(path).get("samples", []):
            merged[(str(row["case"]), int(row["seed"]))] = dict(row)
    return [merged[key] for key in sorted(merged)]


def baseline_record(sample: dict) -> tuple[Path, dict] | None:
    case, seed = str(sample["case"]), int(sample["seed"])
    default_dir = EXPERIMENT_ROOT / "runs" / case / f"seed_{seed:05d}"
    video = Path(str(sample.get("baseline_video") or default_dir / "generated.mp4"))
    run_dir = video.parent
    manifest_path = run_dir / "manifest.json"
    if not video.is_file() or not manifest_path.is_file():
        return None
    manifest = load_json(manifest_path)
    input_json = sample.get("input_json") or manifest.get("input_json")
    if not isinstance(input_json, str) or not Path(input_json).is_file():
        return None
    return run_dir, {
        "case": case,
        "seed": seed,
        "kind": "baseline",
        "variant_id": "baseline",
        "input_json": input_json,
        "video": str(video),
    }


def completed_ablation_records(
    root: Path, kind: str, selected_samples: set[tuple[str, int]]
) -> list[tuple[Path, dict]]:
    records: list[tuple[Path, dict]] = []
    if not root.is_dir():
        return records
    for complete_path in sorted(root.glob("*/seed_*/**/complete.json")):
        run_dir = complete_path.parent
        manifest_path = run_dir / "manifest.json"
        video_path = run_dir / "generated.mp4"
        if not manifest_path.is_file() or not video_path.is_file():
            continue
        try:
            manifest = load_json(manifest_path)
            case, seed = str(manifest["case"]), int(manifest["seed"])
            input_json = str(manifest["input_json"])
            variant_id = str(manifest["variant_id"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if (case, seed) not in selected_samples or not Path(input_json).is_file():
            continue
        records.append(
            (
                run_dir,
                {
                    "case": case,
                    "seed": seed,
                    "kind": kind,
                    "variant_id": variant_id,
                    "input_json": input_json,
                    "video": str(video_path),
                },
            )
        )
    return records


def link_name(metadata: dict) -> str:
    identity = "::".join(
        str(metadata[key]) for key in ("case", "seed", "kind", "variant_id")
    )
    return hashlib.sha1(identity.encode("utf-8")).hexdigest()[:20]


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    selected = samples()
    selected_keys = {(str(row["case"]), int(row["seed"])) for row in selected}
    records: list[tuple[Path, dict]] = []
    for sample in selected:
        record = baseline_record(sample)
        if record is not None:
            records.append(record)
    for root in FIXED_ROOTS:
        records.extend(completed_ablation_records(root, "fixed", selected_keys))
    records.extend(completed_ablation_records(TUBE_ROOT, "tube", selected_keys))
    records.sort(
        key=lambda item: (
            item[1]["case"],
            item[1]["seed"],
            item[1]["kind"],
            item[1]["variant_id"],
        )
    )

    output_root = args.output_root.expanduser().resolve()
    index_root = output_root / "index"
    index_root.mkdir(parents=True, exist_ok=True)
    indexed = []
    for run_dir, metadata in records:
        name = link_name(metadata)
        link = index_root / name
        target = run_dir.resolve()
        if link.is_symlink() and link.resolve() != target:
            link.unlink()
        if not link.exists():
            link.symlink_to(target, target_is_directory=True)
        indexed.append({**metadata, "index_link": str(link), "run_dir": str(target)})

    counts: dict[str, int] = {}
    for row in indexed:
        counts[row["kind"]] = counts.get(row["kind"], 0) + 1
    snapshot = {
        "index_root": str(index_root),
        "sample_count": len(selected),
        "record_count": len(indexed),
        "counts": counts,
        "records": indexed,
    }
    write_json_atomic(output_root / "snapshot.json", snapshot)
    print(json.dumps({key: snapshot[key] for key in ("index_root", "sample_count", "record_count", "counts")}, indent=2))


if __name__ == "__main__":
    main()
