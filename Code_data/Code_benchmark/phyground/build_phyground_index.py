#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DATASET_ROOT = Path("/data/gaoya/dataset/NU-World-Model-Embodied-AI-phyground")
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "phyground_index.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse the PhyGround dataset into a single local index JSON."
    )
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def relpath(path: Path, start: Path) -> str:
    return path.resolve().relative_to(start.resolve()).as_posix()


def build_video_map(dataset_root: Path) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]]]:
    videos_root = dataset_root / "videos"
    if not videos_root.is_dir():
        raise FileNotFoundError(f"Missing videos directory: {videos_root}")

    video_map: dict[str, dict[str, str]] = {}
    model_entries: list[dict[str, Any]] = []

    for model_dir in sorted(path for path in videos_root.iterdir() if path.is_dir()):
        stems: dict[str, str] = {}
        count = 0
        for video_path in sorted(model_dir.glob("*.mp4")):
            stems[video_path.stem] = relpath(video_path, dataset_root)
            count += 1
        video_map[model_dir.name] = stems
        model_entries.append({"model": model_dir.name, "video_count": count})

    # Annotation files collapse the two LTX stage variants into a shared model id.
    for alias, prefixes in {
        "ltx-2-19b-dev": ["ltx-2-19b-dev-one-stage", "ltx-2-19b-dev-two-stage"],
        "ltx-2.3-22b-dev": ["ltx-2.3-22b-dev-one-stage", "ltx-2.3-22b-dev-two-stage"],
    }.items():
        merged: dict[str, str] = {}
        for prefix in prefixes:
            merged.update(video_map.get(prefix, {}))
        if merged:
            video_map[alias] = merged

    return video_map, model_entries


def build_prompt_map(dataset_root: Path) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    prompts_path = dataset_root / "prompts" / "phyground.json"
    prompts = load_json(prompts_path)
    prompt_map: dict[int, dict[str, Any]] = {}
    prompt_entries: list[dict[str, Any]] = []

    for prompt in prompts:
        prompt_id = int(prompt["id"])
        entry = {
            "prompt_id": prompt_id,
            "id_stem": prompt.get("id_stem"),
            "prompt": prompt.get("prompt"),
            "physical_laws": prompt.get("physical_laws") or [],
        }
        prompt_map[prompt_id] = entry
        prompt_entries.append(entry)

    return prompt_map, prompt_entries


def parse_annotations(
    dataset_root: Path,
    prompt_map: dict[int, dict[str, Any]],
    video_map: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], Counter[str], Counter[str]]:
    annotations_root = dataset_root / "annotations"
    annotator_files = sorted(annotations_root.glob("annotator_*.json"))
    if not annotator_files:
        raise FileNotFoundError(f"No annotator files found under: {annotations_root}")

    items: list[dict[str, Any]] = []
    model_counter: Counter[str] = Counter()
    law_counter: Counter[str] = Counter()

    for annotator_path in annotator_files:
        annotator_blob = load_json(annotator_path)
        annotator_id = annotator_blob["annotator_id"]
        for local_index, ann in enumerate(annotator_blob.get("annotations") or []):
            model = ann["model"]
            video_stem = ann["video"]
            prompt_id = int(ann["prompt_id"])
            prompt_entry = prompt_map.get(prompt_id, {})
            video_relpath = video_map.get(model, {}).get(video_stem)
            item = {
                "annotation_id": f"{annotator_id}:{local_index:04d}",
                "annotator_id": annotator_id,
                "model": model,
                "video_stem": video_stem,
                "video_relpath": video_relpath,
                "video_exists": video_relpath is not None,
                "prompt_id": prompt_id,
                "prompt": prompt_entry.get("prompt"),
                "prompt_id_stem": prompt_entry.get("id_stem"),
                "prompt_physical_laws": prompt_entry.get("physical_laws") or [],
                "annotation_physical_laws": ann.get("physical_laws") or [],
                "na_laws": ann.get("na_laws") or [],
                "general_scores": (ann.get("scores") or {}).get("general") or {},
                "physical_scores": (ann.get("scores") or {}).get("physical") or {},
            }
            items.append(item)
            model_counter[model] += 1
            for law in item["annotation_physical_laws"]:
                law_counter[law] += 1

    return items, model_counter, law_counter


def mean_score(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def build_groups(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[item["prompt_id"]].append(item)

    groups: list[dict[str, Any]] = []
    for prompt_id, group_items in sorted(grouped.items()):
        first = group_items[0]
        by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in group_items:
            by_model[item["model"]].append(item)

        model_entries: list[dict[str, Any]] = []
        for model, model_items in sorted(by_model.items()):
            general_keys = sorted(
                {key for item in model_items for key in item.get("general_scores", {}).keys()}
            )
            physical_keys = sorted(
                {key for item in model_items for key in item.get("physical_scores", {}).keys()}
            )
            general_means = {
                key: mean_score(
                    [
                        float(item["general_scores"][key])
                        for item in model_items
                        if key in item.get("general_scores", {})
                    ]
                )
                for key in general_keys
            }
            physical_means = {
                key: mean_score(
                    [
                        float(item["physical_scores"][key])
                        for item in model_items
                        if key in item.get("physical_scores", {})
                    ]
                )
                for key in physical_keys
            }
            model_entries.append(
                {
                    "model": model,
                    "video_stem": model_items[0]["video_stem"],
                    "video_relpath": model_items[0]["video_relpath"],
                    "annotation_count": len(model_items),
                    "annotator_ids": sorted({item["annotator_id"] for item in model_items}),
                    "annotation_physical_laws": sorted(
                        {law for item in model_items for law in item["annotation_physical_laws"]}
                    ),
                    "na_laws": sorted({law for item in model_items for law in item["na_laws"]}),
                    "general_score_means": general_means,
                    "physical_score_means": physical_means,
                }
            )

        groups.append(
            {
                "group_id": f"prompt-{prompt_id:03d}",
                "prompt_id": prompt_id,
                "prompt_id_stem": first["prompt_id_stem"],
                "prompt": first["prompt"],
                "prompt_physical_laws": first["prompt_physical_laws"],
                "models": model_entries,
                "group_annotation_count": len(group_items),
                "model_count": len(model_entries),
            }
        )

    return groups


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_path = args.output.resolve()

    video_map, model_entries = build_video_map(dataset_root)
    prompt_map, prompt_entries = build_prompt_map(dataset_root)
    items, model_counter, law_counter = parse_annotations(dataset_root, prompt_map, video_map)
    groups = build_groups(items)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "total_models": len(model_entries),
        "total_videos": sum(entry["video_count"] for entry in model_entries),
        "total_prompts": len(prompt_entries),
        "total_annotations": len(items),
        "total_groups": len(groups),
        "annotated_models": dict(sorted(model_counter.items())),
        "annotated_laws": dict(sorted(law_counter.items())),
    }

    payload = {
        "summary": summary,
        "models": model_entries,
        "prompts": prompt_entries,
        "items": items,
        "groups": groups,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output_path}")
    print(f"annotations={len(items)} prompts={len(prompt_entries)} videos={summary['total_videos']}")


if __name__ == "__main__":
    main()
