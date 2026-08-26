#!/usr/bin/env python3
"""Build the offline CYCLES strict-dataset atlas."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path("/data/gaoya/AAA_test_video/physv_v2v_0819_strict")
TEMPLATE_ROOT = Path(__file__).resolve().parents[1] / "viewer"

FAMILY_LABELS = {
    "F11": "Table height",
    "F12": "Incline angle",
    "F12_RAMP_LENGTH": "Ramp length",
    "SCENE_DOOR_FRAME_BALL": "Ball · door frame",
    "SCENE_DOOR_FRAME": "Crate · door frame",
    "SCENE_PUCK_BARRIER": "Puck · barrier",
    "V2V_BOWL": "Bowl descent",
    "V2V_DOMINO": "Domino chain",
    "V2V_GAP": "Gap roll-off",
    "V2V_OBSTACLE_SIZE": "Obstacle size",
    "V2V_OBSTACLE": "Obstacle collision",
    "V2V_PENDULUM_CABINET": "Pendulum · cabinet",
    "V2V_PENDULUM": "Pendulum swing",
    "V2V_SEESAW": "Seesaw load",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path, root: Path) -> str:
    """Return a path from visualization/ to an artifact in the dataset root."""
    return "../" + path.relative_to(root).as_posix()


def contact_summary(path: Path) -> dict[str, Any]:
    entries = load_json(path) if path.is_file() else []
    frames = sorted({int(entry.get("frame", -1)) for entry in entries if int(entry.get("frame", -1)) >= 0})
    pairs: Counter[str] = Counter()
    forces: list[float] = []
    for entry in entries:
        pair = f"{entry.get('obj_a', '?')} × {entry.get('obj_b', '?')}"
        contacts = entry.get("contacts") or []
        if contacts:
            pairs[pair] += len(contacts)
        for contact in contacts:
            if contact.get("normal_force_n") is not None:
                forces.append(float(contact["normal_force_n"]))
    return {
        "contact_frames": len(frames),
        "first_contact_frame": frames[0] if frames else None,
        "last_contact_frame": frames[-1] if frames else None,
        "peak_normal_force_n": max(forces) if forces else 0.0,
        "top_pairs": [{"pair": pair, "count": count} for pair, count in pairs.most_common(3)],
    }


def actor_summary(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, actor in metadata.get("actors", {}).items():
        rows.append(
            {
                "name": name,
                "object_id": actor.get("object_id"),
                "role": actor.get("role"),
                "dynamic": bool(actor.get("dynamic")),
                "shape": actor.get("shape"),
                "mass_kg": actor.get("mass_kg"),
                "size_m": actor.get("size_m", {}),
                "initial_position_m": actor.get("initial_position_m", []),
            }
        )
    return rows


def build_case(root: Path, sample_dir: Path, manifest_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metadata = load_json(sample_dir / "metadata.json")
    captions = load_json(sample_dir / "captions/captions.json")
    case_id = sample_dir.name
    family_key = str(metadata.get("family_key", "UNKNOWN"))
    control = metadata.get("control") or {}
    truth_meta_path = root / "truth/cases" / case_id / "truth_metadata.json"
    truth_metadata = load_json(truth_meta_path) if truth_meta_path.is_file() else {}
    manifest = manifest_by_id.get(case_id, {})
    return {
        "id": case_id,
        "family_key": family_key,
        "family_label": FAMILY_LABELS.get(family_key, family_key.replace("_", " ").title()),
        "taxonomy": metadata.get("taxonomy", "Unknown"),
        "task_type": metadata.get("task_type"),
        "source_group": metadata.get("source_group"),
        "title": metadata.get("title", case_id),
        "control": {
            "variable": control.get("variable"),
            "value": control.get("value"),
            "value_label": control.get("value_label"),
            "units": control.get("units"),
        },
        "caption_specific": captions.get("specific", ""),
        "caption_abstract": captions.get("abstract", ""),
        "dynamic_objects": truth_metadata.get("dynamic_objects", [
            name for name, actor in metadata.get("actors", {}).items() if actor.get("dynamic")
        ]),
        "actors": actor_summary(metadata),
        "simulation": metadata.get("simulation", {}),
        "contacts": contact_summary(sample_dir / "contacts.json"),
        "checks": manifest.get("checks", {}),
        "paths": {
            "reference_video": rel(sample_dir / "videos/rgb_cycles.mp4", root),
            "context8": rel(sample_dir / "context/context8_cycles.mp4", root),
            "context16": rel(sample_dir / "context/context16_cycles.mp4", root),
            "first_frame": rel(sample_dir / "frames/00000.png", root),
            "metadata": rel(sample_dir / "metadata.json", root),
            "caption_bundle": rel(sample_dir / "captions/captions.json", root),
            "test_json": rel(root / "testjsons/v2v_jsons/physv_v2v_0819_all_cycles" / f"{case_id}.json", root),
            "dynamic_masks": rel(root / "truth/cases" / case_id / "dynamic_masks.npz", root),
            "depth": rel(root / "truth/cases" / case_id / "cycles_depth.npz", root),
            "trajectory_pixels": rel(root / "truth/cases" / case_id / "trajectory_pixels.npz", root),
            "rigidbench_metadata": rel(root / "truth/cases" / case_id / "rigidbench/metadata.json", root),
        },
    }


def build_data(root: Path) -> dict[str, Any]:
    manifest = load_json(root / "manifest.json")
    meta = load_json(root / "dataset_meta.json")
    manifest_by_id = {row["sample_id"]: row for row in manifest.get("samples", [])}
    cases = [
        build_case(root, sample_dir, manifest_by_id)
        for sample_dir in sorted((root / "samples").iterdir())
        if sample_dir.is_dir() and (sample_dir / "metadata.json").is_file()
    ]
    families: dict[str, dict[str, Any]] = {}
    for case in cases:
        family = families.setdefault(
            case["family_key"],
            {
                "key": case["family_key"],
                "label": case["family_label"],
                "taxonomy": case["taxonomy"],
                "task_type": case["task_type"],
                "source_group": case["source_group"],
                "case_count": 0,
                "cases": [],
            },
        )
        family["case_count"] += 1
        family["cases"].append(case["id"])
    taxonomy_counts = Counter(case["taxonomy"] for case in cases)
    source_counts = Counter(case["source_group"] for case in cases)
    return {
        "dataset": {
            "name": meta.get("dataset"),
            "schema_version": meta.get("schema_version"),
            "description": meta.get("description"),
            "source_selection": meta.get("source_selection"),
            "coordinate_system": meta.get("coordinate_system"),
            "mask_policy": meta.get("mask_policy"),
            "depth": meta.get("depth"),
            "contacts": meta.get("contacts"),
            "protocol": meta.get("protocol", {}),
            "sample_count": len(cases),
            "family_count": len(families),
            "taxonomy_counts": dict(taxonomy_counts),
            "source_counts": dict(source_counts),
            "official_rigidbench": False,
        },
        "families": list(families.values()),
        "cases": cases,
    }


def main() -> None:
    args = parse_args()
    root = args.dataset_root.expanduser().resolve()
    output = (args.output_dir or root / "visualization").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = build_data(root)
    (output / "data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(TEMPLATE_ROOT / "strict_dataset_atlas.html", output / "index.html")
    for filename in ("strict_dataset_atlas.css", "strict_dataset_atlas.js"):
        shutil.copy2(TEMPLATE_ROOT / filename, output / filename)
    (output / "README.md").write_text(
        """# PhysV V2V 0819 · Strict CYCLES 可视化

这是严格 benchmark 包的离线可视化入口，展示 70 个 CYCLES case、14 个控制变量 family、RGB reference video，以及每个 case 对应的 context、mask、depth、2D trajectory、接触记录和 RigidBench adapter。

## 查看

请从 strict 数据集根目录启动静态服务（不要从本目录启动，否则 `../samples` 等资源无法访问）：

```bash
cd /data/gaoya/AAA_test_video/physv_v2v_0819_strict
/usr/bin/python3 -m http.server 8861 --bind 0.0.0.0
```

然后打开 <http://localhost:8861/visualization/>。页面数据索引为 `data.json`；重新生成数据索引和静态资源：

```bash
cd /home/gaoya/Code_Video/Dataset_physv_v2v_0819
/usr/bin/python3 scripts/build_strict_dataset_visualization.py
```

协议：原生 `896×512 / 30 FPS / 90 frames`；reference 为 `samples/*/videos/rgb_cycles.mp4`。
""",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "cases": len(data["cases"]), "families": len(data["families"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
