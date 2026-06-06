#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Output_try0526/bench_jsons_mer")

A_GT_ROOT = Path("/data/gaoya/AAA_test_video/Output_try0526/PDI-Bench/output/GT")
B_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0526dp/videos")
D_GT_ROOT = Path("/data/gaoya/AAA_test_video/Output_try0526/physics-iq-benchmark/output/GT/physics-iq-benchmark")
E_ROOT = Path("/data/gaoya/AAA_test_video/Output_try0526/phygenbench/output/FLUX_1_Kontext/phygenbench")

B1_ROOT = B_ROOT / "ball_block"
B2_ROOT = B_ROOT / "jepa_sensitivity"
B3_ROOT = B_ROOT / "ball_block_appearance"
C_ROOT = B_ROOT / "shuffle_test"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def abs_path_str(value: str | Path | None) -> str | None:
    if value is None:
        return None
    path = Path(value)
    return str(path if path.is_absolute() else path.resolve(strict=False))


def compact_item(
    *,
    category: str | None = None,
    source_video: str | Path | None = None,
    caption: str | None = None,
    first_frame: str | Path | None = None,
    context_video: str | Path | None = None,
) -> dict[str, str]:
    item: dict[str, str] = {}
    if category:
        item["category"] = str(category)
    if source_video:
        item["source_video"] = abs_path_str(source_video)  # type: ignore[assignment]
    if caption:
        item["caption"] = str(caption)
    if first_frame:
        item["first_frame"] = abs_path_str(first_frame)  # type: ignore[assignment]
    if context_video:
        item["context_video"] = abs_path_str(context_video)  # type: ignore[assignment]
    return item


def write_group_json(group_id: str, benchmark: str, task_map: dict[str, list[dict[str, str]]]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for task in sorted(task_map):
        rows.extend(task_map[task])
    out_path = OUTPUT_ROOT / f"{group_id}.json"
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def build_group_a() -> None:
    tasks: dict[str, list[dict[str, str]]] = defaultdict(list)
    for json_path in sorted(A_GT_ROOT.glob("*/*.json")):
        payload = load_json(json_path)
        task = json_path.parent.name
        tasks[task].append(
            compact_item(
                category=task,
                source_video=payload.get("source") or payload.get("video") or payload.get("video_path"),
                caption=payload.get("prompt") or payload.get("caption"),
                first_frame=payload.get("first_frame"),
                context_video=payload.get("context_video"),
            )
        )
    write_group_json("A", "PDI-Bench", tasks)


def build_group_b() -> None:
    tasks: dict[str, list[dict[str, str]]] = defaultdict(list)

    for json_path in sorted(B1_ROOT.glob("*.json")):
        payload = load_json(json_path)
        tasks["B1_ball_block_physics"].append(
            compact_item(
                category="B1_ball_block_physics",
                source_video=payload.get("video") or payload.get("video_path"),
                caption=payload.get("caption") or payload.get("prompt"),
                first_frame=payload.get("first_frame"),
                context_video=payload.get("context_video"),
            )
        )

    b2_task_names = {
        "blk_": "B2_block_mass",
        "grav_": "B2_gravity",
        "mass_": "B2_ball_mass",
        "rev_": "B2_restitution",
        "vel_": "B2_velocity",
        "nomiss": "B2_direct_hit",
    }
    for json_path in sorted(B2_ROOT.glob("*.json")):
        payload = load_json(json_path)
        stem = json_path.stem
        task_name = None
        for prefix, mapped in b2_task_names.items():
            if stem == prefix or stem.startswith(prefix):
                task_name = mapped
                break
        if task_name is None:
            task_name = "B2_other"
        tasks[task_name].append(
            compact_item(
                category=task_name,
                source_video=payload.get("video") or payload.get("video_path"),
                caption=payload.get("caption") or payload.get("prompt"),
                first_frame=payload.get("first_frame"),
                context_video=payload.get("context_video"),
            )
        )

    b3_task_names = {
        "_v1_default": "B3_default_render",
        "_v2_dark_blue": "B3_dark_blue_render",
        "_v3_warm_bright": "B3_warm_bright_render",
    }
    for json_path in sorted(B3_ROOT.glob("*.json")):
        payload = load_json(json_path)
        stem = json_path.stem
        task_name = "B3_other_render"
        for suffix, mapped in b3_task_names.items():
            if stem.endswith(suffix):
                task_name = mapped
                break
        tasks[task_name].append(
            compact_item(
                category=task_name,
                source_video=payload.get("video") or payload.get("video_path"),
                caption=payload.get("caption") or payload.get("prompt"),
                first_frame=payload.get("first_frame"),
                context_video=payload.get("context_video"),
            )
        )

    write_group_json("B", "Dataset_physV", tasks)


def resolve_c_caption(stem: str) -> str | None:
    if stem.startswith("gt_"):
        clip_name = stem.removeprefix("gt_")
        if clip_name.endswith("_shuffled"):
            clip_name = clip_name[: -len("_shuffled")]
        gt_candidates = list(A_GT_ROOT.glob(f"*/*{clip_name}.json"))
        for candidate in gt_candidates:
            payload = load_json(candidate)
            caption = payload.get("prompt") or payload.get("caption")
            if isinstance(caption, str) and caption.strip():
                return caption
        return None

    if stem.startswith("sim_"):
        sample_name = stem.removeprefix("sim_")
        if sample_name.endswith("_shuffled"):
            sample_name = sample_name[: -len("_shuffled")]
        for root in (B1_ROOT, B2_ROOT):
            candidate = root / f"{sample_name}.json"
            if candidate.is_file():
                payload = load_json(candidate)
                caption = payload.get("caption") or payload.get("prompt")
                if isinstance(caption, str) and caption.strip():
                    return caption
        return None

    return None


def resolve_c_original_payload(stem: str) -> dict[str, Any] | None:
    if stem == "gt_ball_original":
        path = C_ROOT / "gt_ball_original.json"
        return load_json(path) if path.is_file() else None

    if stem.startswith("gt_"):
        clip_name = stem.removeprefix("gt_")
        if clip_name.endswith("_shuffled"):
            clip_name = clip_name[: -len("_shuffled")]
        gt_candidates = sorted(A_GT_ROOT.glob(f"*/*{clip_name}.json"))
        if gt_candidates:
            return load_json(gt_candidates[0])
        return None

    if stem.startswith("sim_"):
        sample_name = stem.removeprefix("sim_")
        if sample_name.endswith("_shuffled"):
            sample_name = sample_name[: -len("_shuffled")]
        for root in (B1_ROOT, B2_ROOT):
            candidate = root / f"{sample_name}.json"
            if candidate.is_file():
                return load_json(candidate)
        return None

    return None


def classify_c_task(stem: str) -> str:
    if stem == "gt_ball_original":
        return "reference_original"
    if stem.startswith("gt_"):
        return "pdi_bench_original"
    if stem.startswith("sim_e"):
        return "dataset_physv_ball_block_original"
    if stem.startswith("sim_"):
        return "dataset_physv_jepa_sensitivity_original"
    return "other_original"


def build_group_c() -> None:
    tasks: dict[str, list[dict[str, str]]] = defaultdict(list)
    for json_path in sorted(C_ROOT.glob("*.json")):
        stem = json_path.stem
        payload = resolve_c_original_payload(stem)
        if payload is None:
            continue
        tasks[classify_c_task(stem)].append(
            compact_item(
                category=classify_c_task(stem),
                source_video=payload.get("video") or payload.get("video_path"),
                caption=payload.get("caption") or payload.get("prompt") or resolve_c_caption(stem),
                first_frame=payload.get("first_frame"),
                context_video=payload.get("context_video"),
            )
        )
    write_group_json("C", "Dataset_physV_shuffle_test", tasks)


def build_group_d() -> None:
    tasks: dict[str, list[dict[str, str]]] = defaultdict(list)
    for json_path in sorted(D_GT_ROOT.glob("*.json")):
        payload = load_json(json_path)
        task = str(payload.get("category") or "UNKNOWN")
        source_video = (
            (payload.get("paths") or {}).get("full_video_path")
            or payload.get("full_video")
            or payload.get("video")
            or payload.get("video_path")
        )
        tasks[task].append(
            compact_item(
                category=task,
                source_video=source_video,
                caption=payload.get("caption") or payload.get("prompt"),
                first_frame=payload.get("first_frame"),
                context_video=payload.get("context_video"),
            )
        )
    write_group_json("D", "Physics-IQ", tasks)


def build_group_e() -> None:
    tasks: dict[str, list[dict[str, str]]] = defaultdict(list)
    for json_path in sorted(E_ROOT.glob("*.json")):
        payload = load_json(json_path)
        task = f"{payload.get('main_category') or 'UNKNOWN'} / {payload.get('sub_category') or 'UNKNOWN'}"
        tasks[task].append(
            compact_item(
                category=task,
                source_video=payload.get("source_video"),
                caption=payload.get("caption") or payload.get("prompt"),
                first_frame=payload.get("first_frame"),
                context_video=payload.get("context_video"),
            )
        )
    write_group_json("E", "PhyGenBench", tasks)


def main() -> None:
    build_group_a()
    build_group_b()
    build_group_c()
    build_group_d()
    build_group_e()


if __name__ == "__main__":
    main()
