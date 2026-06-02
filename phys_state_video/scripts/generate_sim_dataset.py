#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import generate_sim_preview_gallery as sim


DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/raw_v1/industrial_s1_pilot")
DEFAULT_THEME = "industrial"
DEFAULT_FAMILY_RATIOS = "F1=0.25,F2=0.30,F3=0.20,F4=0.15,F5=0.10"

FAMILY_SLUGS = {
    "F1 单物体运动": "F1_single_object",
    "F2 双体交互": "F2_two_object",
    "F3 多体连锁": "F3_chain_reaction",
    "F4 遮挡与重现": "F4_occlusion",
    "F5 支撑与跌落": "F5_drop_support",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a batch of industrial simulation raw samples.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--theme", default=DEFAULT_THEME, choices=sorted(sim.THEME_LABELS.keys()))
    parser.add_argument("--train-count", type=int, default=560)
    parser.add_argument("--val-count", type=int, default=70)
    parser.add_argument("--test-count", type=int, default=70)
    parser.add_argument("--family-ratios", default=DEFAULT_FAMILY_RATIOS)
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit-total", type=int, default=None)
    return parser.parse_args()


def parse_family_ratios(value: str) -> dict[str, float]:
    mapping: dict[str, float] = {}
    for item in value.split(","):
        token = item.strip()
        if not token:
            continue
        key, raw = token.split("=", 1)
        mapping[key.strip()] = float(raw.strip())
    total = sum(mapping.values())
    if total <= 0.0:
        raise ValueError("family ratios must sum to a positive value")
    return {key: value / total for key, value in mapping.items()}


def allocate_family_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    keys = ["F1", "F2", "F3", "F4", "F5"]
    raw = {key: total * ratios.get(key, 0.0) for key in keys}
    counts = {key: int(math.floor(value)) for key, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(keys, key=lambda key: raw[key] - counts[key], reverse=True)
    for key in order[:remainder]:
        counts[key] += 1
    return counts


def family_prefix(family: str) -> str:
    return family.split(" ", 1)[0]


def object_extent_xyz(obj: sim.ObjectSpec) -> np.ndarray:
    size = obj.size
    if obj.shape == "sphere":
        radius = float(size["radius"])
        return np.asarray([2.0 * radius, 2.0 * radius, 2.0 * radius], dtype=np.float32)
    if obj.shape == "box":
        return np.asarray([2.0 * size["hx"], 2.0 * size["hy"], 2.0 * size["hz"]], dtype=np.float32)
    if obj.shape == "cylinder":
        return np.asarray([2.0 * size["radius"], 2.0 * size["radius"], size["height"]], dtype=np.float32)
    if obj.shape == "capsule":
        return np.asarray([2.0 * size["radius"], 2.0 * size["radius"], size["height"] + 2.0 * size["radius"]], dtype=np.float32)
    if obj.shape == "puck":
        return np.asarray([2.0 * size["radius"], 2.0 * size["radius"], size["height"]], dtype=np.float32)
    raise ValueError(f"unsupported shape for extent estimation: {obj.shape}")


def jitter_object(obj: sim.ObjectSpec, rng: np.random.Generator) -> sim.ObjectSpec:
    new_obj = replace(obj)
    pos = np.asarray(obj.position, dtype=np.float32).copy()
    lin = np.asarray(obj.linear_velocity, dtype=np.float32).copy()
    ang = np.asarray(obj.angular_velocity, dtype=np.float32).copy()
    ori = np.asarray(obj.orientation_euler_deg, dtype=np.float32).copy()

    if obj.role == "dynamic":
        pos[0] += float(rng.uniform(-0.18, 0.18))
        pos[1] += float(rng.uniform(-0.10, 0.10))
        lin *= float(rng.uniform(0.90, 1.12))
        lin += rng.normal(0.0, 0.08, size=3).astype(np.float32)
        lin[2] += float(rng.uniform(-0.04, 0.04))
        ang *= float(rng.uniform(0.82, 1.20))
        ang += rng.normal(0.0, 0.45, size=3).astype(np.float32)
        ori += rng.normal(0.0, 5.0, size=3).astype(np.float32)
        new_obj.friction = float(np.clip(obj.friction + rng.uniform(-0.08, 0.08), 0.25, 0.98))
        new_obj.restitution = float(np.clip(obj.restitution + rng.uniform(-0.08, 0.08), 0.01, 0.95))
    else:
        pos[0] += float(rng.uniform(-0.05, 0.05))
        pos[1] += float(rng.uniform(-0.04, 0.04))
        ori += rng.normal(0.0, 2.0, size=3).astype(np.float32)
        new_obj.friction = float(np.clip(obj.friction + rng.uniform(-0.04, 0.04), 0.25, 0.99))

    floor_z = object_extent_xyz(obj)[2] * 0.5
    if obj.dynamic and obj.position[2] <= floor_z + 0.03:
        pos[2] = obj.position[2]
        lin[2] = obj.linear_velocity[2]
    else:
        pos[2] += float(rng.uniform(-0.03, 0.03))

    new_obj.position = [float(value) for value in pos]
    new_obj.linear_velocity = [float(value) for value in lin]
    new_obj.angular_velocity = [float(value) for value in ang]
    new_obj.orientation_euler_deg = [float(value) for value in ori]
    return new_obj


def jitter_scenario(template: sim.ScenarioSpec, sample_key: str, seed: int) -> sim.ScenarioSpec:
    rng = np.random.default_rng(seed)
    objects = [jitter_object(obj, rng) for obj in template.objects]
    floor_friction = float(np.clip(template.floor_friction + rng.uniform(-0.08, 0.08), 0.35, 0.98))
    pre_roll_s = float(np.clip(template.pre_roll_s + rng.uniform(-0.05, 0.05), 0.02, 0.90))
    return replace(
        template,
        key=sample_key,
        seed=seed,
        floor_friction=floor_friction,
        pre_roll_s=pre_roll_s,
        objects=objects,
    )


def build_sample_plan(split: str, total: int, templates: list[sim.ScenarioSpec], ratios: dict[str, float], start_index: int) -> list[tuple[str, sim.ScenarioSpec, int]]:
    by_family: dict[str, list[sim.ScenarioSpec]] = {}
    for template in templates:
        by_family.setdefault(template.family, []).append(template)

    counts = allocate_family_counts(total, ratios)
    plan: list[tuple[str, sim.ScenarioSpec, int]] = []
    cursor = start_index
    for family in sorted(by_family.keys(), key=family_prefix):
        prefix = family_prefix(family)
        target = counts.get(prefix, 0)
        pool = by_family[family]
        for local_idx in range(target):
            template = pool[local_idx % len(pool)]
            sample_id = f"sample_{cursor:06d}"
            plan.append((sample_id, template, local_idx))
            cursor += 1
    return plan


def prepare_sample_dir(root: Path, overwrite: bool) -> None:
    if overwrite and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


def normalize_sample_outputs(sample_dir: Path, sample_key: str) -> tuple[Path, Path, Path]:
    video_src = sample_dir / f"{sample_key}.mp4"
    meta_src = sample_dir / f"{sample_key}.json"
    states_src = sample_dir / f"{sample_key}_states.npz"
    video_dst = sample_dir / "video.mp4"
    meta_dst = sample_dir / "meta.json"
    states_dst = sample_dir / "states.npz"
    video_src.replace(video_dst)
    meta_src.replace(meta_dst)
    states_src.replace(states_dst)
    return video_dst, meta_dst, states_dst


def generate_split(split: str, total: int, templates: list[sim.ScenarioSpec], ratios: dict[str, float], output_root: Path, seed: int, start_index: int, overwrite: bool) -> tuple[list[dict], int]:
    records: list[dict] = []
    plan = build_sample_plan(split, total, templates, ratios, start_index)
    family_counter: Counter[str] = Counter()
    if not plan:
        return records, start_index

    sim.p.connect(sim.p.DIRECT)
    try:
        for global_idx, (sample_id, template, family_local_idx) in enumerate(plan):
            family_slug = FAMILY_SLUGS[template.family]
            sample_dir = output_root / split / family_slug / sample_id
            if sample_dir.exists() and not overwrite and (sample_dir / "meta.json").exists():
                print(f"[skip] {split}/{family_slug}/{sample_id}")
                records.append({
                    "sample_id": sample_id,
                    "split": split,
                    "family": template.family,
                    "family_slug": family_slug,
                    "template_key": template.key,
                    "path": str(sample_dir),
                    "status": "skipped",
                })
                continue

            prepare_sample_dir(sample_dir, overwrite=overwrite)
            sample_seed = seed + global_idx + start_index
            sample_key = f"{family_prefix(template.family).lower()}_{sample_id}"
            scenario = jitter_scenario(template, sample_key, sample_seed)
            sim.OUTPUT_ROOT = sample_dir
            sim.VIDEO_DIR = sample_dir
            sim.META_DIR = sample_dir
            renderer = sim.PreviewRenderer()
            try:
                meta = sim.run_scenario(renderer, scenario, overlay_text=False)
            finally:
                renderer.cleanup()

            video_path, meta_path, states_path = normalize_sample_outputs(sample_dir, scenario.key)
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            payload["video"] = str(video_path)
            payload["states"] = str(states_path)
            payload["template_key"] = template.key
            payload["sample_id"] = sample_id
            payload["split"] = split
            payload["family_slug"] = family_slug
            payload["variation_index"] = family_local_idx
            payload["render_overlay_text"] = False
            meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            family_counter[family_slug] += 1
            record = {
                "sample_id": sample_id,
                "split": split,
                "family": template.family,
                "family_slug": family_slug,
                "template_key": template.key,
                "path": str(sample_dir),
                "seed": sample_seed,
            }
            records.append(record)
            print(f"[done] {split}/{family_slug}/{sample_id} <- {template.key}")
    finally:
        sim.p.disconnect()

    return records, start_index + len(plan)


def main() -> None:
    args = parse_args()
    ratios = parse_family_ratios(args.family_ratios)
    templates = sim.apply_theme_to_scenarios(sim.build_preview_scenarios(), args.theme)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifests").mkdir(parents=True, exist_ok=True)

    split_specs = [
        ("train", args.train_count),
        ("val", args.val_count),
        ("test", args.test_count),
    ]
    if args.limit_total is not None:
        remaining = args.limit_total
        adjusted = []
        for split, count in split_specs:
            clipped = min(count, max(remaining, 0))
            adjusted.append((split, clipped))
            remaining -= clipped
        split_specs = adjusted

    next_index = args.start_index
    all_records: list[dict] = []
    split_manifests: dict[str, dict] = {}
    for split, count in split_specs:
        records, next_index = generate_split(
            split=split,
            total=count,
            templates=templates,
            ratios=ratios,
            output_root=output_root,
            seed=args.seed,
            start_index=next_index,
            overwrite=args.overwrite,
        )
        split_manifest = {
            "split": split,
            "count": len(records),
            "theme": args.theme,
            "records": records,
        }
        split_manifests[split] = split_manifest
        (output_root / "manifests" / f"{split}.json").write_text(
            json.dumps(split_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        all_records.extend(records)

    summary = {
        "theme": args.theme,
        "output_root": str(output_root),
        "splits": {split: manifest["count"] for split, manifest in split_manifests.items()},
        "family_ratios": ratios,
        "total_records": len(all_records),
        "start_index": args.start_index,
        "next_index": next_index,
    }
    (output_root / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
