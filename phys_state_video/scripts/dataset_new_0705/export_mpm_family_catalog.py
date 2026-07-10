#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

try:
    from .render_mpm_preview_case import build_family_case_catalog
except ImportError:
    from render_mpm_preview_case import build_family_case_catalog


DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/agent-data/outputs/dataset_new_0705/mpm_family_catalog_20260710")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export an MPM family catalog summary from the current preview case library.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def _case_record(case) -> dict[str, object]:
    return {
        "family": case.family,
        "case_key": case.key,
        "title": case.title,
        "description": case.description,
        "motion_profile": case.motion_profile,
        "scene_theme": case.scene_theme,
        "surface_key": case.surface_key,
        "lighting_key": case.lighting_key,
        "soft_material_key": case.soft_material_key,
        "soft_secondary_material_key": case.soft_secondary_material_key,
        "rigid_material_key": case.rigid_material_key,
        "mpm_vis_mode": case.mpm_vis_mode,
        "camera": {
            "res": list(case.camera.res),
            "pos": list(case.camera.pos),
            "lookat": list(case.camera.lookat),
            "fov": case.camera.fov,
        },
        "sim": {
            "dt": case.sim.dt,
            "substeps": case.sim.substeps,
            "horizon": case.sim.horizon,
            "gravity": list(case.sim.gravity),
            "mpm_lower_bound": list(case.sim.mpm_lower_bound),
            "mpm_upper_bound": list(case.sim.mpm_upper_bound),
            "grid_density": case.sim.grid_density,
        },
    }


def _markdown_table(records: list[dict[str, object]]) -> str:
    lines = [
        "# MPM Family Catalog",
        "",
        "当前这套 MPM 方案是 `family -> representative case` 的结构。",
        "也就是说，现在每个 family 先有一个代表性 preview case，后续批量化入口会以这份 catalog 为基准继续扩展。",
        "",
        "| Family | Case Key | Motion | Title | Soft | Rigid | Surface | Light | Horizon | dt | Grid | Vis |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for item in records:
        sim = item["sim"]
        lines.append(
            "| "
            f"{item['family']} | {item['case_key']} | {item['motion_profile']} | {item['title']} | "
            f"{item['soft_material_key']} | {item['rigid_material_key']} | {item['surface_key']} | "
            f"{item['lighting_key']} | {sim['horizon']} | {sim['dt']} | {sim['grid_density']} | {item['mpm_vis_mode']} |"
        )

    lines.append("")
    lines.append("## Family Notes")
    lines.append("")
    for item in records:
        extra = []
        if item["soft_secondary_material_key"]:
            extra.append(f"secondary soft={item['soft_secondary_material_key']}")
        extra_text = f"；{'；'.join(extra)}" if extra else ""
        lines.append(f"### {item['family']} · {item['case_key']}")
        lines.append(item["description"] + extra_text)
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    catalog = build_family_case_catalog()
    records = [_case_record(case) for family in sorted(catalog) for case in catalog[family]]
    family_counts = Counter(item["family"] for item in records)

    json_payload = {
        "family_count": len(family_counts),
        "case_count": len(records),
        "family_counts": dict(sorted(family_counts.items())),
        "records": records,
    }
    json_path = args.output_root / "mpm_family_catalog.json"
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown_path = args.output_root / "mpm_family_catalog.md"
    markdown_path.write_text(_markdown_table(records), encoding="utf-8")

    print(
        json.dumps(
            {
                "family_count": len(family_counts),
                "case_count": len(records),
                "json": str(json_path),
                "markdown": str(markdown_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
