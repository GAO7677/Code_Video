#!/usr/bin/env python3
"""Make R002 dynamic-object appearance consistent within each control group.

The first R002 visible-texture pass selected a texture independently per case.
This repair treats ``family_key`` as the control-variable group and assigns one
visible texture material to every case in that group. Existing v2 outputs are
preserved; a completed output is reused only when its old material already
matches the new family assignment. All other outputs are left pending for the
normal sequential renderer.
"""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path


ROOT = Path(
    "/data/gaoya/AAA_test_video/physv_v2v_0819_strict/refine/"
    "R002_all_cases_distinct_texture_20260829"
)

# One material per independently controlled family. These choices were made by
# maximizing the minimum display-space color distance against the floor, wall,
# trim, and static/dynamic parent materials across the whole family.
FAMILY_MATERIALS = {
    "F11": "refine_teal_texture",
    "F12": "refine_blue_texture",
    "F12_RAMP_LENGTH": "refine_blue_texture",
    "SCENE_DOOR_FRAME": "refine_blue_texture",
    "SCENE_DOOR_FRAME_BALL": "refine_yellow_texture",
    "SCENE_PUCK_BARRIER": "refine_yellow_texture",
    "V2V_BOWL": "refine_yellow_texture",
    "V2V_DOMINO": "refine_blue_texture",
    "V2V_GAP": "refine_blue_texture",
    "V2V_OBSTACLE": "refine_yellow_texture",
    "V2V_OBSTACLE_SIZE": "refine_charcoal_texture",
    "V2V_PENDULUM": "refine_yellow_texture",
    "V2V_PENDULUM_CABINET": "refine_blue_texture",
    "V2V_SEESAW": "refine_blue_texture",
}
VERSION = "visible_texture_v3_family_consistent"
ARCHIVED_VERSION = "visible_texture_v2_mixed_family"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def archive_once(source: Path, destination: Path) -> bool:
    if not source.exists():
        return False
    if destination.exists():
        raise RuntimeError(f"archive destination already exists: {destination}")
    shutil.move(str(source), str(destination))
    return True


def patch_render_report(path: Path, selection: dict) -> None:
    if not path.is_file():
        return
    report = load(path)
    report["selection"] = selection["selected"]
    report["family_consistent_texture"] = {
        "version": VERSION,
        "family_key": selection["family_key"],
        "family_material": selection["selected"]["selected_material"],
    }
    dump(path, report)


def main() -> None:
    manifest_path = ROOT / "case_selection.jsonl"
    archive_manifest = ROOT / f"case_selection_{ARCHIVED_VERSION}.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if archive_manifest.exists():
        raise RuntimeError(f"repair appears to have run already: {archive_manifest}")

    original_lines = [line for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    selections = [json.loads(line) for line in original_lines]
    if len(selections) != 70:
        raise RuntimeError(f"expected 70 R002 selections, found {len(selections)}")
    families = {item["family_key"] for item in selections}
    missing = sorted(families - set(FAMILY_MATERIALS))
    if missing:
        raise RuntimeError(f"no family material configured for: {missing}")

    # Preserve the exact v2 manifest and the current generated index.
    shutil.copy2(manifest_path, archive_manifest)
    index_path = ROOT / "index.html"
    if index_path.is_file():
        shutil.copy2(index_path, ROOT / f"index_{ARCHIVED_VERSION}.html")

    experiment_path = ROOT / "experiment.json"
    experiment = load(experiment_path)
    experiment.setdefault("rerender_history", []).append({
        "version": ARCHIVED_VERSION,
        "status": experiment.get("status", "partial"),
        "archived_manifest": str(archive_manifest),
        "archived_render_subdirs": [
            "render/full_visible_texture_v2_mixed_family",
            "render/smoke_visible_texture_v2_mixed_family",
        ],
        "reason": "v2 selected texture independently per case; archived before family-consistent repair",
    })
    experiment["visible_texture_repair"] = {
        "version": VERSION,
        "status": "planned",
        "texture_package": "wood_peeling_paint_weathered",
        "family_key_definition": "one independently controlled-variable group",
        "family_materials": FAMILY_MATERIALS,
        "appearance_invariant": "all dynamic actors in one family use the same visible texture and palette",
        "old_outputs_preserved_under": [
            "render/full_visible_texture_v2_mixed_family",
            "render/smoke_visible_texture_v2_mixed_family",
        ],
    }
    experiment["status"] = "planned"
    experiment["progress"] = {
        "completed": 0,
        "failed": 0,
        "pending": len(selections),
        "total": len(selections),
    }
    experiment.setdefault("runs", {})["full"] = {
        "status": "planned", "completed": 0, "failed": 0, "total": len(selections),
    }
    experiment["runs"][ARCHIVED_VERSION] = {
        "status": "archived", "total": len(selections),
    }

    new_lines = []
    reused = 0
    rerender = 0
    for selection in selections:
        case_id = selection["case_id"]
        family = selection["family_key"]
        family_material = FAMILY_MATERIALS[family]
        case_root = ROOT / "cases" / case_id
        selection_path = case_root / "selection.json"
        overrides_path = case_root / "material_overrides.json"
        old_selection = load(selection_path)
        old_material = old_selection["selected"]["selected_material"]

        # Preserve the currently visible v2 selection and outputs before
        # replacing the active selection with the family-consistent one.
        shutil.copy2(selection_path, case_root / f"selection_{ARCHIVED_VERSION}.json")
        if overrides_path.is_file():
            shutil.copy2(overrides_path, case_root / f"material_overrides_{ARCHIVED_VERSION}.json")
        old_full = case_root / "render" / "full"
        old_smoke = case_root / "render" / "smoke"
        archived_full = case_root / "render" / f"full_{ARCHIVED_VERSION}"
        archived_smoke = case_root / "render" / f"smoke_{ARCHIVED_VERSION}"
        had_full = archive_once(old_full, archived_full)
        archive_once(old_smoke, archived_smoke)

        updated = copy.deepcopy(selection)
        updated["status"] = "planned"
        updated["selected"]["previous_visible_texture_material"] = old_material
        updated["selected"]["selected_material"] = family_material
        updated["selected"]["override"] = {
            actor_name: family_material
            for actor_name in selection["selected"]["override"]
        }
        updated["selected"]["selection_pool"] = [family_material]
        updated["selected"]["family_material"] = family_material
        updated["selected"]["family_assignment_policy"] = "fixed_by_family_key"
        updated["selected"]["visible_texture_repair"] = {
            **updated["selected"].get("visible_texture_repair", {}),
            "version": VERSION,
            "family_key": family,
            "family_material": family_material,
            "previous_visible_texture_material": old_material,
        }

        dump(selection_path, updated)
        dump(overrides_path, updated["selected"]["override"])

        # If the already rendered v2 material is exactly the family material,
        # restore it instead of rendering identical pixels a second time.
        if had_full and old_material == family_material:
            shutil.move(str(archived_full), str(old_full))
            patch_render_report(old_full / "render_metadata.json", updated)
            reused += 1
        else:
            rerender += 1
        if old_material == family_material and archived_smoke.is_dir():
            shutil.move(str(archived_smoke), str(old_smoke))
            patch_render_report(old_smoke / "render_metadata.json", updated)

        new_lines.append(json.dumps(updated, ensure_ascii=False, separators=(",", ":")))

    manifest_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    dump(experiment_path, experiment)

    for log_name in ("gpu5_full.results.jsonl", "gpu5_smoke.results.jsonl"):
        log_path = ROOT / "logs" / log_name
        if log_path.is_file():
            archived_log = log_path.with_name(
                log_path.name.replace(".results.jsonl", f"_{ARCHIVED_VERSION}.results.jsonl")
            )
            shutil.move(str(log_path), str(archived_log))

    print(json.dumps({
        "version": VERSION,
        "families": FAMILY_MATERIALS,
        "cases": len(selections),
        "reused_completed_outputs": reused,
        "needs_rerender": rerender,
        "archived_manifest": str(archive_manifest),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
