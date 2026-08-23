#!/usr/bin/env python3
"""Build the Utonia No-Scene/Scene-Enabled three-test matrix dashboard.

The page is intentionally data-driven: checkpoint discovery and video
existence are evaluated every time the 8844 server serves dashboard.json.
This keeps a partially generated cell visible as ``pending`` without
rebuilding the HTML page or copying large video files.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote


HUB_ROOT = Path("/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_hub")
PAGE_ROOT = HUB_ROOT / "utonia-scene-weights-no-scene-three-tests"
PAGE_KEY = "full_sa_physrvg_vjepa_utonia_scene_hardmask_v1_b2gacc2"
ENABLED_PAGE_KEY = "full_sa_physrvg_vjepa_utonia_scene_hardmask_v1_enabled"
ENABLED_RAW_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "physrvg_full_sa_vjepa_utonia_scene_enabled_eval"
)

TEST5_LIST = Path("/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt")
PHYSICIQ_LIST = Path("/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt")
TEST70_LIST = Path(
    "/data/gaoya/AAA_test_video/physv_v2v_0819/testjsons/"
    "physv_v2v_0819_all_cycles_test70_ctx8.txt"
)

TEST70_PAGE = HUB_ROOT / "physv-v2v-0819-utonia-no-scene-test70"
TEST70_ARTIFACT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/physv_v2v_0819_utonia_no_scene_test70"
)

TEST5_RAW_ROOT = Path(
    "/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_three_run_watch/results"
)
TEST5_MEDIA_ROOT = Path(
    "/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_three_run_watch/site/videos/media"
)
PHYSICIQ_RAW_ROOT = Path("/data/gaoya/AAA_test_video/0623/test/v2v_wan/xssc")
PHYSICIQ_MEDIA_ROOT = Path(
    "/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_three_run_watch/"
    "site/physiciq-videos/media"
)

B2_ROOT = Path(
    "/data/gaoya/agent-data/checkpoints/physrvg_full_sa_vjepa_utonia_scene/"
    "full-sa-pybullet-physrvg-vjepa-utonia-scene-hardmask-v1-formal-"
    "b2-gacc2-bf16-restart-20260821/checkpoints"
)
B4_ROOT = Path(
    "/data/gaoya/agent-data/checkpoints/physrvg_full_sa_vjepa_utonia_scene/"
    "full-sa-pybullet-physrvg-vjepa-utonia-scene-hardmask-v1-formal-"
    "b4-gacc1-resume1000-20260822/checkpoints"
)

BRANCHES = (
    {
        "key": "b2gacc2",
        "branch": "b2-gacc2",
        "label": (
            "PHYRVG-Full-SA + V-JEPA Loss · Utonia Scene Weights · "
            "No-Scene · formal · b2-gacc2"
        ),
        "short_label": "Utonia · b2-gacc2",
        "color": "#59A14F",
        "mode": "no_scene",
        "root": B2_ROOT,
    },
    {
        "key": "b4gacc1",
        "branch": "b4-gacc1",
        "label": (
            "PHYRVG-Full-SA + V-JEPA Loss · Utonia Scene Weights · "
            "No-Scene · formal · b4-gacc1 · resume1000"
        ),
        "short_label": "Utonia · b4-gacc1 · resume1000",
        "color": "#76B7B2",
        "mode": "no_scene",
        "root": B4_ROOT,
    },
    {
        "key": "enabled_b2gacc2",
        "branch": "Scene Enabled · b2-gacc2",
        "label": (
            "PHYRVG-Full-SA + V-JEPA Loss · Utonia Scene Weights · "
            "Scene Enabled · formal · b2-gacc2"
        ),
        "short_label": "Scene Enabled · b2-gacc2",
        "color": "#F28E2B",
        "mode": "scene_enabled",
        "root": B2_ROOT,
    },
    {
        "key": "enabled_b4gacc1",
        "branch": "Scene Enabled · b4-gacc1",
        "label": (
            "PHYRVG-Full-SA + V-JEPA Loss · Utonia Scene Weights · "
            "Scene Enabled · formal · b4-gacc1 · resume1000"
        ),
        "short_label": "Scene Enabled · b4-gacc1 · resume1000",
        "color": "#E15759",
        "mode": "scene_enabled",
        "root": B4_ROOT,
    },
)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _quote(value: str) -> str:
    return quote(str(value), safe="")


def _existing_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _checkpoint_is_valid(path: Path) -> bool:
    required = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "physrvg_utonia_scene_manifest.json",
        "physrvg_utonia_scene_trainable.safetensors",
    )
    return all(_existing_file(path / name) for name in required)


def _discover_weights() -> list[dict]:
    weights: list[dict] = []
    step_pattern = re.compile(r"^step-(\d+)$")
    for branch in BRANCHES:
        root = branch["root"]
        if not root.is_dir():
            continue
        for checkpoint in root.iterdir():
            match = step_pattern.match(checkpoint.name)
            if not match or not checkpoint.is_dir() or not _checkpoint_is_valid(checkpoint):
                continue
            step = int(match.group(1))
            if branch["mode"] == "no_scene":
                task_id = (
                    "full_sa_physrvg_vjepa_utonia_scene_"
                    f"{branch['key']}__step-{step:06d}"
                )
            else:
                task_id = f"{ENABLED_PAGE_KEY}__step-{step:06d}"
            weights.append(
                {
                    "task_id": task_id,
                    "branch_key": branch["key"],
                    "branch": branch["branch"],
                    "label": branch["label"],
                    "short_label": branch["short_label"],
                    "color": branch["color"],
                    "step": step,
                    "step_label": f"step-{step:06d}",
                    "column_label": f"{branch['branch']} · step-{step:06d}",
                    "checkpoint_dir": str(checkpoint),
                    "mode": branch["mode"],
                }
            )
    return sorted(
        weights,
        key=lambda item: (
            0 if item["mode"] == "no_scene" else 1,
            item["step"],
            item["branch_key"],
        ),
    )


def _family_for_test5(stem: str) -> str:
    if stem.startswith("0613pybullet_"):
        return "PyBullet"
    if stem.startswith("physicIQ_"):
        return "PhysicIQ"
    if stem.startswith("phyco_kubric_"):
        return "PHYCO Kubric"
    return "Other"


def _family_for_physiciq(stem: str) -> str:
    parts = stem.split("_")
    if len(parts) >= 3 and parts[0].lower() == "physiciq":
        if parts[1].isdigit():
            return parts[2].replace("-", " ")
        return parts[1].replace("-", " ")
    return "PhysicIQ"


def _case_from_json(path: Path, family: str) -> dict:
    payload = _read_json(path)
    stem = path.stem
    caption = str(payload.get("input_caption") or "").strip()
    return {
        "case_id": stem,
        "stem": stem,
        "family_key": family,
        "title": caption or stem,
        "caption": caption,
        "input_json": str(path),
    }


def _load_path_cases(path: Path, family_fn) -> list[dict]:
    cases: list[dict] = []
    if not path.is_file():
        return cases
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        input_path = Path(raw_line)
        if input_path.is_file():
            cases.append(_case_from_json(input_path, family_fn(input_path.stem)))
        else:
            stem = input_path.stem
            cases.append(
                {
                    "case_id": stem,
                    "stem": stem,
                    "family_key": family_fn(stem),
                    "title": stem,
                    "caption": "",
                    "input_json": raw_line,
                }
            )
    return cases


def _load_test70_cases() -> list[dict]:
    dashboard = _read_json(TEST70_PAGE / "dashboard.json")
    cases = dashboard.get("cases")
    if isinstance(cases, list) and cases:
        return [
            {
                "case_id": str(item.get("case_id") or item.get("stem") or ""),
                "stem": str(item.get("stem") or item.get("case_id") or ""),
                "family_key": str(item.get("family_key") or "未分组"),
                "title": str(item.get("title") or item.get("stem") or ""),
                "caption": str(item.get("caption") or ""),
                "input_json": str(item.get("input_json") or ""),
            }
            for item in cases
            if item.get("stem") or item.get("case_id")
        ]
    return _load_path_cases(TEST70_LIST, lambda stem: "test70")


def _attach_case_urls(cases: list[dict], test_key: str) -> list[dict]:
    result: list[dict] = []
    for case in cases:
        item = dict(case)
        stem = str(item["stem"])
        if test_key == "test5":
            gt_path = TEST5_MEDIA_ROOT / "_source" / stem / "gt_49f_30fps.mp4"
            item["gt_url"] = (
                f"../gallery/media/_source/{_quote(stem)}/gt_49f_30fps.mp4"
                if _existing_file(gt_path)
                else ""
            )
        elif test_key == "physiciq":
            gt_path = PHYSICIQ_MEDIA_ROOT / "_source" / stem / "gt_49f_30fps.mp4"
            item["gt_url"] = (
                f"../physiciq-gallery/media/_source/{_quote(stem)}/gt_49f_30fps.mp4"
                if _existing_file(gt_path)
                else ""
            )
        else:
            gt_path = TEST70_ARTIFACT_ROOT / "ground_truth" / stem / "source.mp4"
            context_path = TEST70_ARTIFACT_ROOT / "ground_truth" / stem / "context.mp4"
            item["gt_url"] = (
                f"../physv-v2v-0819-utonia-no-scene-test70/ground_truth/"
                f"{_quote(stem)}/source.mp4"
                if _existing_file(gt_path)
                else ""
            )
            item["context_url"] = (
                f"../physv-v2v-0819-utonia-no-scene-test70/ground_truth/"
                f"{_quote(stem)}/context.mp4"
                if _existing_file(context_path)
                else ""
            )
        result.append(item)
    return result


def _task_state(task_id: str) -> dict:
    return _read_json(TEST70_ARTIFACT_ROOT / "state" / "tasks" / f"{task_id}.json")


def _display_video_path(test_key: str, weight: dict, stem: str) -> Path:
    step = weight["step"]
    page_key = ENABLED_PAGE_KEY if weight["mode"] == "scene_enabled" else PAGE_KEY
    if test_key == "test5":
        return TEST5_MEDIA_ROOT / page_key / f"step-{step:06d}" / f"{stem}.mp4"
    if test_key == "physiciq":
        return PHYSICIQ_MEDIA_ROOT / page_key / f"step-{step:06d}" / f"{stem}.mp4"
    return TEST70_ARTIFACT_ROOT / "results" / weight["task_id"] / f"{stem}.mp4"


def _raw_video_candidates(test_key: str, weight: dict, stem: str) -> list[Path]:
    step = weight["step"]
    if weight["mode"] == "scene_enabled":
        if test_key == "test5":
            return [
                ENABLED_RAW_ROOT / "test5" / (
                    f"full_sa_physrvg_vjepa_utonia_scene_enabled_step-{step:06d}_"
                    "steps8_512x896_ctx08_49f"
                ) / f"{stem}.mp4"
            ]
        if test_key == "physiciq":
            return [
                ENABLED_RAW_ROOT / "physiciq" / (
                    f"full_sa_physrvg_vjepa_utonia_scene_enabled_step-{step:06d}_"
                    "steps40_512x896_ctx08_49f"
                ) / f"{stem}.mp4"
            ]
        return []
    if test_key == "test5":
        return [
            TEST5_RAW_ROOT / PAGE_KEY / f"step-{step:06d}_steps8_512x896_ctx08_49f" / f"{stem}.mp4"
        ]
    if test_key == "physiciq":
        return [
            path / f"{stem}.mp4"
            for path in PHYSICIQ_RAW_ROOT.glob(
                f"xssc_lora_{PAGE_KEY}_step-{step:06d}_steps40*"
            )
        ]
    return []


def _video_exists(test_key: str, weight: dict, stem: str) -> bool:
    display_path = _display_video_path(test_key, weight, stem)
    if _existing_file(display_path):
        return True
    return any(_existing_file(path) for path in _raw_video_candidates(test_key, weight, stem))


def _video_url(test_key: str, weight: dict, stem: str) -> str:
    step = weight["step"]
    page_key = ENABLED_PAGE_KEY if weight["mode"] == "scene_enabled" else PAGE_KEY
    filename = f"{_quote(stem)}.mp4"
    if test_key == "test5":
        return f"../gallery/media/{page_key}/step-{step:06d}/{filename}"
    if test_key == "physiciq":
        return f"../physiciq-gallery/media/{page_key}/step-{step:06d}/{filename}"
    return (
        "../physv-v2v-0819-utonia-no-scene-test70/results/"
        f"{weight['task_id']}/{filename}"
    )


def _weight_status(test_key: str, weight: dict, case_count: int, generated: int) -> str:
    if generated >= case_count:
        return "complete"
    if generated > 0:
        return "running"
    if test_key == "test70":
        state = _task_state(weight["task_id"])
        if state.get("status") == "failed":
            return "failed"
        if state.get("status") == "running":
            return "running"
    return "pending"


def _build_test(test_key: str, label: str, cases: list[dict], weights: list[dict], inference: dict) -> dict:
    attached_cases = _attach_case_urls(cases, test_key)
    columns: list[dict] = []
    generated_cells = 0
    for weight in weights:
        cells: list[dict] = []
        generated = 0
        for case in attached_cases:
            stem = case["stem"]
            exists = _video_exists(test_key, weight, stem)
            if exists:
                generated += 1
                generated_cells += 1
            display_path = _display_video_path(test_key, weight, stem)
            cell = {
                "case_id": case["case_id"],
                "stem": stem,
                "status": "complete" if exists else "pending",
                "video_url": _video_url(test_key, weight, stem) if exists else "",
                "file_path": str(display_path) if exists else "",
            }
            cells.append(cell)
        column = dict(weight)
        column["generated_cases"] = generated
        column["case_count"] = len(attached_cases)
        column["status"] = _weight_status(test_key, weight, len(attached_cases), generated)
        column["cells"] = cells
        columns.append(column)
    return {
        "key": test_key,
        "label": label,
        "case_count": len(attached_cases),
        "expected_cells": len(attached_cases) * len(weights),
        "generated_cells": generated_cells,
        "complete_weights": sum(item["status"] == "complete" for item in columns),
        "weights": columns,
        "cases": attached_cases,
        "inference": inference,
    }


def build_dashboard(write: bool = True) -> dict:
    weights = _discover_weights()
    test5_cases = _load_path_cases(TEST5_LIST, _family_for_test5)
    physiciq_cases = _load_path_cases(PHYSICIQ_LIST, _family_for_physiciq)
    test70_cases = _load_test70_cases()
    tests = [
        _build_test(
            "test5",
            "test_5 · 20-case mixed sanity set",
            test5_cases,
            weights,
            {
                "num_inference_steps": 8,
                "height": 512,
                "width": 896,
                "num_frames": 49,
                "context_frames": 8,
                "seed": 42,
            },
        ),
        _build_test(
            "physiciq",
            "PhysicIQ · 67-case",
            physiciq_cases,
            weights,
            {
                "num_inference_steps": 40,
                "height": 512,
                "width": 896,
                "num_frames": 49,
                "context_frames": 8,
                "seed": 42,
            },
        ),
        _build_test(
            "test70",
            "physV V2V 0819 · all-cycles test70",
            test70_cases,
            [item for item in weights if item["mode"] == "no_scene"],
            {
                "num_inference_steps": 40,
                "height": 512,
                "width": 896,
                "num_frames": 49,
                "context_frames": 8,
                "seed": 42,
            },
        ),
    ]
    payload = {
        "schema_version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "title": "PHYRVG-Full-SA · Utonia Scene Weights · No-Scene / Scene Enabled · 三测试集矩阵",
        "weights_count": len(weights),
        "weights": weights,
        "tests": tests,
    }
    if write:
        PAGE_ROOT.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix="dashboard.", suffix=".json", dir=PAGE_ROOT)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(temp_name, PAGE_ROOT / "dashboard.json")
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
    return payload


# Keep the module importable by the 8844 server even if a caller uses the
# misspelled legacy constant name from an older local script.
TESTICIQ_LIST = PHYSICIQ_LIST


if __name__ == "__main__":
    dashboard = build_dashboard(write=True)
    print(
        json.dumps(
            {
                "updated_at": dashboard["updated_at"],
                "weights": dashboard["weights_count"],
                "tests": [
                    {
                        "key": item["key"],
                        "cases": item["case_count"],
                        "generated_cells": item["generated_cells"],
                        "expected_cells": item["expected_cells"],
                        "complete_weights": item["complete_weights"],
                    }
                    for item in dashboard["tests"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
