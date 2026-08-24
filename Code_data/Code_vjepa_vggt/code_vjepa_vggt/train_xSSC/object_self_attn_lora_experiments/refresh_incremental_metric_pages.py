#!/usr/bin/env python3
"""Refresh aggregate metric pages from the already-built case-page records.

Metric workers commit one marker at a time.  The old dashboard builder only
ran after a complete metric set, so the average pages could remain stale even
though the case result JSONs and marker files were already updated.  This
lightweight updater reuses the current case-page payload, reorders methods,
and rewrites only the aggregate pages.  It never runs a metric or inference
job.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any

from build_xssc_lora_checkpoint_dashboard import (
    build_average_metrics_page,
    display_methods,
)


HUB_ROOT = Path("/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_hub")
WATCH_ROOT = Path(
    "/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_three_run_watch"
)
SCENE_ENABLED_METHOD = "full_sa_physrvg_vjepa_utonia_scene_hardmask_v1_enabled"
DATA_RE = re.compile(r"const D=(\{.*?\});\n    const caseSelect", re.DOTALL)

# These columns are emitted by one benchmark invocation.  Keep them together
# when the corresponding marker is present, otherwise an old result JSON can
# make derived columns look complete before the current metric is rerun.
MARKER_COLUMNS = {
    "videophy2": {
        "videophy2_pc_raw",
        "videophy2",
        "videophy2_sa",
        "videophy2_pc",
        "videophy2_joint_rate",
    },
}

PAGE_SPECS = {
    "test5": {
        "average_title": "test_5 · 全 case 平均指标",
        "gt_dataset_key": "test5",
    },
    "physiciq": {
        "average_title": "PhysicIQ · 67-case 平均指标",
        "gt_dataset_key": "physiciq",
    },
}


def load_page_data(page_path: Path) -> tuple[str, dict[str, Any], re.Match[str]]:
    text = page_path.read_text(encoding="utf-8")
    match = DATA_RE.search(text)
    if match is None:
        raise RuntimeError(f"embedded dashboard data not found: {page_path}")
    payload = json.loads(match.group(1))
    if not isinstance(payload, dict):
        raise TypeError(f"dashboard payload is not an object: {page_path}")
    return text, payload, match


def reorder_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    methods = payload.get("methods", [])
    if not isinstance(methods, list):
        raise TypeError("dashboard methods must be a list")
    ordered_methods = display_methods(
        [method for method in methods if isinstance(method, dict)]
    )
    order = {
        str(method.get("key", "")): index
        for index, method in enumerate(ordered_methods)
    }
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise TypeError("dashboard records must be a list")
    payload["methods"] = ordered_methods
    payload["records"] = sorted(
        [record for record in records if isinstance(record, dict)],
        key=lambda record: (
            order.get(str(record.get("method_key", "")), 999),
            int(record.get("step", 0)),
            str(record.get("step_kind", "training")),
            str(record.get("origin", "")),
        ),
    )
    return ordered_methods


def gate_scene_enabled_metrics(
    payload: dict[str, Any],
    page_name: str,
) -> None:
    """Expose only metrics committed by the current Scene-Enabled run."""

    if page_name == "test5":
        marker_root = WATCH_ROOT / "state" / "metrics"
    else:
        marker_root = WATCH_ROOT / "state" / "physiciq" / "metrics"
    records = payload.get("records", [])
    if not isinstance(records, list):
        return
    for record in records:
        if record.get("method_key") != SCENE_ENABLED_METHOD:
            continue
        step = int(record.get("step", 0))
        marker_dir = marker_root / SCENE_ENABLED_METHOD / f"step-{step:06d}"
        markers = {
            marker.stem for marker in marker_dir.glob("*.json")
        } if marker_dir.is_dir() else set()
        metrics = record.get("metrics", {})
        if not isinstance(metrics, dict):
            record["metrics"] = {}
            continue
        gated: dict[str, dict[str, Any]] = {}
        for stem, values in metrics.items():
            if not isinstance(values, dict):
                continue
            filtered: dict[str, Any] = {}
            for metric, value in values.items():
                marker_name = next(
                    (
                        source
                        for source, columns in MARKER_COLUMNS.items()
                        if metric in columns
                    ),
                    metric,
                )
                if marker_name in markers:
                    filtered[metric] = value
            if filtered:
                gated[str(stem)] = filtered
        record["metrics"] = gated


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_page_payload(page_path: Path, text: str, payload: dict[str, Any], match: re.Match[str]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    replacement = "const D=" + encoded + ";\n    const caseSelect"
    atomic_write(page_path, text[: match.start()] + replacement + text[match.end() :])


def metric_cell_counts(records: list[dict[str, Any]], cases: list[dict[str, Any]]) -> tuple[int, int]:
    total = 0
    complete = 0
    stems = [str(case.get("stem", "")) for case in cases]
    for record in records:
        metrics = record.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        for stem in stems:
            values = metrics.get(stem, {})
            if not isinstance(values, dict):
                continue
            total += len(values)
            complete += sum(
                1
                for value in values.values()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            )
    return complete, total


def refresh_one(page_name: str, spec: dict[str, str]) -> dict[str, Any]:
    page_path = HUB_ROOT / page_name / "index.html"
    text, payload, match = load_page_data(page_path)
    gate_scene_enabled_metrics(payload, page_name)
    methods = reorder_payload(payload)
    cases = payload.get("cases", [])
    records = payload.get("records", [])
    if not isinstance(cases, list) or not isinstance(records, list):
        raise TypeError(f"invalid case/record payload: {page_path}")
    write_page_payload(page_path, text, payload, match)

    average_html = build_average_metrics_page(
        records,
        cases,
        page_title=spec["average_title"],
        methods_override=methods,
        gt_dataset_key=spec["gt_dataset_key"],
    )
    average_path = HUB_ROOT / f"{page_name}-average-metrics" / "index.html"
    average_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(average_path, average_html)

    extra_pages: list[str] = []
    if page_name == "physiciq":
        solid_cases = [
            case
            for case in cases
            if "_Solid_Mechanics_" in json.dumps(case, ensure_ascii=False)
        ]
        solid_html = build_average_metrics_page(
            records,
            solid_cases,
            page_title=f"PhysicIQ · Solid Mechanics · {len(solid_cases)}-case 平均指标",
            methods_override=methods,
            gt_dataset_key="physiciq",
        )
        solid_path = HUB_ROOT / "physiciq-average-metrics" / "solid-mechanics" / "index.html"
        solid_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(solid_path, solid_html)
        extra_pages.append(str(solid_path))

    complete, total = metric_cell_counts(records, cases)
    return {
        "case_page": str(page_path),
        "average_page": str(average_path),
        "extra_pages": extra_pages,
        "methods": len(methods),
        "records": len(records),
        "metric_values": f"{complete}/{total}",
    }


def main() -> None:
    results = [refresh_one(name, spec) for name, spec in PAGE_SPECS.items()]
    print(json.dumps({"refreshed": results}, ensure_ascii=False))


if __name__ == "__main__":
    main()
