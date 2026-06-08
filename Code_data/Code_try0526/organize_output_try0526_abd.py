#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any


ROOT = Path("/data/gaoya/AAA_test_video/Output_try0526")
ABD_ROOT = ROOT / "ABD_test"
MANIFEST_ROOT = ROOT / "bench_jsons_mer"

A_OUTPUT_ROOT = ROOT / "PDI-Bench" / "output"
A_REPORT_ROOT = ROOT / "PDI-Bench" / "report"
A_REPORT_SUBSET_ROOT = ROOT / "PDI-Bench" / "report_subset"
A_RESULT_ROOT = ROOT / "PDI-Bench" / "result"

D_OUTPUT_ROOT = ROOT / "physics-iq-benchmark" / "output"
D_REPORT_ROOT = ROOT / "physics-iq-benchmark" / "report_progress"
D_REPORT_SUBSET_ROOT = ROOT / "physics-iq-benchmark" / "report_subset"
D_RESULT_ROOT = ROOT / "physics-iq-benchmark" / "result"

B_SOURCE_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0526dp/videos")
B_REPORT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0526dp/reports")


def _slugify(text: str) -> str:
    normalized = re.sub(r"\s+", "_", text.strip())
    normalized = re.sub(r"[^0-9A-Za-z_\-]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "case"


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _replace_with_symlink(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"missing source for symlink: {src}")
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(src, dst)


def _replace_with_copy(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"missing source for copy: {src}")
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.is_dir():
        shutil.rmtree(dst)
    shutil.copy2(src, dst)


def _replace_tree_with_copy(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"missing source tree for copy: {src}")
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.is_dir():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _link_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    if src.is_dir():
        _replace_tree_with_copy(src, dst)
    else:
        _replace_with_copy(src, dst)
    return True


def _copy_file_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    _replace_with_copy(src, dst)
    return True


def _extract_metric_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    keys = [
        "pdi_score",
        "wmreward_jepa",
        "vjepa_proxy",
        "cosmos_reason1_score",
        "videophy2_auto_pc",
        "videophy2_auto_sa",
    ]
    for key in keys:
        if key in payload:
            summary[key] = payload[key]
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        for key in ["grade", "scale_component", "traj_component", "epsilon_rigidity", "vp_component"]:
            if key in metrics:
                summary[key] = metrics[key]
    pdi = payload.get("pdi")
    if isinstance(pdi, dict) and "grade" in pdi and "grade" not in summary:
        summary["grade"] = pdi["grade"]
    return summary


def _case_key(index: int, category: str, name: str) -> str:
    return f"{index:03d}_{_slugify(category)}_{_slugify(name)}"


def _rel_symlink_summary(src: Path, dst: Path) -> dict[str, str]:
    return {
        "organized_path": str(dst),
        "original_path": str(src),
    }


def organize_group_a(summary: dict[str, Any]) -> None:
    group_dir = _ensure_dir(ABD_ROOT / "A")
    manifest_dir = _ensure_dir(group_dir / "_meta")
    _link_if_exists(MANIFEST_ROOT / "A.json", manifest_dir / "source_manifest.json")
    _link_if_exists(A_RESULT_ROOT / "metrics.csv", manifest_dir / "metrics.csv")
    _link_if_exists(A_RESULT_ROOT / "official_metrics.csv", manifest_dir / "official_metrics.csv")
    _link_if_exists(A_REPORT_ROOT / "index.html", manifest_dir / "report.html")
    _link_if_exists(A_REPORT_SUBSET_ROOT / "index.html", manifest_dir / "report_subset.html")
    _link_if_exists(A_REPORT_SUBSET_ROOT / "selected_cases.json", manifest_dir / "report_subset_selected_cases.json")

    methods = {
        "GT": A_OUTPUT_ROOT / "GT",
        "wan22-5B-TI2V": A_OUTPUT_ROOT / "wan22-5B-TI2V",
        "VACE_1p3B_TI2V": A_OUTPUT_ROOT / "VACE_1p3B_TI2V",
        "VACE_1p3B_ctx08": A_OUTPUT_ROOT / "VACE_1p3B_ctx08",
    }
    rows = _load_json(MANIFEST_ROOT / "A.json")
    group_summary: dict[str, Any] = {"count": len(rows), "methods": {}}

    for method in methods:
        _ensure_dir(group_dir / method)
        group_summary["methods"][method] = {"present": 0, "missing": 0, "missing_cases": []}

    for index, row in enumerate(rows):
        category = str(row["category"])
        clip_name = Path(str(row["source_video"])).stem
        case_key = _case_key(index, category, clip_name)
        for method, method_root in methods.items():
            original_video = method_root / category / f"{clip_name}.mp4"
            original_json = method_root / category / f"{clip_name}.json"
            out_video = group_dir / method / f"{case_key}.mp4"
            out_json = group_dir / method / f"{case_key}.json"
            if not original_video.exists():
                group_summary["methods"][method]["missing"] += 1
                group_summary["methods"][method]["missing_cases"].append(case_key)
                continue
            _replace_with_copy(original_video, out_video)
            original_payload = _load_json(original_json) if original_json.exists() else {}
            normalized = {
                "group": "A",
                "benchmark": "PDI-Bench",
                "method_name": method,
                "case_key": case_key,
                "category": category,
                "clip_name": clip_name,
                "input_prompt": row.get("caption"),
                "input_image": row.get("first_frame"),
                "source_video": row.get("source_video"),
                "organized_video": str(out_video),
                "output_video": str(original_video),
                "original_json": str(original_json) if original_json.exists() else None,
                "metric_summary": _extract_metric_summary(original_payload) if isinstance(original_payload, dict) else {},
            }
            _write_json(out_json, normalized)
            group_summary["methods"][method]["present"] += 1

    summary["A"] = group_summary


def organize_group_b(summary: dict[str, Any]) -> None:
    group_dir = _ensure_dir(ABD_ROOT / "B")
    manifest_dir = _ensure_dir(group_dir / "_meta")
    _copy_file_if_exists(MANIFEST_ROOT / "B.json", manifest_dir / "source_manifest.json")
    report_index_map: dict[str, str] = {}
    if B_REPORT_ROOT.exists():
        for report_dir in sorted(B_REPORT_ROOT.iterdir()):
            if report_dir.is_dir():
                index_html = report_dir / "index.html"
                copied = _copy_file_if_exists(index_html, manifest_dir / f"{report_dir.name}.index.html")
                report_index_map[report_dir.name] = str(index_html) if copied else str(report_dir)
    _write_json(manifest_dir / "report_index_map.json", report_index_map)

    method_name = "GT"
    _ensure_dir(group_dir / method_name)
    rows = _load_json(MANIFEST_ROOT / "B.json")
    group_summary: dict[str, Any] = {
        "count": len(rows),
        "methods": {
            method_name: {"present": 0, "missing": 0, "missing_cases": []}
        },
    }

    for index, row in enumerate(rows):
        category = str(row["category"])
        source_video = Path(str(row["source_video"]))
        clip_name = source_video.stem
        case_key = _case_key(index, category, clip_name)
        original_json = source_video.with_suffix(".json")
        out_video = group_dir / method_name / f"{case_key}.mp4"
        out_json = group_dir / method_name / f"{case_key}.json"
        if not source_video.exists():
            group_summary["methods"][method_name]["missing"] += 1
            group_summary["methods"][method_name]["missing_cases"].append(case_key)
            continue
        # B 组主要作为可直接浏览的 source video 视图，使用真实文件拷贝，
        # 避免本地预览器对 mp4 符号链接兼容性不稳定。
        _replace_with_copy(source_video, out_video)
        original_payload = _load_json(original_json) if original_json.exists() else {}
        normalized = {
            "group": "B",
            "benchmark": "Dataset_physV",
            "method_name": method_name,
            "case_key": case_key,
            "category": category,
            "clip_name": clip_name,
            "input_prompt": row.get("caption"),
            "source_video": row.get("source_video"),
            "organized_video": str(out_video),
            "output_video": str(source_video),
            "original_json": str(original_json) if original_json.exists() else None,
            "metric_summary": _extract_metric_summary(original_payload) if isinstance(original_payload, dict) else {},
        }
        _write_json(out_json, normalized)
        group_summary["methods"][method_name]["present"] += 1

    summary["B"] = group_summary


def organize_group_d(summary: dict[str, Any]) -> None:
    group_dir = _ensure_dir(ABD_ROOT / "D")
    manifest_dir = _ensure_dir(group_dir / "_meta")
    _link_if_exists(MANIFEST_ROOT / "D.json", manifest_dir / "source_manifest.json")
    _link_if_exists(D_RESULT_ROOT / "method_metrics_summary.csv", manifest_dir / "method_metrics_summary.csv")
    _link_if_exists(D_REPORT_ROOT / "index.html", manifest_dir / "report_progress.html")
    _link_if_exists(D_REPORT_SUBSET_ROOT / "index.html", manifest_dir / "report_subset.html")
    _link_if_exists(D_REPORT_SUBSET_ROOT / "selected_cases.json", manifest_dir / "report_subset_selected_cases.json")

    methods = {
        "GT": D_OUTPUT_ROOT / "GT" / "physics-iq-benchmark",
        "wan22-5B-TI2V": D_OUTPUT_ROOT / "wan22-5B-TI2V" / "physics-iq-benchmark",
        "VACE_1p3B_TI2V": D_OUTPUT_ROOT / "VACE_1p3B_TI2V" / "physics-iq-benchmark",
        "VACE_1p3B_ctx08": D_OUTPUT_ROOT / "VACE_1p3B_ctx08" / "physics-iq-benchmark",
    }
    rows = _load_json(MANIFEST_ROOT / "D.json")
    group_summary: dict[str, Any] = {"count": len(rows), "methods": {}}

    for method in methods:
        _ensure_dir(group_dir / method)
        group_summary["methods"][method] = {"present": 0, "missing": 0, "missing_cases": []}

    for index, row in enumerate(rows):
        category = str(row["category"])
        sample_name = Path(str(row["first_frame"])).parent.name if row.get("first_frame") else Path(str(row["source_video"])).stem
        case_key = _case_key(index, category, sample_name)
        for method, method_root in methods.items():
            original_video = method_root / f"{sample_name}.mp4"
            original_json = method_root / f"{sample_name}.json"
            out_video = group_dir / method / f"{case_key}.mp4"
            out_json = group_dir / method / f"{case_key}.json"
            if not original_video.exists():
                group_summary["methods"][method]["missing"] += 1
                group_summary["methods"][method]["missing_cases"].append(case_key)
                continue
            _replace_with_copy(original_video, out_video)
            original_payload = _load_json(original_json) if original_json.exists() else {}
            normalized = {
                "group": "D",
                "benchmark": "physics-iq-benchmark",
                "method_name": method,
                "case_key": case_key,
                "category": category,
                "sample_name": sample_name,
                "input_prompt": row.get("caption"),
                "input_image": row.get("first_frame"),
                "input_context_video": row.get("context_video"),
                "source_video": row.get("source_video"),
                "organized_video": str(out_video),
                "output_video": str(original_video),
                "original_json": str(original_json) if original_json.exists() else None,
                "metric_summary": _extract_metric_summary(original_payload) if isinstance(original_payload, dict) else {},
            }
            _write_json(out_json, normalized)
            group_summary["methods"][method]["present"] += 1

    summary["D"] = group_summary


def write_readmes(summary: dict[str, Any]) -> None:
    root_md = """# Output_try0526 Directory Notes

This root now contains two layers:

- `ABD_test/`: the clean, consumer-facing unified view for groups `A/B/D`
- original benchmark directories: preserved in place so historical scripts keep working

Directory usefulness:

- `bench_jsons_mer/`: useful. Compact manifests for `A/B/D` and the source of truth for ABD grouping.
- `benchmark_result/`: currently empty and not useful.
- `runs/`: useful as raw evaluation artifacts archive. Some manifests and old reports point back here.
- `tmp/`: low-value scratch/cache directory. Not needed for reading ABD results; can be removed later if you do not need old temp artifacts.

Benchmark internals:

- `PDI-Bench/`
  - `output/`: method videos and per-case jsons
  - `report/`, `report_subset/`: html summaries
  - `result/`: aggregated csv metrics
  - `manifest_*.json`: export bookkeeping
- `physics-iq-benchmark/`
  - `output/`: method videos and per-case jsons
  - `report_progress/`, `report_subset/`: html summaries
  - `result/`: aggregated csv metrics
  - `manifest.json`: export bookkeeping
- `phygenbench/`
  - `output/`: generated first frames, context videos, method videos/jsons
  - `logs/`: generation/eval logs
  - `result/`: aggregated csv metrics
  - `manifest.json`: export bookkeeping

Recommendation:

- consume `ABD_test/` for A/B/D inspection
- keep original benchmark directories as archival/raw provenance
- treat `benchmark_result/` as removable
- treat `tmp/` as removable after manual confirmation
"""
    _write_text(ROOT / "README_ORGANIZATION.md", root_md)

    abd_md = f"""# ABD_test

This directory is a unified, non-destructive view over historical `Output_try0526` results.

Structure:

- `A/`: `PDI-Bench` results
- `B/`: `Dataset_physV` source videos
- `D/`: `physics-iq-benchmark` results
- each method directory is flat: `000_category_case.mp4/.json`
- `*_meta/` contains source manifests, reports, and aggregate csvs

Important:

- all videos and copied metadata files here are real files, not symlinks
- jsons here are normalized lightweight metadata files
- `B/_meta` keeps lightweight report entry files only; large historical report assets stay in the original directories
- original benchmark directories remain unchanged

Coverage summary:

```json
{json.dumps(summary, ensure_ascii=False, indent=2)}
```
"""
    _write_text(ABD_ROOT / "README.md", abd_md)


def link_report_dir_contents(src_dir: Path, dst_dir: Path) -> None:
    if not src_dir.exists():
        return
    _ensure_dir(dst_dir)
    for child in src_dir.iterdir():
        if child.is_file():
            _link_if_exists(child, dst_dir / child.name)


def main() -> None:
    _ensure_dir(ABD_ROOT)
    summary: dict[str, Any] = {}

    organize_group_a(summary)
    organize_group_b(summary)
    organize_group_d(summary)

    # lightweight top-level meta links
    meta_dir = _ensure_dir(ABD_ROOT / "_meta")
    _link_if_exists(MANIFEST_ROOT / "A.json", meta_dir / "A.json")
    _link_if_exists(MANIFEST_ROOT / "B.json", meta_dir / "B.json")
    _link_if_exists(MANIFEST_ROOT / "D.json", meta_dir / "D.json")
    link_report_dir_contents(A_RESULT_ROOT, meta_dir / "A_result")
    link_report_dir_contents(D_RESULT_ROOT, meta_dir / "D_result")

    _write_json(ABD_ROOT / "coverage_summary.json", summary)
    write_readmes(summary)
    print(f"organized ABD view written to {ABD_ROOT}")


if __name__ == "__main__":
    main()
