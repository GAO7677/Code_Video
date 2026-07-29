#!/usr/bin/env python3
"""Build one incremental dose-control gallery page per source case."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from run_head_role_dose_control_coordinator import _baseline_root
from run_head_role_dose_control_pilot_worker import _input_cases
from summarize_head_role_dose_control import METRICS


DEFAULT_GALLERY_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/gallery/"
    "head-role-dose-control-pilot"
)
DEFAULT_S_DEPTH_CONFIG = Path(__file__).with_name("s_depth_strata_experiment.json")
DEFAULT_S_FEATURE_CONFIG = Path(__file__).with_name(
    "head_role_s_feature_split_pilot.json"
)
DEFAULT_S_FEATURE_UNION_CONFIG = Path(__file__).with_name(
    "head_role_s_feature_union_pilot.json"
)
DEFAULT_S_FEATURE_PHASED_CONFIG = Path(__file__).with_name(
    "head_role_s_feature_phased_pilot.json"
)
MODEL_LABELS = {
    "wan_lora": "Wan+LoRA",
    "xssc": "Wan+xSSC",
    "physrvg": "PhysRVG",
}
METRIC_LABELS = {
    "physics_iq_with_context": "Physics-IQ ctx",
    "physics_iq_without_context": "Physics-IQ noctx",
    "pmf_with_context": "PMF ctx",
    "pmf_without_context": "PMF noctx",
    "wmreward_surprise": "WMReward surprise",
    "vbench_subject_consistency": "VBench subject",
    "vbench_background_consistency": "VBench background",
    "vbench_temporal_flickering": "VBench flicker",
    "vbench_motion_smoothness": "VBench smoothness",
    "vbench_dynamic_degree": "VBench dynamic",
    "vbench_aesthetic_quality": "VBench aesthetic",
    "vbench_imaging_quality": "VBench imaging",
    "videophy2_sa": "VideoPhy2 SA",
    "videophy2_pc": "VideoPhy2 PC",
    "videophy2_joint_rate": "VideoPhy2 joint",
    "videophy2_pc_raw": "VideoPhy2 PC raw",
    "cosmos_reason1": "Cosmos-Reason1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--s-depth-config",
        type=Path,
        default=DEFAULT_S_DEPTH_CONFIG,
    )
    parser.add_argument(
        "--s-feature-config",
        type=Path,
        default=DEFAULT_S_FEATURE_CONFIG,
    )
    parser.add_argument(
        "--s-feature-union-config",
        type=Path,
        default=DEFAULT_S_FEATURE_UNION_CONFIG,
    )
    parser.add_argument(
        "--s-feature-phased-config",
        type=Path,
        default=DEFAULT_S_FEATURE_PHASED_CONFIG,
    )
    parser.add_argument("--gallery-root", type=Path, default=DEFAULT_GALLERY_ROOT)
    return parser.parse_args()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def ensure_link(source: Path, destination: Path) -> None:
    source = source.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source:
            return
        destination.unlink()
    elif destination.exists():
        raise RuntimeError(f"Refusing to replace non-link path: {destination}")
    destination.symlink_to(source)


def finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def nested(payload: dict[str, Any], path: tuple[str, ...]) -> float | None:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return finite(value)


def metrics(payload: dict[str, Any]) -> dict[str, float | None]:
    return {metric.name: nested(payload, metric.path) for metric in METRICS}


def load_sidecar(video: Path) -> dict[str, Any]:
    sidecar = video.with_suffix(".json")
    if not sidecar.is_file():
        return {}
    try:
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def baseline_video_map(root: Path, cases: set[str]) -> dict[str, Path]:
    result = {}
    for path in root.rglob("*.mp4"):
        if path.stem not in cases or path.stat().st_size <= 1024:
            continue
        if path.stem in result:
            raise RuntimeError(f"Duplicate baseline video {path.stem} under {root}")
        result[path.stem] = path.resolve()
    if set(result) != cases:
        raise RuntimeError(f"Baseline {root} has {len(result)}/{len(cases)} cases")
    return result


def source_cases(input_list: Path) -> list[dict[str, Any]]:
    result = []
    for line in input_list.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        path = Path(line.strip()).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        result.append(
            {
                "id": path.stem,
                "input_json": str(path),
                "caption": str(
                    payload.get("input_caption")
                    or payload.get("caption")
                    or payload.get("prompt")
                    or ""
                ),
                "source_video": str(
                    Path(payload["source_video"]).expanduser().resolve()
                ),
                "context_video": str(
                    Path(
                        payload.get("input_video")
                        or payload.get("context_video")
                        or payload.get("input_video_randomf")
                        or ""
                    )
                    .expanduser()
                    .resolve()
                ),
            }
        )
    return result


def media_url(path: Path, base: Path, route: str) -> str:
    return f"/head-role-dose-control-pilot/media/{route}/{path.resolve().relative_to(base.resolve())}"


def build_records(
    config: dict[str, Any],
    root: Path,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    case_ids = {case["id"] for case in cases}
    generation_root = root / "generation"
    baseline_base = Path(config["metrics"]["baseline_root"]).expanduser().resolve()
    records = []
    for state_path in sorted((root / "state").glob("*.json")):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") != "complete":
            continue
        subset_id = str(state["subset_id"])
        role = subset_id[0]
        matching = "exact_block" if "exact" in subset_id else "approx_depth"
        replicate = int(subset_id.split("_r", 1)[1].split("_", 1)[0])
        for case_id, value in sorted(state["videos"].items()):
            video = Path(value).resolve()
            payload = load_sidecar(video)
            records.append(
                {
                    "kind": "ablation",
                    "model": state["model"],
                    "seed": int(state["seed"]),
                    "case_id": case_id,
                    "subset_id": subset_id,
                    "role": role,
                    "k": int(state["k"]),
                    "replicate": replicate,
                    "matching": matching,
                    "start": int(state["step_range"][0]),
                    "end": int(state["step_range"][1]),
                    "video": media_url(video, generation_root, "generation"),
                    "sidecar": media_url(
                        video.with_suffix(".json"), generation_root, "generation"
                    ),
                    "metrics": metrics(payload),
                }
            )
    for seed in config["seeds"]:
        for model in config["models"]:
            baseline_root = _baseline_root(baseline_base, str(model), int(seed))
            for case_id, video in baseline_video_map(baseline_root, case_ids).items():
                payload = load_sidecar(video)
                records.append(
                    {
                        "kind": "baseline",
                        "model": str(model),
                        "seed": int(seed),
                        "case_id": case_id,
                        "subset_id": "baseline",
                        "role": "baseline",
                        "k": 0,
                        "replicate": -1,
                        "matching": "baseline",
                        "start": -1,
                        "end": -1,
                        "video": media_url(video, baseline_base, "baselines"),
                        "sidecar": media_url(
                            video.with_suffix(".json"), baseline_base, "baselines"
                        ),
                        "metrics": metrics(payload),
                    }
                )
    return records


def build_s_depth_records(
    root: Path,
) -> list[dict[str, Any]]:
    generation_root = root / "generation"
    records = []
    for state_path in sorted((root / "state").glob("*.json")):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if state.get("status") != "complete":
            continue
        subset_id = str(state["subset_id"])
        for case_id, value in sorted(state["videos"].items()):
            video = Path(value).resolve()
            payload = load_sidecar(video)
            records.append(
                {
                    "kind": "s_depth",
                    "model": state["model"],
                    "seed": int(state["seed"]),
                    "case_id": case_id,
                    "subset_id": subset_id,
                    "role": "S",
                    "k": int(state["k"]),
                    "replicate": 0,
                    "matching": "s_depth_stratum",
                    "start": int(state["step_range"][0]),
                    "end": int(state["step_range"][1]),
                    "video": media_url(
                        video,
                        generation_root,
                        "s-depth-generation",
                    ),
                    "sidecar": media_url(
                        video.with_suffix(".json"),
                        generation_root,
                        "s-depth-generation",
                    ),
                    "metrics": metrics(payload),
                }
            )
    return records


def build_s_feature_records(
    root: Path,
    subset_payload: dict[str, Any],
    route: str,
) -> list[dict[str, Any]]:
    generation_root = root / "generation"
    records = []
    for state_path in sorted((root / "state").glob("*.json")):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if state.get("status") not in {"running", "complete"}:
            continue
        subset_id = str(state["subset_id"])
        subset = subset_payload["subsets"][subset_id]
        if state.get("status") == "complete":
            videos = {
                case_id: Path(value).resolve()
                for case_id, value in state["videos"].items()
            }
        else:
            start, end = (int(value) for value in state["step_range"])
            variant = f"{subset_id}_steps{start:02d}_{end:02d}"
            job_root = (
                root
                / "generation"
                / str(state["model"])
                / f"seed-{int(state['seed']):06d}"
                / variant
            )
            videos = {
                video.stem: video.resolve()
                for video in job_root.rglob("*.mp4")
                if video.stat().st_size > 1024
                and video.with_suffix(".json").is_file()
                and not video.with_suffix(".json.lock").exists()
            }
        for case_id, video in sorted(videos.items()):
            payload = load_sidecar(video)
            records.append(
                {
                    "kind": "s_feature_split",
                    "model": state["model"],
                    "seed": int(state["seed"]),
                    "case_id": case_id,
                    "subset_id": subset_id,
                    "role": "S",
                    "feature_subtype": subset["feature_subtype"],
                    "k": int(state["k"]),
                    "replicate": 0,
                    "matching": "s_feature_exact_block",
                    "start": int(state["step_range"][0]),
                    "end": int(state["step_range"][1]),
                    "video": media_url(
                        video,
                        generation_root,
                        route,
                    ),
                    "sidecar": media_url(
                        video.with_suffix(".json"),
                        generation_root,
                        route,
                    ),
                    "metrics": metrics(payload),
                }
            )
    return records


def case_page(case_id: str) -> str:
    escaped = html.escape(case_id)
    return PAGE_TEMPLATE.replace("__CASE_ID__", escaped)


def index_page(first_case: str) -> str:
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="0;url={html.escape(first_case)}/">
<title>Head Role Dose-Control Pilot</title></head>
<body><a href="{html.escape(first_case)}/">进入可视化</a></body></html>"""


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.expanduser().resolve().read_text(encoding="utf-8"))
    s_depth_config = json.loads(
        args.s_depth_config.expanduser().resolve().read_text(encoding="utf-8")
    )
    s_feature_config = json.loads(
        args.s_feature_config.expanduser().resolve().read_text(encoding="utf-8")
    )
    s_feature_union_config = json.loads(
        args.s_feature_union_config.expanduser().resolve().read_text(encoding="utf-8")
    )
    s_feature_phased_config = json.loads(
        args.s_feature_phased_config.expanduser().resolve().read_text(encoding="utf-8")
    )
    root = Path(config["storage"]["output_root"]).expanduser().resolve()
    s_depth_root = (
        Path(s_depth_config["storage"]["output_root"]).expanduser().resolve()
    )
    s_feature_root = (
        Path(s_feature_config["storage"]["output_root"]).expanduser().resolve()
    )
    s_feature_union_root = (
        Path(s_feature_union_config["storage"]["output_root"]).expanduser().resolve()
    )
    s_feature_phased_root = (
        Path(s_feature_phased_config["storage"]["output_root"]).expanduser().resolve()
    )
    gallery = args.gallery_root.expanduser().resolve()
    input_list = Path(config["input_list"]).expanduser().resolve()
    cases = source_cases(input_list)
    subset_payload = json.loads(
        Path(config["matched_subset_manifest"])
        .expanduser()
        .resolve()
        .read_text(encoding="utf-8")
    )
    compact_subsets = {
        subset_id: {
            "role": record["role"],
            "k": int(record["k"]),
            "replicate": int(record["replicate"]),
            "matching": record["matching"],
            "block_histogram": record["block_histogram"],
        }
        for subset_id, record in subset_payload["subsets"].items()
    }
    s_depth_subset_payload = json.loads(
        Path(s_depth_config["matched_subset_manifest"])
        .expanduser()
        .resolve()
        .read_text(encoding="utf-8")
    )
    compact_s_depth_subsets = {
        subset_id: {
            "role": record["role"],
            "k": int(record["k"]),
            "matching": record["matching"],
            "depth_stratum": record["depth_stratum"],
            "block_start_inclusive": int(record["block_start_inclusive"]),
            "block_end_exclusive": int(record["block_end_exclusive"]),
            "block_histogram": record["block_histogram"],
        }
        for subset_id, record in s_depth_subset_payload["subsets"].items()
    }
    s_feature_subset_payload = json.loads(
        Path(s_feature_config["matched_subset_manifest"])
        .expanduser()
        .resolve()
        .read_text(encoding="utf-8")
    )
    compact_s_feature_subsets = {
        subset_id: {
            "role": record["role"],
            "feature_subtype": record["feature_subtype"],
            "k": int(record["k"]),
            "matching": record["matching"],
            "block_histogram": record["block_histogram"],
        }
        for subset_id, record in s_feature_subset_payload["subsets"].items()
    }
    s_feature_union_subset_payload = json.loads(
        Path(s_feature_union_config["matched_subset_manifest"])
        .expanduser()
        .resolve()
        .read_text(encoding="utf-8")
    )
    compact_s_feature_union_subsets = {
        subset_id: {
            "role": record["role"],
            "feature_subtype": record["feature_subtype"],
            "k": int(record["k"]),
            "matching": record["matching"],
            "block_histogram": record["block_histogram"],
        }
        for subset_id, record in s_feature_union_subset_payload["subsets"].items()
    }
    compact_s_feature_subsets.update(compact_s_feature_union_subsets)
    s_feature_phased_subset_payload = json.loads(
        Path(s_feature_phased_config["matched_subset_manifest"])
        .expanduser()
        .resolve()
        .read_text(encoding="utf-8")
    )
    ensure_link(root / "generation", gallery / "media" / "generation")
    ensure_link(root / "progress.json", gallery / "live-progress.json")
    ensure_link(
        s_depth_root / "generation",
        gallery / "media" / "s-depth-generation",
    )
    ensure_link(
        s_depth_root / "progress.json",
        gallery / "s-depth-progress.json",
    )
    ensure_link(
        s_feature_root / "generation",
        gallery / "media" / "s-feature-generation",
    )
    ensure_link(
        s_feature_root / "progress.json",
        gallery / "s-feature-progress.json",
    )
    ensure_link(
        s_feature_union_root / "generation",
        gallery / "media" / "s-feature-union-generation",
    )
    ensure_link(
        s_feature_union_root / "progress.json",
        gallery / "s-feature-union-progress.json",
    )
    ensure_link(
        s_feature_phased_root / "generation",
        gallery / "media" / "s-feature-phased-generation",
    )
    ensure_link(
        s_feature_phased_root / "progress.json",
        gallery / "s-feature-phased-progress.json",
    )
    baseline_base = Path(config["metrics"]["baseline_root"]).expanduser().resolve()
    ensure_link(baseline_base, gallery / "media" / "baselines")
    for case in cases:
        source = Path(case["source_video"])
        context = Path(case["context_video"])
        if source.is_file():
            destination = gallery / "media" / "references" / f"{case['id']}__source.mp4"
            ensure_link(source, destination)
            case["source_url"] = f"/head-role-dose-control-pilot/media/references/{destination.name}"
        else:
            case["source_url"] = None
        if context.is_file():
            destination = gallery / "media" / "references" / f"{case['id']}__context.mp4"
            ensure_link(context, destination)
            case["context_url"] = f"/head-role-dose-control-pilot/media/references/{destination.name}"
        else:
            case["context_url"] = None
    records = build_records(config, root, cases)
    s_depth_records = build_s_depth_records(s_depth_root)
    records.extend(s_depth_records)
    s_feature_records = build_s_feature_records(
        s_feature_root,
        s_feature_subset_payload,
        "s-feature-generation",
    )
    s_feature_union_records = build_s_feature_records(
        s_feature_union_root,
        s_feature_union_subset_payload,
        "s-feature-union-generation",
    )
    s_feature_records.extend(s_feature_union_records)
    s_feature_phased_records = build_s_feature_records(
        s_feature_phased_root,
        s_feature_phased_subset_payload,
        "s-feature-phased-generation",
    )
    s_feature_records.extend(s_feature_phased_records)
    records.extend(s_feature_records)
    s_feature_complete_tasks = 0
    for state_path in (s_feature_root / "state").glob("*.json"):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        s_feature_complete_tasks += state.get("status") == "complete"
    for state_path in (s_feature_union_root / "state").glob("*.json"):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        s_feature_complete_tasks += state.get("status") == "complete"
    for state_path in (s_feature_phased_root / "state").glob("*.json"):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        s_feature_complete_tasks += state.get("status") == "complete"
    completed_tasks = len(
        {
            (
                record["model"],
                record["seed"],
                record["subset_id"],
                record["start"],
                record["end"],
            )
            for record in records
            if record["kind"] == "ablation"
        }
    )
    data = {
        "schema_version": 1,
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "generation_tasks_complete": completed_tasks,
        "generation_tasks_expected": 252,
        "s_depth_generation_tasks_complete": len(
            {
                (
                    record["model"],
                    record["seed"],
                    record["subset_id"],
                    record["start"],
                    record["end"],
                )
                for record in s_depth_records
            }
        ),
        "s_depth_generation_tasks_expected": (
            len(s_depth_config["models"])
            * len(s_depth_config["seeds"])
            * len(s_depth_config["step_ranges"])
            * len(compact_s_depth_subsets)
        ),
        "s_depth_step_ranges": s_depth_config["step_ranges"],
        "s_feature_generation_tasks_complete": s_feature_complete_tasks,
        "s_feature_videos_visible": len(s_feature_records),
        "s_feature_generation_tasks_expected": (
            len(s_feature_config["models"])
            * len(s_feature_config["seeds"])
            * len(s_feature_config["step_ranges"])
            * len(s_feature_subset_payload["subsets"])
            + len(s_feature_union_config["models"])
            * len(s_feature_union_config["seeds"])
            * len(s_feature_union_config["step_ranges"])
            * len(s_feature_union_subset_payload["subsets"])
            + len(s_feature_phased_config["models"])
            * len(s_feature_phased_config["seeds"])
            * len(s_feature_phased_config["step_ranges"])
            * len(s_feature_phased_subset_payload["subsets"])
        ),
        "s_feature_step_ranges": sorted(
            {
                tuple(int(value) for value in step_range)
                for source_config in (
                    s_feature_config,
                    s_feature_union_config,
                    s_feature_phased_config,
                )
                for step_range in source_config["step_ranges"]
            },
            key=lambda step_range: (
                step_range == (0, 40),
                step_range[0],
                step_range[1],
            ),
        ),
        "s_feature_subsets": compact_s_feature_subsets,
        "cases": cases,
        "records": records,
        "models": list(MODEL_LABELS),
        "model_labels": MODEL_LABELS,
        "subsets": compact_subsets,
        "s_depth_subsets": compact_s_depth_subsets,
        "metric_definitions": [
            {
                "name": metric.name,
                "label": METRIC_LABELS[metric.name],
                "direction": metric.direction,
                "path": list(metric.path),
            }
            for metric in METRICS
        ],
    }
    atomic_write(
        gallery / "manifest.json",
        json.dumps(data, ensure_ascii=False, separators=(",", ":")),
    )
    for case in cases:
        atomic_write(
            gallery / "cases" / case["id"] / "index.html",
            case_page(case["id"]),
        )
    atomic_write(gallery / "cases" / "index.html", index_page(cases[0]["id"]))
    print(
        f"[dose-case-gallery] cases={len(cases)} records={len(records)} "
        f"complete_tasks={completed_tasks} "
        f"s_depth_tasks={data['s_depth_generation_tasks_complete']}/"
        f"{data['s_depth_generation_tasks_expected']} "
        f"s_feature_tasks={data['s_feature_generation_tasks_complete']}/"
        f"{data['s_feature_generation_tasks_expected']} "
        f"s_feature_videos={data['s_feature_videos_visible']} "
        f"output={gallery / 'cases'}"
    )


PAGE_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__CASE_ID__ · Head Role Dose-Control</title>
<style>
:root{--bg:#101315;--band:#181d20;--line:#343b40;--text:#eef1f2;--muted:#a8b0b5;--accent:#58bda8;--good:#73d3a4;--bad:#ff978e;--pending:#d3b569}
*{box-sizing:border-box}body{margin:0;padding-bottom:58px;background:var(--bg);color:var(--text);font:13px/1.45 system-ui,sans-serif}
header{position:sticky;top:0;z-index:5;padding:11px 16px;background:#101315f2;border-bottom:1px solid var(--line)}
.top,.controls{display:flex;gap:9px;align-items:end;flex-wrap:wrap}.top{justify-content:space-between;align-items:center}
h1,h2,h3,p{margin:0}h1{font-size:18px;overflow-wrap:anywhere}h2{font-size:21px}.hub{color:var(--accent);font-weight:750;text-decoration:none}
label{display:grid;gap:2px;color:var(--muted);font-size:10px}select{min-width:110px;max-width:420px;padding:6px 8px;border:1px solid var(--line);background:#242a2e;color:var(--text)}
#case{min-width:min(520px,80vw)}.controls{margin-top:8px}.status{color:var(--accent);font-weight:700}.prompt{margin-top:7px;color:var(--muted)}
main{padding:14px 16px}.references{display:grid;grid-template-columns:repeat(2,minmax(260px,448px));gap:8px;margin-top:8px}
.model-group{margin-top:34px;border-top:4px solid var(--accent)}.model-group-title{padding:11px 0;border-bottom:1px solid var(--line)}.model-group-title h2{font-size:24px}
.setting-row{margin-top:20px}.setting-row-title{display:flex;gap:12px;align-items:baseline;padding:7px 0}.setting-row-title h3{font-size:16px}.setting-row-title p{color:var(--muted)}.setting-row-title a{color:#efb36d;font-weight:700;text-decoration:none}.stage-row{margin-top:12px}.stage-row-title{display:flex;align-items:center;gap:10px;padding:5px 8px;background:#202629;border-left:3px solid var(--accent);font-size:14px}.stage-row-title h3{font-size:14px}.row-actions{display:flex;gap:5px;margin-left:auto}.row-actions button{padding:3px 8px;font-size:11px}.block-depths{display:flex;flex-wrap:wrap;gap:5px 14px;margin:-2px 0 7px;color:#c4ccd0;font-size:10px}.block-depths b{color:var(--accent)}.videos{display:grid;grid-template-columns:repeat(10,minmax(210px,1fr));gap:7px;overflow-x:auto}
.video-cell{border:1px solid var(--line);background:var(--band)}.video-cell h3{padding:6px 8px;background:#242a2e;font-size:13px}
.video-cell.all-head{border-color:#8f673d}.video-cell.all-head h3{background:#3a2c20;color:#f0bd80}
.video-cell.s-depth{border-color:#8f673d}.video-cell.s-depth h3{background:#47301f;color:#ffd19a}
.video-cell.s-feature{border-color:#4f8f86}.video-cell.s-feature h3{background:#23413d;color:#9de0d1}.videos.feature-split{grid-template-columns:repeat(7,minmax(260px,448px))}
video{display:block;width:100%;aspect-ratio:7/4;object-fit:contain;background:#050606}.meta{padding:5px 8px;color:var(--muted);font-size:10px;min-height:31px}
.missing{display:grid;place-items:center;aspect-ratio:7/4;color:var(--pending);background:#1c2225}.metrics{margin-top:38px}.metric-model{margin-top:30px;border-top:3px solid var(--accent)}.metric-model h3{margin:9px 0;font-size:20px}.metric-setting{margin-top:18px;overflow:auto}.metric-setting h4{margin:0 0 6px;font-size:14px}
.metrics-note{color:var(--muted);margin:5px 0 9px}table{width:100%;border-collapse:collapse;font-size:11px;font-variant-numeric:tabular-nums}
th,td{border:1px solid var(--line);padding:5px 6px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left;position:sticky;left:0;background:#181d20}
thead th{position:sticky;top:0;background:#242a2e}.value{display:block}.delta{display:block;color:var(--muted);font-size:9px}.good .delta{color:var(--good)}.bad .delta{color:var(--bad)}.pending{color:var(--pending)}
.playbar{position:fixed;z-index:8;left:0;right:0;bottom:0;display:flex;gap:8px;align-items:center;padding:9px 16px;background:#171b1ef2;border-top:1px solid var(--line);color:var(--muted)}
button{padding:6px 10px;border:1px solid var(--line);background:#242a2e;color:#fff;cursor:pointer}.time{margin-left:auto;color:var(--muted)}
@media(max-width:620px){.references{grid-template-columns:1fr}}
</style></head><body>
<header><div class="top"><h1 id="title">__CASE_ID__</h1><span class="status" id="status">读取中</span><a class="hub" href="../../metrics/">聚合指标</a><a class="hub" href="/visualizations/">可视化总入口</a></div>
<div class="controls"><label>Case<select id="case"></select></label><label>Seed<select id="seed"></select></label></div>
<p class="prompt" id="prompt"></p></header><main><section><h2>输入与参考</h2><div class="references" id="references"></div></section><div id="settings"></div>
<section class="metrics"><h2>当前指标</h2><p class="metrics-note">指标按与视频相同的匹配方式和replicate分组。每格第一行是原始分数，第二行是相对同模型、同seed baseline的变化；绿色表示改善，红色表示下降。</p><div id="metric-groups"></div></section></main>
<div class="playbar"><span>每行独立同步播放，只加载当前行视频</span><span class="time" id="time"></span></div>
<script>
const CASE_ID="__CASE_ID__",OLD_BASE=`/test5-st-phased-seed851/cases/${encodeURIComponent(CASE_ID)}/`;let D,R,C,OLD=null;
const q=id=>document.getElementById(id),roles=["baseline","S","T","C"];
function options(id,values,label=x=>x){q(id).innerHTML=values.map(x=>`<option value="${x}">${label(x)}</option>`).join("")}
function score(x){return x===null||x===undefined||!Number.isFinite(Number(x))?"Pending":Number(x).toPrecision(4)}
function signed(x){return !Number.isFinite(x)?"Pending":`${x>=0?"+":""}${Number(x).toPrecision(3)}`}
function selection(){return{seed:+q("seed").value}}
function stages(){const seen=new Set();for(const r of R){if(r.kind==="ablation")seen.add(`${r.start}-${r.end}`)}return[...seen].map(x=>x.split("-").map(Number)).sort((a,b)=>a[0]-b[0]||a[1]-b[1])}
function stageLabel(stage){return`${stage[0]}–${stage[1]}`}
function settings(){const map=new Map();for(const r of R){if(r.kind!=="ablation"||r.case_id!==CASE_ID)continue;const key=`${r.matching}:${r.replicate}`;if(!map.has(key))map.set(key,{matching:r.matching,replicate:r.replicate,k:r.k})}return[...map.values()].sort((a,b)=>(a.matching==="approx_depth"?0:1)-(b.matching==="approx_depth"?0:1)||a.replicate-b.replicate)}
function settingLabel(setting){return setting.matching==="approx_depth"?`近似深度匹配 · k=8 · replicate ${String(setting.replicate).padStart(2,"0")}`:`完全同Block匹配 · k=5 · replicate ${String(setting.replicate).padStart(2,"0")}`}
function subset(setting,role){return Object.values(D.subsets).find(x=>x.role===role&&x.replicate===setting.replicate&&((setting.matching==="approx_depth"&&x.matching==="approximate_depth_profile")||(setting.matching==="exact_block"&&x.matching==="exact_same_one_head_per_common_block")))}
function histogramText(histogram){return Object.entries(histogram||{}).sort((a,b)=>Number(a[0])-Number(b[0])).map(([block,count])=>`B${String(block).padStart(2,"0")}${Number(count)>1?`×${count}`:""}`).join(", ")}
function blockDetails(setting){if(setting.matching==="exact_block"){const item=subset(setting,"S");return`<div class="block-depths"><span><b>S/T/C共享 Block：</b>${histogramText(item&&item.block_histogram)}</span></div>`}return`<div class="block-depths">${["S","T","C"].map(role=>{const item=subset(setting,role);return`<span><b>${role} Block：</b>${histogramText(item&&item.block_histogram)}</span>`}).join("")}</div>`}
function selectedRecords(model,setting,stage){const s=selection(),rows=R.filter(r=>r.case_id===CASE_ID&&r.model===model&&r.seed===s.seed);return{baseline:rows.find(r=>r.kind==="baseline"),ablations:rows.filter(r=>r.kind==="ablation"&&r.start===stage[0]&&r.end===stage[1]&&r.matching===setting.matching&&r.replicate===setting.replicate)}}
function depthSubsets(){const order={early:0,middle:1,late:2};return Object.entries(D.s_depth_subsets||{}).map(([id,x])=>({id,...x})).sort((a,b)=>(order[a.depth_stratum]??99)-(order[b.depth_stratum]??99))}
function depthStages(){return(D.s_depth_step_ranges||[]).map(x=>x.map(Number)).sort((a,b)=>a[0]-b[0]||a[1]-b[1])}
function baselineRecord(model){const s=selection();return R.find(r=>r.kind==="baseline"&&r.case_id===CASE_ID&&r.model===model&&r.seed===s.seed)}
function depthRecord(model,subsetId,stage){const s=selection();return R.find(r=>r.kind==="s_depth"&&r.case_id===CASE_ID&&r.model===model&&r.seed===s.seed&&r.subset_id===subsetId&&r.start===stage[0]&&r.end===stage[1])}
function depthLabel(item){const names={early:"Early",middle:"Middle",late:"Late"},end=item.block_end_exclusive-1;return`${names[item.depth_stratum]||item.depth_stratum} · B${String(item.block_start_inclusive).padStart(2,"0")}–${String(end).padStart(2,"0")} · All S (${item.k} heads)`}
function featureSubsets(){const order={local_enrichment:0,same_frame_mass:1,local_same_union:2};return Object.entries(D.s_feature_subsets||{}).map(([id,x])=>({id,...x})).sort((a,b)=>(order[a.feature_subtype]??99)-(order[b.feature_subtype]??99))}
function featureStages(){return(D.s_feature_step_ranges||[]).map(x=>x.map(Number))}
function combinedStages(){const seen=new Map();for(const stage of [...featureStages(),...depthStages()])seen.set(`${stage[0]}-${stage[1]}`,stage);return[...seen.values()].sort((a,b)=>a[0]-b[0]||a[1]-b[1])}
function featureRecord(model,subtype,stage){const s=selection();return R.find(r=>r.kind==="s_feature_split"&&r.case_id===CASE_ID&&r.model===model&&r.seed===s.seed&&r.feature_subtype===subtype&&r.start===stage[0]&&r.end===stage[1])}
function featureLabel(item){const labels={local_enrichment:"Local-enrichment dominant S (32)",same_frame_mass:"Same-frame-mass dominant S (32)",local_same_union:"联合两类 S (64)"};return labels[item.feature_subtype]||item.id}
function media(record,label,missing="该配置仍在生成"){const meta=record&&record.kind==="reference"?"原始输入":record&&record.kind==="baseline"?"同模型、同seed未消融":record?`${record.subset_id} · k=${record.k} · steps ${record.start}-${record.end}`:"Pending";const klass=record&&record.kind==="all_head"?" all-head":record&&record.kind==="s_depth"?" s-depth":record&&record.kind==="s_feature_split"?" s-feature":"";return `<article class="video-cell${klass}"><h3>${label}</h3>${record?`<video muted playsinline preload="none" src="${record.video}"></video><div class="meta">${meta}</div>`:`<div class="missing">${missing}</div><div class="meta">Pending</div>`}</article>`}
function renderReferences(){q("references").innerHTML=media(C.context_url?{video:C.context_url,kind:"reference"}:null,"8帧 Context")+media(C.source_url?{video:C.source_url,kind:"reference"}:null,"Source / GT");q("prompt").textContent=C.caption}
function oldMediaUrl(src){return src?`${OLD_BASE}${src}`:null}
function oldRecord(model,role,range){const s=selection();if(!OLD||s.seed!==851)return null;const key=`${String(range[0]).padStart(2,"0")}_${String(range[1]).padStart(2,"0")}`,stage=OLD.videos.stages[key],scores=OLD.metric_scores.stages[key];if(!stage||!stage[model]||!stage[model][role])return null;const counts={S:OLD.head_distribution.roles.S.total,T:OLD.head_distribution.roles.T.total,ST:OLD.head_distribution.roles.ST.total};return{kind:"all_head",video:oldMediaUrl(stage[model][role]),subset_id:`All-${role}`,role,k:counts[role],start:range[0],end:range[1],metrics:scores&&scores[model]?scores[model][role]:{}}}
function oldLabel(role){return role==="ST"?"旧版 All-S+T":"旧版 All-"+role}
function rowHeader(label){return`<div class="stage-row-title"><h3>${label}</h3><div class="row-actions"><button data-row-action="play">播放本行</button><button data-row-action="replay">从头播放本行</button><button data-row-action="pause">暂停本行</button></div></div>`}
function renderSettings(){stopRowSync();const depth=depthSubsets(),feature=featureSubsets();let out="";for(const model of D.models){out+=`<section class="model-group"><div class="model-group-title"><h2>${D.model_labels[model]} · seed ${selection().seed}</h2></div><div class="setting-row"><div class="setting-row-title"><h3>Head 子类与 S-depth 消融</h3><p>相同去噪阶段固定为同一行</p></div><div class="block-depths"><span><b>Head 子类：</b>Local-enrichment / Same-frame / 联合</span><span><b>S-depth：</b>Early / Middle / Late</span></div>`;for(const stage of combinedStages()){out+=`<section class="stage-row">${rowHeader(`去噪阶段 ${stageLabel(stage)}`)}<div class="videos feature-split">`;out+=media(baselineRecord(model),"Baseline");for(const item of feature)out+=media(featureRecord(model,item.feature_subtype,stage),`Head 子类 · ${featureLabel(item)}`,"该 Head 子类配置仍在生成");for(const item of depth)out+=media(depthRecord(model,item.id,stage),`S-depth · ${depthLabel(item)}`,"该阶段尚无 S-depth 结果");out+="</div></section>"}out+="</div></section>"}q("settings").innerHTML=out}
function metricCell(record,baseline,metric){if(!record)return`<td class="pending">Pending</td>`;const value=record.metrics[metric.name];if(value===null||value===undefined)return`<td class="pending">Pending</td>`;if(record.kind==="baseline")return`<td><span class="value">${score(value)}</span><span class="delta">baseline</span></td>`;const base=baseline&&baseline.metrics[metric.name];if(base===null||base===undefined)return`<td><span class="value">${score(value)}</span><span class="delta">baseline Pending</span></td>`;const raw=value-base,improvement=metric.direction==="higher"?raw:-raw,state=improvement>0?"good":improvement<0?"bad":"";return`<td class="${state}"><span class="value">${score(value)}</span><span class="delta">Δ ${signed(raw)}</span></td>`}
function renderMetrics(){const depth=depthSubsets(),feature=featureSubsets(),head="<tr><th>阶段 / 方法</th>"+D.metric_definitions.map(m=>`<th>${m.label}<br><small>${m.direction}</small></th>`).join("")+"</tr>";let out="";for(const model of D.models){const baseline=baselineRecord(model);let body="";out+=`<section class="metric-model"><h3>${D.model_labels[model]} · seed ${selection().seed}</h3>`;for(const stage of combinedStages()){const prefix=`[${stageLabel(stage)}] `;body+=`<tr><td>${prefix}Baseline</td>`+D.metric_definitions.map(m=>metricCell(baseline,baseline,m)).join("")+"</tr>";for(const item of feature){const r=featureRecord(model,item.feature_subtype,stage);body+=`<tr><td>${prefix}Head 子类 · ${featureLabel(item)}</td>`+D.metric_definitions.map(m=>metricCell(r,baseline,m)).join("")+"</tr>"}for(const item of depth){const r=depthRecord(model,item.id,stage);body+=`<tr><td>${prefix}S-depth · ${depthLabel(item)}</td>`+D.metric_definitions.map(m=>metricCell(r,baseline,m)).join("")+"</tr>"}}out+=`<div class="metric-setting"><h4>按去噪阶段对齐的 Head 子类与 S-depth 消融</h4><table><thead>${head}</thead><tbody>${body}</tbody></table></div></section>`}q("metric-groups").innerHTML=out}
function activeRecords(){const map=new Map();for(const record of R){if(record.case_id!==CASE_ID||record.seed!==selection().seed||!record.sidecar)continue;if(["baseline","s_depth","s_feature_split"].includes(record.kind))map.set(record.sidecar,record)}return[...map.values()]}
function nested(payload,path){let value=payload;for(const key of path){if(!value||typeof value!=="object")return null;value=value[key]}return typeof value==="number"&&Number.isFinite(value)?value:typeof value==="boolean"?Number(value):null}
async function refreshMetrics(){const records=activeRecords();await Promise.all(records.map(async record=>{try{const payload=await fetch(`${record.sidecar}?t=${Date.now()}`).then(x=>x.json());record.metrics=Object.fromEntries(D.metric_definitions.map(metric=>[metric.name,nested(payload,metric.path)]))}catch(error){console.warn("sidecar refresh failed",record.sidecar,error)}}));renderMetrics()}
async function refreshProgress(){const get=path=>fetch(`${path}?t=${Date.now()}`).then(x=>x.ok?x.json():null).catch(()=>null),[dose,depth,feature,union,phased]=await Promise.all([get(`/head-role-dose-control-pilot/live-progress.json`),get(`/head-role-dose-control-pilot/s-depth-progress.json`),get(`/head-role-dose-control-pilot/s-feature-progress.json`),get(`/head-role-dose-control-pilot/s-feature-union-progress.json`),get(`/head-role-dose-control-pilot/s-feature-phased-progress.json`)]),doseText=dose?`${(dose.generation_states||{}).complete||0}/${dose.expected_generation_tasks}`:`${D.generation_tasks_complete}/${D.generation_tasks_expected}`,depthText=depth?`${(depth.state_counts||{}).complete||0}/${depth.expected_tasks}`:`${D.s_depth_generation_tasks_complete}/${D.s_depth_generation_tasks_expected}`,featureText=feature?`${(feature.state_counts||{}).complete||0}/${feature.expected_tasks}`:"6/6",unionText=union?`${(union.state_counts||{}).complete||0}/${union.expected_tasks}`:"0/3",phasedText=phased?`${(phased.state_counts||{}).complete||0}/${phased.expected_tasks}`:"0/18";q("status").textContent=`Dose ${doseText} · S-depth ${depthText} · S-feature ${featureText} + union ${unionText} + phased ${phasedText}`}
function render(){renderSettings();renderMetrics();refreshMetrics();refreshProgress();q("time").textContent=`本页 ${[...document.querySelectorAll("video")].length} 个视频`}
let syncTimer=null,activeRow=null;function stopRowSync(){if(syncTimer!==null)clearInterval(syncTimer);syncTimer=null;activeRow=null}function rowVideos(button){const row=button.closest(".stage-row");return row?[...row.querySelectorAll("video")]:[]}function playRow(button,replay=false){stopRowSync();const videos=rowVideos(button);if(!videos.length)return;if(replay)for(const video of videos)video.currentTime=0;for(const video of videos){video.playbackRate=1;video.play().catch(()=>{})}activeRow=button.closest(".stage-row");syncTimer=setInterval(()=>{const current=activeRow?[...activeRow.querySelectorAll("video")]:[];if(current.length<2)return;const leader=current[0];if(leader.readyState<2)return;for(const video of current.slice(1)){if(video.readyState>=2&&Math.abs(video.currentTime-leader.currentTime)>0.12)video.currentTime=leader.currentTime}},250)}function pauseRow(button){const row=button.closest(".stage-row");if(row===activeRow)stopRowSync();for(const video of rowVideos(button))video.pause()}document.addEventListener("click",event=>{const button=event.target.closest("[data-row-action]");if(!button)return;const action=button.dataset.rowAction;if(action==="play")playRow(button,false);else if(action==="replay")playRow(button,true);else pauseRow(button)});
fetch("../../manifest.json").then(x=>x.json()).then(data=>{D=data;R=data.records;C=data.cases.find(x=>x.id===CASE_ID);options("case",D.cases.map(x=>x.id));q("case").value=CASE_ID;q("case").onchange=()=>location.href=`../${q("case").value}/`;options("seed",[...new Set(R.map(x=>x.seed))].sort((a,b)=>a-b));q("seed").onchange=render;renderReferences();render();setInterval(refreshProgress,30000)}).catch(error=>{q("status").textContent=`加载失败: ${error}`});
</script></body></html>"""


if __name__ == "__main__":
    main()
