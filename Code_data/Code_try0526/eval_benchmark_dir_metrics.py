#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from physv_eval.cosmos_reason1_official import OfficialCosmosReason1Runner
from physv_eval.official_pdi import OfficialPDIRunner
from physv_eval.proxy_runner import ProxyRunner
from physv_eval.records import (
    get_cosmos_reason1,
    get_official_pdi,
    get_proxy,
    get_videophy2_auto,
    get_wmreward,
    load_payload,
    metric_value,
    save_payload,
    set_cosmos_reason1,
    set_official_pdi,
    set_proxy,
    set_videophy2_auto,
    set_wmreward,
)
from physv_eval.videophy2_auto import VideoPhy2Runner, resolve_videophy2_sa_query
from physv_eval.wmreward_official import WMRewardRunner


DEFAULT_METRICS = ["pdi", "wmreward", "proxy", "videophy2", "cosmos"]
SUMMARY_METRICS = [
    "official_pdi",
    "scale_component",
    "traj_component",
    "epsilon_rigidity",
    "vp_component",
    "wmreward_surprise",
    "cosmos_reason1",
    "vjepa_temporal_relation_raw_error",
    "vjepa_delta_relation_raw_error",
    "vjepa_delta_profile_error",
    "videophy2_auto_sa",
    "videophy2_auto_pc",
    "videophy2_auto_joint",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recursively evaluate a benchmark output directory with reusable metrics.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--metrics", nargs="+", choices=DEFAULT_METRICS, default=DEFAULT_METRICS)
    parser.add_argument("--refresh-pdi", action="store_true")
    parser.add_argument("--refresh-wmreward", action="store_true")
    parser.add_argument("--refresh-proxy", action="store_true")
    parser.add_argument("--refresh-videophy2", action="store_true")
    parser.add_argument("--refresh-cosmos", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--proxy-device", default=None)
    parser.add_argument("--pdi-python", default=None)
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--wmreward-cuda-visible-devices", default=None)
    parser.add_argument("--videophy-device", default="cuda")
    parser.add_argument("--videophy-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--summary-csv", type=Path, default=None)
    return parser.parse_args()


def iter_json_paths(input_root: Path) -> list[Path]:
    search_root = input_root / "output" if (input_root / "output").is_dir() else input_root
    return sorted(path for path in search_root.rglob("*.json") if path.is_file())


def method_name_for(json_path: Path, payload: dict[str, Any], base_root: Path) -> str:
    method = payload.get("method")
    if isinstance(method, str) and method.strip():
        return method.strip()
    try:
        rel = json_path.relative_to(base_root)
    except ValueError:
        return json_path.parent.name
    parts = rel.parts
    if len(parts) >= 2:
        return parts[0]
    return json_path.parent.name


def resolve_video_path(json_path: Path, payload: dict[str, Any]) -> Path:
    candidates = [
        payload.get("video"),
        payload.get("video_path"),
        (payload.get("paths") or {}).get("output_video_path"),
        payload.get("output_video"),
        json_path.with_suffix(".mp4"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_file():
            return path
    raise FileNotFoundError(f"No video found for {json_path}")


def resolve_context_video_path(payload: dict[str, Any], fallback: Path) -> Path:
    candidates = [
        payload.get("context_video"),
        (payload.get("paths") or {}).get("context_video_path"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_file():
            return path
    return fallback


def resolve_text_query(payload: dict[str, Any]) -> str:
    for key in ["prompt", "caption", "text_prompt", "description", "target_object", "scenario", "experiment"]:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "ball"


def should_run_pdi(payload: dict[str, Any], refresh: bool) -> bool:
    return refresh or get_official_pdi(payload) is None or metric_value(payload, "official_pdi") is None


def should_run_wmreward(payload: dict[str, Any], refresh: bool) -> bool:
    return refresh or get_wmreward(payload) is None or metric_value(payload, "wmreward_jepa") is None


def should_run_proxy(payload: dict[str, Any], refresh: bool) -> bool:
    return refresh or get_proxy(payload) is None or metric_value(payload, "vjepa_proxy") is None


def should_run_videophy2_pc(payload: dict[str, Any], refresh: bool) -> bool:
    return refresh or get_videophy2_auto(payload) is None or metric_value(payload, "videophy2_auto_pc") is None


def should_run_videophy2_sa(payload: dict[str, Any], refresh: bool) -> bool:
    bucket = get_videophy2_auto(payload)
    return refresh or not isinstance(bucket, dict) or bucket.get("sa_score") is None


def should_run_cosmos(payload: dict[str, Any], refresh: bool) -> bool:
    return refresh or get_cosmos_reason1(payload) is None or metric_value(payload, "cosmos_reason1") is None


def mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def write_summary_csv(base_root: Path, rows: list[dict[str, Any]], output_path: Path) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row["payload"])

    summary_rows: list[dict[str, Any]] = []
    for method in sorted(grouped):
        payloads = grouped[method]
        summary: dict[str, Any] = {
            "benchmark_root": str(base_root),
            "method": method,
            "num_videos": len(payloads),
        }
        for metric_name in SUMMARY_METRICS:
            values = [metric_value(payload, metric_name) for payload in payloads]
            clean = [float(value) for value in values if value is not None]
            summary[metric_name] = "" if not clean else f"{mean_or_none(clean):.6f}"
        summary_rows.append(summary)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    args = parse_args()
    json_paths = iter_json_paths(args.input_root)
    search_root = args.input_root / "output" if (args.input_root / "output").is_dir() else args.input_root

    enabled_metrics = set(args.metrics)
    pdi_runner = OfficialPDIRunner(
        python_bin=args.pdi_python,
        cuda_visible_devices=args.cuda_visible_devices,
    ) if "pdi" in enabled_metrics else None
    wmreward_runner = WMRewardRunner(
        cuda_visible_devices=args.wmreward_cuda_visible_devices or args.cuda_visible_devices,
    ) if "wmreward" in enabled_metrics else None
    proxy_runner = ProxyRunner(device=args.proxy_device or args.device) if "proxy" in enabled_metrics else None
    videophy_runner = VideoPhy2Runner(
        device=args.videophy_device,
        dtype=args.videophy_dtype,
    ) if "videophy2" in enabled_metrics else None
    cosmos_runner = OfficialCosmosReason1Runner() if "cosmos" in enabled_metrics else None

    processed_rows: list[dict[str, Any]] = []
    selected_methods = set(args.methods or [])
    selected_jsons: list[Path] = []
    for json_path in json_paths:
        payload = load_payload(json_path)
        method = method_name_for(json_path, payload, search_root)
        if selected_methods and method not in selected_methods:
            continue
        selected_jsons.append(json_path)

    for index, json_path in enumerate(selected_jsons, start=1):
        payload = load_payload(json_path)
        method = method_name_for(json_path, payload, search_root)
        video_path = resolve_video_path(json_path, payload)
        context_video_path = resolve_context_video_path(payload, video_path)
        changed = False
        print(f"[{index}/{len(selected_jsons)}] {method} :: {json_path.name}", flush=True)

        if pdi_runner is not None and should_run_pdi(payload, args.refresh_pdi):
            result = pdi_runner.run(video_path, resolve_text_query(payload), refresh=args.refresh_pdi)
            set_official_pdi(payload, result)
            changed = True

        if wmreward_runner is not None and should_run_wmreward(payload, args.refresh_wmreward):
            result = wmreward_runner.score(video_path)
            set_wmreward(payload, result)
            changed = True

        if proxy_runner is not None and should_run_proxy(payload, args.refresh_proxy):
            result = proxy_runner.score(video_path, context_video_path=context_video_path)
            if result is not None:
                set_proxy(payload, result)
                changed = True

        if videophy_runner is not None:
            if should_run_videophy2_pc(payload, args.refresh_videophy2):
                result = videophy_runner.score_video(video_path, task="pc")
                set_videophy2_auto(payload, result)
                changed = True
            if should_run_videophy2_sa(payload, args.refresh_videophy2):
                caption = resolve_videophy2_sa_query(video_path, payload)
                result = videophy_runner.score_video(video_path, task="sa", caption=caption)
                set_videophy2_auto(payload, result)
                changed = True

        if cosmos_runner is not None and should_run_cosmos(payload, args.refresh_cosmos):
            result = cosmos_runner.score(video_path)
            set_cosmos_reason1(payload, result)
            changed = True

        if changed:
            save_payload(json_path, payload)
        processed_rows.append({"method": method, "payload": payload})

    if not processed_rows:
        print("No matching JSON files found.", flush=True)
        return

    summary_csv = args.summary_csv or (args.input_root / "result" / "method_metrics_summary.csv")
    write_summary_csv(args.input_root, processed_rows, summary_csv)
    print(f"summary_csv={summary_csv}", flush=True)


if __name__ == "__main__":
    main()
