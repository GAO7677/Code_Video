from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .datasets import GROUP_SPECS, iter_group_jsons
from .official_pdi import OfficialPDIRunner, resolve_text_query
from .paths import REPO_ROOT, VPHY_PYTHON
from .proxy_runner import ProxyRunner
from .records import (
    get_official_pdi,
    get_proxy,
    get_videophy2_auto,
    get_wmreward,
    load_payload,
    metric_value,
    resolve_video_path,
    save_payload,
    set_official_pdi,
    set_proxy,
    set_wmreward,
)
from .wmreward_official import WMRewardRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PhysV ABC groups with reusable runners.")
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["A", "B1", "B2", "B3", "C"],
        choices=list(GROUP_SPECS),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["pdi", "wmreward", "proxy"],
        choices=["pdi", "wmreward", "proxy", "videophy2"],
    )
    parser.add_argument("--refresh-pdi", action="store_true")
    parser.add_argument("--refresh-wmreward", action="store_true")
    parser.add_argument("--refresh-proxy", action="store_true")
    parser.add_argument("--refresh-videophy2", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--proxy-device", default=None)
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES"))
    parser.add_argument("--pdi-python", default=sys.executable)
    parser.add_argument("--wmreward-autocast-dtype", default="bfloat16", choices=["bfloat16", "float16", "none"])
    parser.add_argument("--videophy-python", default=str(VPHY_PYTHON))
    parser.add_argument("--videophy-cuda-visible-devices", default=None)
    return parser.parse_args()


def should_run_pdi(payload: dict[str, Any], refresh: bool) -> bool:
    return refresh or get_official_pdi(payload) is None or metric_value(payload, "official_pdi") is None


def should_run_wmreward(payload: dict[str, Any], refresh: bool) -> bool:
    return refresh or get_wmreward(payload) is None or metric_value(payload, "wmreward_jepa") is None


def should_run_proxy(payload: dict[str, Any], refresh: bool) -> bool:
    return refresh or get_proxy(payload) is None or metric_value(payload, "vjepa_proxy") is None


def should_run_videophy2(payload: dict[str, Any], refresh: bool) -> bool:
    return refresh or get_videophy2_auto(payload) is None or metric_value(payload, "videophy2_auto_pc") is None


def update_payload(
    json_path: Path,
    *,
    pdi_runner: OfficialPDIRunner | None,
    wmreward_runner: WMRewardRunner | None,
    proxy_runner: ProxyRunner | None,
    refresh_pdi: bool,
    refresh_wmreward: bool,
    refresh_proxy: bool,
) -> dict[str, bool]:
    payload = load_payload(json_path)
    video_path = resolve_video_path(json_path, payload)
    changed = {"pdi": False, "wmreward": False, "proxy": False}
    needs_save = False
    canonical_video = str(video_path)
    if payload.get("video") != canonical_video:
        payload["video"] = canonical_video
        needs_save = True
    if payload.get("video_path") is not None and payload.get("video_path") != canonical_video:
        payload["video_path"] = canonical_video
        needs_save = True

    if pdi_runner is not None and should_run_pdi(payload, refresh_pdi):
        result = pdi_runner.run(video_path, resolve_text_query(video_path, payload), refresh=refresh_pdi)
        set_official_pdi(payload, result)
        changed["pdi"] = True

    if wmreward_runner is not None and should_run_wmreward(payload, refresh_wmreward):
        result = wmreward_runner.score(video_path)
        set_wmreward(payload, result)
        changed["wmreward"] = True

    if proxy_runner is not None and should_run_proxy(payload, refresh_proxy):
        result = proxy_runner.score(video_path)
        if result is not None:
            set_proxy(payload, result)
            changed["proxy"] = True

    if any(changed.values()) or needs_save:
        save_payload(json_path, payload)
    return changed


def summarize_group(group_id: str) -> dict[str, int]:
    stats = {"total": 0, "pdi": 0, "wmreward": 0, "proxy": 0, "videophy2": 0}
    for json_path in iter_group_jsons(group_id):
        payload = load_payload(json_path)
        stats["total"] += 1
        if metric_value(payload, "official_pdi") is not None:
            stats["pdi"] += 1
        if metric_value(payload, "wmreward_jepa") is not None:
            stats["wmreward"] += 1
        if metric_value(payload, "vjepa_proxy") is not None:
            stats["proxy"] += 1
        if metric_value(payload, "videophy2_auto_pc") is not None:
            stats["videophy2"] += 1
    return stats


def run_videophy2_batch(
    *,
    groups: list[str],
    refresh: bool,
    python_bin: str,
    cuda_visible_devices: str | None,
) -> None:
    cmd = [
        python_bin,
        str(REPO_ROOT / "physics_sim" / "eval_videophy2_auto.py"),
        "--task",
        "pc",
        "--groups",
        *groups,
    ]
    if refresh:
        cmd.append("--refresh")
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    args = parse_args()

    enabled_metrics = set(args.metrics)
    pdi_runner = (
        OfficialPDIRunner(python_bin=args.pdi_python, cuda_visible_devices=args.cuda_visible_devices)
        if "pdi" in enabled_metrics
        else None
    )
    wmreward_runner = (
        WMRewardRunner(
            device=args.device,
            autocast_dtype=args.wmreward_autocast_dtype if args.wmreward_autocast_dtype != "none" else "none",
        )
        if "wmreward" in enabled_metrics
        else None
    )
    proxy_device = args.proxy_device or args.device
    proxy_runner = ProxyRunner(device=proxy_device) if "proxy" in enabled_metrics else None

    summary: dict[str, Any] = {}
    for group_id in args.groups:
        rows = iter_group_jsons(group_id)
        stats = {"total": len(rows), "pdi": 0, "wmreward": 0, "proxy": 0, "videophy2": 0}
        print(f"[{group_id}] {GROUP_SPECS[group_id].title}: {len(rows)} files", flush=True)
        for index, json_path in enumerate(rows, start=1):
            print(f"  [{index}/{len(rows)}] {json_path.name}", flush=True)
            changed = update_payload(
                json_path,
                pdi_runner=pdi_runner,
                wmreward_runner=wmreward_runner,
                proxy_runner=proxy_runner,
                refresh_pdi=args.refresh_pdi,
                refresh_wmreward=args.refresh_wmreward,
                refresh_proxy=args.refresh_proxy,
            )
            for key, flag in changed.items():
                if flag:
                    stats[key] += 1
        summary[group_id] = summarize_group(group_id)

    if "videophy2" in enabled_metrics:
        print("[videophy2] running official VideoPhy-2 AutoEval in vphy env", flush=True)
        run_videophy2_batch(
            groups=args.groups,
            refresh=args.refresh_videophy2,
            python_bin=args.videophy_python,
            cuda_visible_devices=args.videophy_cuda_visible_devices,
        )
        for group_id in args.groups:
            summary[group_id] = summarize_group(group_id)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
