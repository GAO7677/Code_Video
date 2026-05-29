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
from .paths import A_OUTPUT, REPO_ROOT, VPHY_PYTHON
from .paths import FLUX_PYTHON
from .proxy_runner import ProxyRunner
from .records import (
    get_cosmos_reason1,
    get_phyground,
    get_official_pdi,
    get_proxy,
    get_videophy2_auto,
    get_wmreward,
    load_payload,
    metric_value,
    resolve_video_path,
    save_payload,
    set_cosmos_reason1,
    set_official_pdi,
    set_phyground,
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
        choices=["pdi", "wmreward", "proxy", "videophy2", "phyground", "cosmos"],
    )
    parser.add_argument("--refresh-pdi", action="store_true")
    parser.add_argument("--refresh-wmreward", action="store_true")
    parser.add_argument("--refresh-proxy", action="store_true")
    parser.add_argument("--refresh-videophy2", action="store_true")
    parser.add_argument("--refresh-phyground", action="store_true")
    parser.add_argument("--refresh-cosmos", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--proxy-device", default=None)
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES"))
    parser.add_argument("--pdi-python", default=sys.executable)
    parser.add_argument("--wmreward-cuda-visible-devices", default=None)
    parser.add_argument("--videophy-python", default=str(VPHY_PYTHON))
    parser.add_argument("--videophy-cuda-visible-devices", default=None)
    parser.add_argument("--flux-python", default=str(FLUX_PYTHON))
    parser.add_argument("--flux-cuda-visible-devices", default=None)
    return parser.parse_args()


def should_run_pdi(payload: dict[str, Any], refresh: bool) -> bool:
    return refresh or get_official_pdi(payload) is None or metric_value(payload, "official_pdi") is None


def should_run_wmreward(payload: dict[str, Any], refresh: bool) -> bool:
    return refresh or get_wmreward(payload) is None or metric_value(payload, "wmreward_jepa") is None


def should_run_proxy(payload: dict[str, Any], refresh: bool) -> bool:
    return refresh or get_proxy(payload) is None or metric_value(payload, "vjepa_proxy") is None


def should_run_videophy2(payload: dict[str, Any], refresh: bool) -> bool:
    return refresh or get_videophy2_auto(payload) is None or metric_value(payload, "videophy2_auto_pc") is None


def should_run_phyground(payload: dict[str, Any], refresh: bool) -> bool:
    return refresh or get_phyground(payload) is None or metric_value(payload, "phyground_general_avg") is None


def should_run_cosmos(payload: dict[str, Any], refresh: bool) -> bool:
    return refresh or get_cosmos_reason1(payload) is None or metric_value(payload, "cosmos_reason1") is None


def resolve_proxy_context_video(json_path: Path, video_path: Path) -> Path:
    try:
        rel = json_path.relative_to(A_OUTPUT)
    except ValueError:
        return video_path

    if not rel.parts:
        return video_path
    method = rel.parts[0]
    if method == "GT":
        return video_path

    gt_json = A_OUTPUT / "GT" / Path(*rel.parts[1:])
    if not gt_json.is_file():
        return video_path

    gt_payload = load_payload(gt_json)
    return resolve_video_path(gt_json, gt_payload)


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
        result = proxy_runner.score(
            video_path,
            context_video_path=resolve_proxy_context_video(json_path, video_path),
        )
        if result is not None:
            set_proxy(payload, result)
            changed["proxy"] = True

    if any(changed.values()) or needs_save:
        save_payload(json_path, payload)
    return changed


def summarize_group(group_id: str) -> dict[str, int]:
    stats = {"total": 0, "pdi": 0, "wmreward": 0, "proxy": 0, "videophy2": 0, "phyground": 0, "cosmos": 0}
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
        if metric_value(payload, "phyground_general_avg") is not None:
            stats["phyground"] += 1
        if metric_value(payload, "cosmos_reason1") is not None:
            stats["cosmos"] += 1
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


def run_flux_batch(
    *,
    script_name: str,
    groups: list[str],
    refresh: bool,
    python_bin: str,
    cuda_visible_devices: str | None,
    extra_args: list[str] | None = None,
) -> None:
    cmd = [
        python_bin,
        str(REPO_ROOT / "physics_sim" / script_name),
        "--groups",
        *groups,
    ]
    if refresh:
        cmd.append("--refresh")
    if extra_args:
        cmd.extend(extra_args)
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
            cuda_visible_devices=args.wmreward_cuda_visible_devices or args.cuda_visible_devices,
        )
        if "wmreward" in enabled_metrics
        else None
    )
    proxy_device = args.proxy_device or args.device
    proxy_runner = ProxyRunner(device=proxy_device) if "proxy" in enabled_metrics else None

    summary: dict[str, Any] = {}
    for group_id in args.groups:
        rows = iter_group_jsons(group_id)
        stats = {"total": len(rows), "pdi": 0, "wmreward": 0, "proxy": 0, "videophy2": 0, "phyground": 0, "cosmos": 0}
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

    if "phyground" in enabled_metrics:
        print("[phyground] running official-compatible PhyGround batch in flux env", flush=True)
        run_flux_batch(
            script_name="eval_phyground.py",
            groups=args.groups,
            refresh=args.refresh_phyground,
            python_bin=args.flux_python,
            cuda_visible_devices=args.flux_cuda_visible_devices or args.cuda_visible_devices,
            extra_args=["--general-only"],
        )
        for group_id in args.groups:
            summary[group_id] = summarize_group(group_id)

    if "cosmos" in enabled_metrics:
        print("[cosmos] running official-compatible Cosmos Reason1 batch in flux env", flush=True)
        run_flux_batch(
            script_name="eval_cosmos_reason1.py",
            groups=args.groups,
            refresh=args.refresh_cosmos,
            python_bin=args.flux_python,
            cuda_visible_devices=args.flux_cuda_visible_devices or args.cuda_visible_devices,
        )
        for group_id in args.groups:
            summary[group_id] = summarize_group(group_id)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
