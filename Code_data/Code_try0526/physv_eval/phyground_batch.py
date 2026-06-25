from __future__ import annotations

import argparse
from pathlib import Path

from .datasets import GROUP_SPECS, iter_group_jsons
from .phyground_official import GENERAL_METRICS
from .records import get_phyground, load_payload, resolve_video_path, save_payload, set_phyground
from .single_case.phyground import score_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official-compatible PhyGround evaluation.")
    parser.add_argument("--groups", nargs="+", default=["A", "B1", "B2", "B3", "C"], choices=list(GROUP_SPECS))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--general-only", action="store_true")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-pixels", type=int, default=360 * 640)
    return parser.parse_args()


def should_run(payload: dict, refresh: bool) -> bool:
    if refresh:
        return True
    bucket = get_phyground(payload)
    if not isinstance(bucket, dict):
        return True
    general = bucket.get("general", {})
    return any(general.get(name) is None for name in GENERAL_METRICS)


def main() -> None:
    args = parse_args()
    for group_id in args.groups:
        rows = iter_group_jsons(group_id)
        print(f"[phyground:{group_id}] {len(rows)} files", flush=True)
        for index, json_path in enumerate(rows, start=1):
            payload = load_payload(json_path)
            if not should_run(payload, args.refresh):
                print(f"  [{index}/{len(rows)}] skip {json_path.name}", flush=True)
                continue
            video_path = resolve_video_path(json_path, payload)
            caption = payload.get("caption") or payload.get("description") or payload.get("prompt") or video_path.stem
            laws = [] if args.general_only else None
            print(f"  [{index}/{len(rows)}] {json_path.name}", flush=True)
            result = score_case(
                video_path,
                caption=str(caption),
                metrics=list(GENERAL_METRICS),
                laws=laws,
            )
            set_phyground(payload, result)
            save_payload(json_path, payload)


if __name__ == "__main__":
    main()
