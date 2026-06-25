from __future__ import annotations

import argparse

from .datasets import GROUP_SPECS, iter_group_jsons
from .records import (
    get_cosmos_reason1,
    load_payload,
    resolve_video_path,
    save_payload,
    set_cosmos_reason1,
)
from .single_case.cosmos_reason1 import score_case


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official-compatible Cosmos-Reason1 physical plausibility scoring.")
    parser.add_argument("--groups", nargs="+", default=["A", "B1", "B2", "B3", "C"], choices=list(GROUP_SPECS))
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def should_run(payload: dict, refresh: bool) -> bool:
    if refresh:
        return True
    bucket = get_cosmos_reason1(payload)
    return not isinstance(bucket, dict) or bucket.get("score") is None


def main() -> None:
    args = parse_args()
    for group_id in args.groups:
        rows = iter_group_jsons(group_id)
        print(f"[cosmos:{group_id}] {len(rows)} files", flush=True)
        for index, json_path in enumerate(rows, start=1):
            payload = load_payload(json_path)
            if not should_run(payload, args.refresh):
                print(f"  [{index}/{len(rows)}] skip {json_path.name}", flush=True)
                continue
            video_path = resolve_video_path(json_path, payload)
            print(f"  [{index}/{len(rows)}] {json_path.name}", flush=True)
            result = score_case(video_path)
            set_cosmos_reason1(payload, result)
            save_payload(json_path, payload)


if __name__ == "__main__":
    main()
