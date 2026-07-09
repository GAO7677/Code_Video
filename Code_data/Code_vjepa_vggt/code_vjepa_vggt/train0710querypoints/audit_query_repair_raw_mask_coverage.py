"""
Audit how often GT-mask query repair can resolve raw mask assets.

Example:
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0710querypoints/audit_query_repair_raw_mask_coverage.py \
  --kubric-root /data/gaoya/dataset/nnsriram97-phyco_kubric \
  --kubric-limit 64 \
  --phys-state-root /data/gaoya/AAA_test_video/0529/phys_state_video/openvid_clip_smoke_ep256_lb \
  --phys-state-limit 64 \
  --output-json /data/gaoya/agent-data/outputs/query_repair_raw_mask_coverage/audit.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from code_vjepa_vggt.train0710querypoints.gt_mask_query_repair import resolve_raw_sample_dir_from_sample


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _audit_kubric_root(root: Path, *, limit: int) -> dict[str, Any]:
    rgba_paths = sorted(root.glob("**/rgba.mp4"))
    if limit > 0:
        rgba_paths = rgba_paths[:limit]
    results: list[dict[str, Any]] = []
    for rgba_path in rgba_paths:
        metadata_path = rgba_path.with_name("metadata.json")
        metadata = _load_json(metadata_path) if metadata_path.is_file() else {}
        sample = {
            "video_path": str(rgba_path),
            "metadata": {
                "source_video_path": str(rgba_path),
                **metadata,
            },
        }
        resolved, debug = resolve_raw_sample_dir_from_sample(sample)
        results.append(
            {
                "sample_key": str(rgba_path.parent.relative_to(root)),
                "video_path": str(rgba_path),
                "resolved": resolved is not None,
                "resolved_sample_dir": None if resolved is None else str(resolved),
                "debug": debug,
            }
        )
    return _summarize_results("kubric", str(root), results)


def _audit_phys_state_root(root: Path, *, split: str, limit: int) -> dict[str, Any]:
    split_root = root / split
    meta_paths = sorted(split_root.glob("*.json"))
    if limit > 0:
        meta_paths = meta_paths[:limit]
    results: list[dict[str, Any]] = []
    for meta_path in meta_paths:
        npz_path = meta_path.with_suffix(".npz")
        metadata = _load_json(meta_path)
        sample = {
            "video_path": str(npz_path),
            "metadata": metadata,
        }
        resolved, debug = resolve_raw_sample_dir_from_sample(sample)
        results.append(
            {
                "sample_key": meta_path.stem,
                "meta_path": str(meta_path),
                "video_path": str(npz_path),
                "resolved": resolved is not None,
                "resolved_sample_dir": None if resolved is None else str(resolved),
                "debug": debug,
            }
        )
    return _summarize_results("phys_state", str(split_root), results)


def _summarize_results(kind: str, root: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    reason_counter = Counter()
    resolver_counter = Counter()
    resolved_count = 0
    for item in results:
        if item["resolved"]:
            resolved_count += 1
            resolver_counter[item["debug"].get("resolver", "resolved")] += 1
        else:
            reason_counter[item["debug"].get("resolver", "unresolved")] += 1
    unresolved_examples = [item for item in results if not item["resolved"]][:10]
    resolved_examples = [item for item in results if item["resolved"]][:10]
    return {
        "kind": kind,
        "root": root,
        "total": len(results),
        "resolved": resolved_count,
        "unresolved": len(results) - resolved_count,
        "resolved_ratio": (float(resolved_count) / float(len(results))) if results else 0.0,
        "resolver_counter": dict(resolver_counter),
        "unresolved_reason_counter": dict(reason_counter),
        "resolved_examples": resolved_examples,
        "unresolved_examples": unresolved_examples,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kubric-root", type=Path, default=None)
    parser.add_argument("--kubric-limit", type=int, default=64)
    parser.add_argument("--phys-state-root", type=Path, default=None)
    parser.add_argument("--phys-state-split", type=str, default="train")
    parser.add_argument("--phys-state-limit", type=int, default=64)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload: dict[str, Any] = {"audits": []}
    if args.kubric_root is not None:
        payload["audits"].append(
            _audit_kubric_root(args.kubric_root.expanduser().resolve(), limit=max(int(args.kubric_limit), 0))
        )
    if args.phys_state_root is not None:
        payload["audits"].append(
            _audit_phys_state_root(
                args.phys_state_root.expanduser().resolve(),
                split=str(args.phys_state_split),
                limit=max(int(args.phys_state_limit), 0),
            )
        )

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output_json is not None:
        output_json = args.output_json.expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(text, encoding="utf-8")
        print(output_json)
        return
    print(text)


if __name__ == "__main__":
    main()
