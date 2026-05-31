from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a high-motion subset from an episode dataset.")
    parser.add_argument("--input-root", type=Path, required=True, help="Episode dataset root with train/val directories.")
    parser.add_argument("--output-root", type=Path, required=True, help="Directory for the selected subset.")
    parser.add_argument("--min-fgmean", type=float, default=0.020, help="Minimum foreground mean frame difference.")
    parser.add_argument("--min-path", type=float, default=0.60, help="Track path threshold for obvious motion.")
    parser.add_argument("--min-disp", type=float, default=0.18, help="Net displacement threshold for obvious motion.")
    parser.add_argument("--min-scale", type=float, default=0.0, help="Optional log-scale-span threshold.")
    parser.add_argument("--min-gmean", type=float, default=0.0, help="Optional global mean frame difference threshold.")
    parser.add_argument("--obvious-fgmean", type=float, default=0.0, help="Optional foreground mean threshold that also counts as obvious motion.")
    parser.add_argument("--max-train", type=int, default=0, help="Optional cap for train samples. 0 means keep all.")
    parser.add_argument("--max-val", type=int, default=0, help="Optional cap for val samples. 0 means keep all.")
    parser.add_argument("--symlink", action="store_true", help="Symlink instead of copying files.")
    return parser.parse_args()


def ensure_link_or_copy(src: Path, dst: Path, symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if symlink:
        dst.symlink_to(src)
    else:
        dst.write_bytes(src.read_bytes())


def motion_score(payload: dict[str, object]) -> float:
    track_motion = payload.get("track_motion") or {}
    clip_motion = payload.get("clip_motion") or {}
    path = float(track_motion.get("path_length", 0.0))
    disp = float(track_motion.get("net_displacement", 0.0))
    scale = float(track_motion.get("scale_span", 0.0))
    gmean = float(clip_motion.get("global_mean", 0.0))
    fgmean = float(clip_motion.get("foreground_mean", 0.0))
    return path + 2.0 * disp + 0.15 * scale + 8.0 * gmean + 10.0 * fgmean


def keep_record(payload: dict[str, object], args: argparse.Namespace) -> tuple[bool, dict[str, float]]:
    track_motion = payload.get("track_motion") or {}
    clip_motion = payload.get("clip_motion") or {}
    metrics = {
        "path_length": float(track_motion.get("path_length", 0.0)),
        "net_displacement": float(track_motion.get("net_displacement", 0.0)),
        "scale_span": float(track_motion.get("scale_span", 0.0)),
        "global_mean": float(clip_motion.get("global_mean", 0.0)),
        "foreground_mean": float(clip_motion.get("foreground_mean", 0.0)),
    }
    obvious_checks = []
    if args.min_path > 0.0:
        obvious_checks.append(metrics["path_length"] >= args.min_path)
    if args.min_disp > 0.0:
        obvious_checks.append(metrics["net_displacement"] >= args.min_disp)
    if args.min_scale > 0.0:
        obvious_checks.append(metrics["scale_span"] >= args.min_scale)
    if args.min_gmean > 0.0:
        obvious_checks.append(metrics["global_mean"] >= args.min_gmean)
    if args.obvious_fgmean > 0.0:
        obvious_checks.append(metrics["foreground_mean"] >= args.obvious_fgmean)
    obvious_motion = any(obvious_checks) if obvious_checks else True
    return metrics["foreground_mean"] >= args.min_fgmean and obvious_motion, metrics


def main() -> None:
    args = parse_args()
    summary: dict[str, object] = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "thresholds": {
            "min_fgmean": args.min_fgmean,
            "min_path": args.min_path,
            "min_disp": args.min_disp,
            "min_scale": args.min_scale,
            "min_gmean": args.min_gmean,
            "obvious_fgmean": args.obvious_fgmean,
        },
        "splits": {},
        "examples": [],
    }

    for split in ("train", "val"):
        input_dir = args.input_root / split
        if not input_dir.exists():
            continue
        records: list[dict[str, object]] = []
        rejected = 0
        for json_path in sorted(input_dir.glob("*.json")):
            npz_path = json_path.with_suffix(".npz")
            if not npz_path.exists():
                continue
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            keep, metrics = keep_record(payload, args)
            if not keep:
                rejected += 1
                continue
            source = payload.get("source") or {}
            records.append(
                {
                    "json_path": json_path,
                    "npz_path": npz_path,
                    "payload": payload,
                    "score": motion_score(payload),
                    "source_dataset": source.get("dataset", "unknown"),
                    "categories": source.get("categories") or [],
                    "metrics": metrics,
                }
            )

        records.sort(key=lambda item: item["score"], reverse=True)
        limit = args.max_train if split == "train" else args.max_val
        selected = records[:limit] if limit and limit > 0 else records

        split_dir = args.output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for item in selected:
            dst_npz = split_dir / item["npz_path"].name
            dst_json = split_dir / item["json_path"].name
            ensure_link_or_copy(item["npz_path"], dst_npz, args.symlink)
            ensure_link_or_copy(item["json_path"], dst_json, args.symlink)

        summary["splits"][split] = {
            "selected": len(selected),
            "eligible_before_cap": len(records),
            "rejected": rejected,
        }
        for item in selected[:10]:
            if len(summary["examples"]) >= 20:
                break
            summary["examples"].append(
                {
                    "split": split,
                    "sample": item["json_path"].stem,
                    "source_dataset": item["source_dataset"],
                    "categories": item["categories"],
                    "score": round(float(item["score"]), 6),
                    "metrics": item["metrics"],
                    "prompt": str(item["payload"].get("prompt", ""))[:300],
                }
            )

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
