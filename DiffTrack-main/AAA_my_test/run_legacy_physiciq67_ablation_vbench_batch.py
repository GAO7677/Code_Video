#!/usr/bin/env python3
"""Batch the expensive per-frame VBench dimensions and backfill manifests."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
from typing import Any

from physv_eval.vbench_official import OfficialVBenchRunner


OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/legacy_physiciq67_attention_ablation_vbench"
)
DIMENSIONS = {
    "vbench_aesthetic_quality": "aesthetic_quality",
    "vbench_imaging_quality": "imaging_quality",
}


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", required=True, choices=sorted(DIMENSIONS))
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    snapshot = load_json(output_root / "snapshot.json")
    metric = str(args.metric)
    dimension = DIMENSIONS[metric]
    pending: list[dict[str, Any]] = []
    for row in snapshot.get("records", []):
        manifest_path = Path(str(row["run_dir"])) / "manifest.json"
        manifest = load_json(manifest_path)
        if manifest.get(metric) is not None:
            continue
        video_path = Path(str(row["video"])).expanduser().resolve()
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        pending.append(
            {
                "manifest_path": manifest_path,
                "video_path": video_path,
                "caption": manifest.get("input_caption") or manifest.get("caption"),
            }
        )

    print(f"[vbench-batch:start] metric={metric} pending={len(pending)}")
    if not pending:
        print(f"[vbench-batch:done] metric={metric} written=0")
        return

    runner = OfficialVBenchRunner(
        output_root=output_root / "raw" / metric,
        device=str(args.device),
    )
    batch_output = output_root / "raw" / metric / "legacy_physiciq67_batch"
    result = runner.score_batch(
        [
            {"video": str(row["video_path"]), "caption": row["caption"]}
            for row in pending
        ],
        dimension=dimension,
        output_path=batch_output,
        run_name=f"legacy_physiciq67_{dimension}",
    )
    raw_by_video = {
        str(Path(str(item["video_path"])).expanduser().resolve()): item
        for item in result.get("raw_results", [])
        if isinstance(item, dict) and item.get("video_path")
    }
    if len(raw_by_video) != len(pending):
        raise RuntimeError(
            f"VBench returned {len(raw_by_video)} per-video values for {len(pending)} inputs"
        )

    written = 0
    for row in pending:
        video_key = str(row["video_path"])
        raw = raw_by_video[video_key]
        score = float(raw["video_results"])
        if dimension == "imaging_quality":
            score /= 100.0
        metric_payload = {
            "score": round(score, 4),
            "dimension": dimension,
            "metric_direction": "higher_is_better",
            "official": True,
            "method": "vbench_official_custom_input",
            "raw_dimension_score": round(score, 4),
            "raw_results": [raw],
            "video": video_key,
            "caption_used": row["caption"],
            "result_json": result.get("result_json"),
            "full_info_json": result.get("full_info_json"),
            "output_path": result.get("output_path"),
            "cache_dir": result.get("cache_dir"),
            "device": str(args.device),
            "mode": "custom_full_info_batch",
        }
        manifest_path = row["manifest_path"]
        lock_path = manifest_path.with_name(f"{manifest_path.name}.lock")
        with lock_path.open("w", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            manifest = load_json(manifest_path)
            if manifest.get(metric) is None:
                manifest[metric] = metric_payload
                write_json_atomic(manifest_path, manifest)
                written += 1
        if written % 25 == 0 or written == len(pending):
            print(f"[vbench-batch:write] metric={metric} {written}/{len(pending)}")
    print(f"[vbench-batch:done] metric={metric} written={written}")


if __name__ == "__main__":
    main()
