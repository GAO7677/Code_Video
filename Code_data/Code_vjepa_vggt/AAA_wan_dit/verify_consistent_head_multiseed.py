#!/usr/bin/env python3
"""Verify paired baseline and six Head-zero videos for seeds 42 through 46."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from consistent_head_targets import (
    CATEGORIES,
    load_consistent_category_targets,
)


CASE = "0613pybullet_sample_001460_w002"
VARIANTS = ("baseline", *CATEGORIES)
FFPROBE = Path("/data/gaoya/miniconda3/envs/vjepa2/bin/ffprobe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--legacy-ablation-root", type=Path, required=True)
    parser.add_argument("--legacy-baseline-video", type=Path, required=True)
    parser.add_argument("--classification-metadata", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe(path: Path) -> str:
    return subprocess.check_output(
        [
            str(FFPROBE),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,nb_frames,r_frame_rate",
            "-of",
            "csv=p=0",
            str(path),
        ],
        text=True,
    ).strip()


def _tag(variant: str) -> str:
    if variant == "baseline":
        return "baseline"
    return f"self_attn_consistent_head_zero_category_{variant.lower()}"


def main() -> None:
    args = parse_args()
    new_root = args.new_root.expanduser().resolve()
    legacy_root = args.legacy_ablation_root.expanduser().resolve()
    legacy_baseline = args.legacy_baseline_video.expanduser().resolve()
    targets, source = load_consistent_category_targets(
        args.classification_metadata
    )
    records = []
    failures = []
    for seed in args.seeds:
        hashes: dict[str, str] = {}
        for variant in VARIANTS:
            if seed == 42:
                if variant == "baseline":
                    video = legacy_baseline
                    sidecar = None
                else:
                    directory = legacy_root / _tag(variant)
                    video = directory / f"{CASE}.mp4"
                    sidecar = directory / f"{CASE}.json"
            else:
                directory = new_root / f"seed_{seed:04d}" / _tag(variant)
                video = directory / f"{CASE}.mp4"
                sidecar = directory / f"{CASE}.json"
            checks = {
                "video_exists": video.is_file(),
                "video_probe": False,
                "sidecar": seed == 42 and variant == "baseline",
                "seed": seed == 42 and variant == "baseline",
                "variant": seed == 42 and variant == "baseline",
                "targets": seed == 42 and variant == "baseline",
                "calls": seed == 42 and variant == "baseline",
                "classification_sha256": (
                    seed == 42 and variant == "baseline"
                ),
            }
            payload = None
            if video.is_file():
                checks["video_probe"] = _probe(video) == "896,512,30/1,49"
                hashes[variant] = _sha256(video)
            if sidecar is not None and sidecar.is_file():
                payload = json.loads(sidecar.read_text(encoding="utf-8"))
                metadata = payload.get("dit_ablation", {})
                checks["sidecar"] = payload.get("status") == "generated"
                checks["seed"] = payload.get("seed") == seed
                checks["variant"] = (
                    payload.get("experiment", {}).get("variant") == variant
                    if seed != 42
                    else metadata.get("category") == variant
                )
                if variant == "baseline":
                    checks["targets"] = metadata.get("num_targets") == 0
                    checks["calls"] = (
                        metadata.get("observed_target_forward_calls") == 0
                    )
                    checks["classification_sha256"] = True
                else:
                    actual = [
                        (int(item["block_id"]), int(item["head_id"]))
                        for item in metadata.get("targets", [])
                    ]
                    expected_calls = (
                        len(targets[variant]) * 40 * 2
                    )
                    checks["targets"] = actual == targets[variant]
                    checks["calls"] = (
                        metadata.get("observed_target_forward_calls")
                        == expected_calls
                        and metadata.get("expected_target_forward_calls")
                        == expected_calls
                        and metadata.get("target_forward_call_count_ok")
                        is True
                    )
                    checks["classification_sha256"] = (
                        metadata.get("target_selection", {}).get("sha256")
                        == source["sha256"]
                    )
            record = {
                "seed": seed,
                "variant": variant,
                "video": str(video),
                "sha256": hashes.get(variant),
                "checks": checks,
                "all_checks": all(checks.values()),
            }
            records.append(record)
            if not record["all_checks"]:
                failures.append(record)
        if len(hashes) == len(VARIANTS) and len(set(hashes.values())) != len(VARIANTS):
            failures.append(
                {
                    "seed": seed,
                    "error": "duplicate video hashes within paired variants",
                    "hashes": hashes,
                }
            )

    reversibility_path = new_root / "reversibility_check.json"
    reversibility = (
        json.loads(reversibility_path.read_text(encoding="utf-8"))
        if reversibility_path.is_file()
        else {}
    )
    reversibility_ok = (
        reversibility.get("decoded_frames_exact_match") is True
    )
    if not reversibility_ok:
        failures.append({"error": "missing or failed reversibility check"})
    report = {
        "seeds": args.seeds,
        "variants": list(VARIANTS),
        "num_expected": len(args.seeds) * len(VARIANTS),
        "num_verified": sum(record["all_checks"] for record in records),
        "classification_source": source,
        "reversibility": reversibility,
        "reversibility_ok": reversibility_ok,
        "records": records,
        "failures": failures,
        "all_checks": not failures,
    }
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else new_root / "verification_report.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(output),
                "num_verified": report["num_verified"],
                "num_expected": report["num_expected"],
                "reversibility_ok": reversibility_ok,
                "all_checks": report["all_checks"],
            },
            ensure_ascii=False,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
