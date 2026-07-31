#!/usr/bin/env python3
"""Add the OpenVid LoRA 34-config run to the shared dose-control case gallery."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_head_role_dose_control_case_gallery import (
    ensure_link,
    load_sidecar,
    media_url,
    metrics,
)


DEFAULT_CONFIG = Path(__file__).with_name(
    "head_role_openvid_lora_head34_experiment.json"
)
DEFAULT_GALLERY = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/"
    "gallery/head-role-dose-control-pilot"
)
MODEL = "openvid_lora_step10000"
MODEL_LABEL = "Wan+OpenVid LoRA (step 10000)"
ROUTE = "openvid-head34-generation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gallery-root", type=Path, default=DEFAULT_GALLERY)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.expanduser().resolve().read_text(encoding="utf-8"))
    root = Path(config["storage"]["output_root"]).expanduser().resolve()
    generation_root = root / "generation"
    gallery = args.gallery_root.expanduser().resolve()
    manifest_path = gallery / "manifest.json"
    shared = json.loads(manifest_path.read_text(encoding="utf-8"))
    subset_manifest = Path(config["matched_subset_manifest"]).expanduser().resolve()
    subsets = json.loads(subset_manifest.read_text(encoding="utf-8"))["subsets"]
    cases = {str(case["id"]) for case in shared["cases"]}
    seed = int(config["seeds"][0])

    ensure_link(generation_root, gallery / "media" / ROUTE)
    records: list[dict[str, Any]] = []
    baseline_root = (
        generation_root / MODEL / f"seed-{seed:06d}" / "baseline"
    )
    baseline_videos = {
        video.stem: video.resolve()
        for video in baseline_root.rglob("*.mp4")
        if video.stem in cases and video.stat().st_size > 1024
    }
    if set(baseline_videos) != cases:
        raise RuntimeError(
            f"OpenVid baseline has {len(baseline_videos)}/{len(cases)} cases"
        )
    for case_id, video in sorted(baseline_videos.items()):
        payload = load_sidecar(video)
        records.append(
            {
                "kind": "baseline",
                "model": MODEL,
                "seed": seed,
                "case_id": case_id,
                "subset_id": "baseline",
                "role": "baseline",
                "k": 0,
                "replicate": -1,
                "matching": "baseline",
                "start": -1,
                "end": -1,
                "video": media_url(video, generation_root, ROUTE),
                "sidecar": media_url(video.with_suffix(".json"), generation_root, ROUTE),
                "metrics": metrics(payload),
            }
        )

    complete_states = 0
    for state_path in sorted((root / "state").glob("*.json")):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("status") != "complete":
            continue
        complete_states += 1
        subset_id = str(state["subset_id"])
        subset = subsets[subset_id]
        matching = str(subset["matching"])
        kind = (
            "s_feature_split"
            if matching in {"exact_block_feature_contrast", "exact_block_feature_union"}
            else "s_dominant_depth"
        )
        videos = {
            case_id: Path(value).expanduser().resolve()
            for case_id, value in state["videos"].items()
        }
        if set(videos) != cases:
            raise RuntimeError(
                f"{state_path.name} has {len(videos)}/{len(cases)} cases"
            )
        for case_id, video in sorted(videos.items()):
            if not video.is_file() or not video.with_suffix(".json").is_file():
                raise FileNotFoundError(f"Missing OpenVid gallery asset: {video}")
            payload = load_sidecar(video)
            records.append(
                {
                    "kind": kind,
                    "model": MODEL,
                    "seed": int(state["seed"]),
                    "case_id": case_id,
                    "subset_id": subset_id,
                    "role": "S",
                    "feature_subtype": subset.get("feature_subtype"),
                    "dominance_class": subset.get("dominance_class"),
                    "depth_stratum": subset.get("depth_stratum"),
                    "k": int(state["k"]),
                    "replicate": int(subset.get("replicate", 0)),
                    "matching": matching,
                    "start": int(state["step_range"][0]),
                    "end": int(state["step_range"][1]),
                    "video": media_url(video, generation_root, ROUTE),
                    "sidecar": media_url(
                        video.with_suffix(".json"), generation_root, ROUTE
                    ),
                    "metrics": metrics(payload),
                }
            )

    if complete_states != 33 or len(records) != 680:
        raise RuntimeError(
            f"Expected 33 complete states and 680 records, got "
            f"{complete_states} states and {len(records)} records"
        )
    shared["records"] = [
        record for record in shared["records"] if record.get("model") != MODEL
    ] + records
    shared["models"] = [
        model for model in shared["models"] if model != MODEL
    ] + [MODEL]
    shared["model_labels"][MODEL] = MODEL_LABEL
    shared["openvid_head34_generation_tasks_complete"] = complete_states + 1
    shared["openvid_head34_generation_tasks_expected"] = 34
    shared["openvid_head34_videos_visible"] = len(records)
    shared["updated_utc"] = datetime.now(timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    atomic_json(manifest_path, shared)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "models": shared["models"],
                "openvid_records": len(records),
                "openvid_baselines": len(baseline_videos),
                "openvid_ablation_tasks": complete_states,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
