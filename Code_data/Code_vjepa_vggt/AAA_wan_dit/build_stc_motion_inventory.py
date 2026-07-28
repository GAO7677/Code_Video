#!/usr/bin/env python3
"""Build a strict video inventory for S/T/C phased motion analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any


DEFAULT_GALLERY_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/gallery/multiseed"
)
DEFAULT_MANIFEST = DEFAULT_GALLERY_ROOT / "manifest.json"
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_stc_motion_analysis"
)
DEFAULT_SEEDS = (851, 3278, 11395, 20379, 28221, 32098)
DEFAULT_MODELS = ("wan_lora", "xssc", "physrvg")
PHASED_VARIANT = re.compile(r"^(?P<role>[STC])_steps(?P<start>\d{2})_(?P<end>\d{2})$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--roles", nargs="+", choices=("S", "T", "C"), default=["S", "T", "C"])
    parser.add_argument(
        "--require-all-seeds",
        action="store_true",
        help="Fail if any selected model/variant is absent for a selected seed.",
    )
    return parser.parse_args()


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    canonical = str(path.resolve())
    material = f"{canonical}\0{stat.st_size}\0{stat.st_mtime_ns}".encode()
    return {
        "path": canonical,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "cache_key": hashlib.sha256(material).hexdigest()[:24],
    }


def validate_video(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 1024


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    gallery_root = manifest_path.parent
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    if len(cases) != 1:
        raise ValueError(f"Expected one case in manifest, found {len(cases)}")
    case = cases[0]
    case_id = str(case["id"])
    video_table = payload["videos"][case_id]
    selected_seeds = [int(seed) for seed in args.seeds]
    selected_models = [str(model) for model in args.models]
    selected_roles = set(args.roles)

    variants = [
        variant
        for variant in payload["variants"]
        if (match := PHASED_VARIANT.fullmatch(variant))
        and match.group("role") in selected_roles
    ]
    variants.sort(
        key=lambda value: (
            "STC".index(PHASED_VARIANT.fullmatch(value).group("role")),
            int(PHASED_VARIANT.fullmatch(value).group("start")),
            int(PHASED_VARIANT.fullmatch(value).group("end")),
        )
    )

    entries: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    gt_path = (
        gallery_root
        / "media"
        / "references"
        / f"{case_id}__source_video_49f.mp4"
    )
    if not validate_video(gt_path):
        gt_path = Path(case["source_video"])
    if not validate_video(gt_path):
        raise FileNotFoundError(f"Missing GT/source video: {gt_path}")
    entries.append(
        {
            "entry_id": "gt",
            "kind": "gt",
            "case_id": case_id,
            "model": "gt",
            "seed": None,
            "variant": "gt",
            "role": None,
            "denoise_step_range": None,
            "source": fingerprint(gt_path),
        }
    )

    for seed in selected_seeds:
        seed_key = str(seed)
        if seed_key not in video_table:
            raise KeyError(f"Seed {seed} is absent from the gallery manifest")
        for model in selected_models:
            model_table = video_table[seed_key][model]
            for variant in ["baseline", *variants]:
                relative_path = model_table.get(variant)
                path = gallery_root / relative_path if relative_path else None
                if path is None or not validate_video(path):
                    missing.append(
                        {
                            "model": model,
                            "seed": seed,
                            "variant": variant,
                            "path": str(path) if path else None,
                        }
                    )
                    continue
                match = PHASED_VARIANT.fullmatch(variant)
                entries.append(
                    {
                        "entry_id": (
                            f"{model}__seed-{seed:06d}__{variant}"
                        ),
                        "kind": "generated",
                        "case_id": case_id,
                        "model": model,
                        "seed": seed,
                        "variant": variant,
                        "role": match.group("role") if match else None,
                        "denoise_step_range": (
                            [int(match.group("start")), int(match.group("end"))]
                            if match
                            else None
                        ),
                        "source": fingerprint(path),
                    }
                )

    if args.require_all_seeds and missing:
        examples = "\n".join(
            f"  {item['model']} seed={item['seed']} {item['variant']}"
            for item in missing[:20]
        )
        raise RuntimeError(f"{len(missing)} expected videos are missing:\n{examples}")

    output = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "case": case,
        "generation": payload.get("generation", {}),
        "selected_seeds": selected_seeds,
        "selected_models": selected_models,
        "selected_roles": sorted(selected_roles),
        "selected_variants": variants,
        "entries": entries,
        "missing": missing,
        "counts": {
            "entries": len(entries),
            "generated": sum(item["kind"] == "generated" for item in entries),
            "missing": len(missing),
        },
    }
    output_path = args.output_root / "inventory.json"
    atomic_write_json(output_path, output)
    print(
        f"[inventory] entries={len(entries)} generated={output['counts']['generated']} "
        f"missing={len(missing)}"
    )
    print(output_path)


if __name__ == "__main__":
    main()
