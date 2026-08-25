#!/usr/bin/env python3
"""Download a larger, independent Poly Haven background library.

The files are intentionally kept outside the source tree under
``/data/gaoya/agent-data/assets``.  The downloader uses Poly Haven's public
API and records URLs/checksums so the Eevee demo renders remain reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import requests


DEFAULT_ROOT = Path("/data/gaoya/agent-data/assets/texture_realism_backgrounds_20260825")
API_ROOT = "https://api.polyhaven.com"
USER_AGENT = "gaoya-physv-texture-realism-backgrounds/1.0"

# These are deliberately different from the three HDRIs used by the test70
# reference page (old_hall, brown_photostudio_02, poly_haven_studio).
HDRI_IDS = (
    "empty_warehouse_01",
    "machine_shop_02",
    "colorful_studio",
    "glasshouse_interior",
    "courtyard",
    "industrial_workshop_foundry",
    "auto_service",
    "ferndale_studio_05",
)

# A small set of actual surface maps makes the new sets visibly different,
# instead of merely changing the world light.
TEXTURE_IDS = (
    "brick_wall_001",
    "brick_wall_003",
    "blue_metal_plate",
    "box_profile_metal_sheet",
    "asphalt_floor",
    "brick_floor",
    "concrete_block_wall",
    "blue_plaster_wall",
)

# Object-facing materials are deliberately separate from the set/background
# library.  They provide visibly non-uniform albedo while preserving the
# physical material cues in the normal, roughness, and AO maps.
OBJECT_TEXTURE_IDS = (
    "rubber_tiles",
    "rubberized_track",
    "metal_plate_02",
    "rusty_metal_03",
    "painted_metal_shutter",
    "dark_wood",
    "oak_wood_planks",
    "hessian_230",
    "fabric_leather_01",
    "denim_fabric_03",
)
ALL_TEXTURE_IDS = tuple(dict.fromkeys((*TEXTURE_IDS, *OBJECT_TEXTURE_IDS)))


def request_json(session: requests.Session, url: str) -> dict[str, Any]:
    response = session.get(url, timeout=90)
    response.raise_for_status()
    return response.json()


def download(session: requests.Session, url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with session.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    temporary.replace(destination)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_texture_url(entry: dict[str, Any], resolution: str) -> str:
    files = entry.get(resolution)
    if not files:
        # Poly Haven occasionally exposes only a neighbouring resolution.
        resolution_key = sorted(entry.keys())[-1]
        files = entry[resolution_key]
    for extension in ("jpg", "png", "exr"):
        if extension in files:
            return str(files[extension]["url"])
    return str(next(iter(files.values()))["url"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--hdri-resolution", default="2k", choices=("1k", "2k", "4k"))
    parser.add_argument("--texture-resolution", default="2k", choices=("1k", "2k", "4k"))
    args = parser.parse_args()

    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    manifest: dict[str, Any] = {
        "source": "Poly Haven",
        "source_url": "https://polyhaven.com/",
        "license": "CC0",
        "license_url": "https://polyhaven.com/license",
        "hdri_resolution": args.hdri_resolution,
        "texture_resolution": args.texture_resolution,
        "hdris": {},
        "textures": {},
        "object_texture_ids": list(OBJECT_TEXTURE_IDS),
    }

    for asset_id in HDRI_IDS:
        metadata = request_json(session, f"{API_ROOT}/files/{asset_id}")
        entry = metadata["hdri"][args.hdri_resolution]["hdr"]
        url = str(entry["url"])
        destination = root / "hdris" / asset_id / Path(url).name
        print(f"[HDRI] {asset_id} -> {destination}", flush=True)
        download(session, url, destination)
        manifest["hdris"][asset_id] = {
            "asset_id": asset_id,
            "path": str(destination),
            "source_url": url,
            "sha256": sha256(destination),
            "resolution": args.hdri_resolution,
            "format": "hdr",
        }

    for asset_id in ALL_TEXTURE_IDS:
        metadata = request_json(session, f"{API_ROOT}/files/{asset_id}")
        texture_root = root / "textures" / asset_id
        maps: dict[str, dict[str, str]] = {}
        for map_name, polyhaven_key in (
            ("albedo", "Diffuse"),
            ("roughness", "Rough"),
            ("normal", "nor_gl"),
            ("ao", "AO"),
        ):
            if polyhaven_key not in metadata:
                continue
            url = choose_texture_url(metadata[polyhaven_key], args.texture_resolution)
            destination = texture_root / Path(url).name
            print(f"[TEXTURE] {asset_id}/{map_name} -> {destination}", flush=True)
            download(session, url, destination)
            maps[map_name] = {
                "path": str(destination),
                "source_url": url,
                "sha256": sha256(destination),
            }
        manifest["textures"][asset_id] = {
            "asset_id": asset_id,
            "role": "object" if asset_id in OBJECT_TEXTURE_IDS else "background",
            "resolution": args.texture_resolution,
            "maps": maps,
        }

    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[DONE] {root}", flush=True)


if __name__ == "__main__":
    main()
