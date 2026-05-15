# 用途：从官方开放资源下载一小套 Blender Cycles 渲染资产库到 /data/gaoya/dataset，供 Genesis 预览脚本直接复用。
"""Download a compact Poly Haven render asset library for Cycles previews.

The library is intentionally small and reproducible:
- 3 HDRIs for indoor / studio lighting
- several PBR texture sets for floor / wall / cloth-like surfaces

All assets are downloaded through the official Poly Haven API and saved under
the target root with a manifest.json that the preview builder can consume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import requests


DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/dataset/blender_render_assets/polyhaven_v1")
USER_AGENT = "gaoya-blender-render-setup/1.0"

HDRI_SPECS = {
    "poly_haven_studio": {"resolution": "4k", "format": "hdr", "slot": "studio_soft"},
    "brown_photostudio_02": {"resolution": "4k", "format": "hdr", "slot": "studio_warm"},
    "old_hall": {"resolution": "4k", "format": "hdr", "slot": "interior_warm"},
}

TEXTURE_SPECS = {
    "painted_concrete": {"resolution": "2k", "slot": "painted_concrete"},
    "wood_floor": {"resolution": "2k", "slot": "wood_floor"},
    "beige_wall_001": {"resolution": "2k", "slot": "beige_wall_001"},
    "fabric_pattern_07": {"resolution": "2k", "slot": "fabric_pattern_07"},
    "brown_leather": {"resolution": "2k", "slot": "brown_leather"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256sum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def request_json(url: str) -> dict[str, Any]:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=60)
    r.raise_for_status()
    return r.json()


def download_file(url: str, dst: Path, *, overwrite: bool) -> Path:
    if dst.exists() and not overwrite:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=120, stream=True) as r:
        r.raise_for_status()
        with dst.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return dst


def choose_texture_url(entry: dict[str, Any], resolution: str) -> str:
    if resolution not in entry:
        raise KeyError(f"resolution {resolution} not available")
    preferred_exts = ("jpg", "png", "exr")
    files = entry[resolution]
    for ext in preferred_exts:
        if ext in files:
            return str(files[ext]["url"])
    first_key = next(iter(files))
    return str(files[first_key]["url"])


def build_manifest(output_root: Path, *, overwrite: bool) -> dict[str, Any]:
    hdris: dict[str, Any] = {}
    textures: dict[str, Any] = {}

    for asset_id, spec in HDRI_SPECS.items():
        meta = request_json(f"https://api.polyhaven.com/files/{asset_id}")
        hdri_meta = meta["hdri"][spec["resolution"]][spec["format"]]
        filename = Path(str(hdri_meta["url"])).name
        dst = output_root / "hdris" / asset_id / filename
        download_file(str(hdri_meta["url"]), dst, overwrite=overwrite)
        hdris[spec["slot"]] = {
            "asset_id": asset_id,
            "path": str(dst),
            "source_url": str(hdri_meta["url"]),
            "sha256": sha256sum(dst),
            "resolution": spec["resolution"],
            "format": spec["format"],
        }

    for asset_id, spec in TEXTURE_SPECS.items():
        meta = request_json(f"https://api.polyhaven.com/files/{asset_id}")
        texture_root = output_root / "textures" / asset_id
        maps: dict[str, str] = {}
        for manifest_key, poly_key in (
            ("base_color", "Diffuse"),
            ("roughness", "Rough"),
            ("normal", "nor_gl"),
            ("ao", "AO"),
        ):
            if poly_key not in meta:
                continue
            url = choose_texture_url(meta[poly_key], spec["resolution"])
            filename = Path(url).name
            dst = texture_root / filename
            download_file(url, dst, overwrite=overwrite)
            maps[manifest_key] = str(dst)
        textures[spec["slot"]] = {
            "asset_id": asset_id,
            "resolution": spec["resolution"],
            "maps": maps,
        }

    return {
        "source": "Poly Haven",
        "source_url": "https://polyhaven.com/",
        "license": "CC0",
        "license_url": "https://polyhaven.com/license",
        "hdris": hdris,
        "textures": textures,
    }


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(output_root, overwrite=bool(args.overwrite))
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] asset library ready: {output_root}")
    print(f"[DONE] manifest: {output_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
