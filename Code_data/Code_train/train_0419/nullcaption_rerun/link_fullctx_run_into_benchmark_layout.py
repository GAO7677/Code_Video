#!/usr/bin/env python3
"""Expose the ongoing full-context run under the benchmark's standard directory layout."""

from __future__ import annotations

import json
from pathlib import Path


BENCH_ROOT = Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V_nullcaption")
LIVE_ROOT = BENCH_ROOT / "tools" / "fullctx_runs" / "physicsiq_fullvideo"

OUTPUT_CONTEXT_DIR = BENCH_ROOT / "output" / "VACE_1_3B_V2V" / "context_fullctx_fullvideo"
RUNTIME_DIR = BENCH_ROOT / "tools" / "runtime" / "vace_v2v_fullctx_fullvideo"
META_DIR = BENCH_ROOT / "tools" / "meta" / "physicsiq_fullctx_fullvideo"
LOG_DIR = BENCH_ROOT / "tools" / "logs" / "physicsiq_fullctx_fullvideo"


def ensure_clean_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        path.unlink()


def ensure_symlink(target: Path, link_path: Path) -> None:
    ensure_clean_parent(link_path)
    link_path.symlink_to(target)


def clear_generated_links(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("*"):
        if path.is_symlink() or path.is_file():
            path.unlink()


def main() -> None:
    if not LIVE_ROOT.exists():
        raise SystemExit(f"Live root missing: {LIVE_ROOT}")

    output_generated_root = LIVE_ROOT / "generated"
    runtime_generated_root = LIVE_ROOT / "runtime"
    meta_root = LIVE_ROOT / "meta"
    log_root = LIVE_ROOT / "logs"

    clear_generated_links(OUTPUT_CONTEXT_DIR)
    RUNTIME_DIR.parent.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Runtime summary symlink target.
    ensure_symlink(runtime_generated_root, RUNTIME_DIR)

    # Meta and logs are linked case-by-case / file-by-file to stay inspectable.
    for meta_file in sorted(meta_root.rglob("*")):
        if meta_file.is_file():
            ensure_symlink(meta_file, META_DIR / meta_file.relative_to(meta_root))

    for log_file in sorted(log_root.glob("*")):
        if log_file.is_file():
            ensure_symlink(log_file, LOG_DIR / log_file.name)

    # Flatten generated caption/nullcaption outputs into the benchmark-style output directory.
    for variant_dir in sorted(output_generated_root.iterdir()):
        if not variant_dir.is_dir():
            continue
        for path in sorted(variant_dir.glob("*")):
            if path.suffix not in {".mp4", ".json"}:
                continue
            name = path.name
            stem = path.stem
            suffix = path.suffix
            if variant_dir.name.startswith("caption_"):
                link_name = f"{stem}__caption_fullctx_fullvideo{suffix}"
            elif variant_dir.name.startswith("nullcaption_"):
                link_name = f"{stem}__nullcaption_fullctx_fullvideo{suffix}"
            else:
                link_name = name
            ensure_symlink(path, OUTPUT_CONTEXT_DIR / link_name)

    manifest = {
        "live_root": str(LIVE_ROOT),
        "output_context_dir": str(OUTPUT_CONTEXT_DIR),
        "runtime_dir": str(RUNTIME_DIR),
        "meta_dir": str(META_DIR),
        "log_dir": str(LOG_DIR),
    }
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
