from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PathsConfig:
    vbench_repo_root: str
    work_root: str
    vbench_cache_dir: str
    vbench2_cache_dir: str


@dataclass
class RuntimeConfig:
    device: str = "cuda"
    load_ckpt_from_local: bool = False
    read_frame: bool = False
    imaging_quality_preprocessing_mode: str = "longer"
    use_symlink: bool = True
    cleanup_staging: bool = False


@dataclass
class ContinuationConfig:
    max_video_frames: int = -1
    generated_start_frame: int = 0
    gt_start_frame: int = 0
    enable_lpips: bool = False
    lpips_net: str = "alex"


@dataclass
class BenchConfig:
    paths: PathsConfig
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    continuation: ContinuationConfig = field(default_factory=ContinuationConfig)
    weights_paths: dict[str, str] = field(default_factory=dict)
    dataset_paths: dict[str, str] = field(default_factory=dict)
    extra_env: dict[str, str] = field(default_factory=dict)


def _expand_path(path: str | None) -> str | None:
    if path is None or path == "":
        return path
    return str(Path(path).expanduser().resolve())


def _normalize_map(values: dict[str, Any] | None) -> dict[str, str]:
    values = values or {}
    normalized: dict[str, str] = {}
    for key, value in values.items():
        if value in (None, ""):
            continue
        normalized[key] = _expand_path(str(value))
    return normalized


def load_config(path: str) -> BenchConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    paths_raw = raw.get("paths", {})
    runtime_raw = raw.get("runtime", {})
    continuation_raw = raw.get("continuation", {})

    repo_root = _expand_path(paths_raw.get("vbench_repo_root", "../VBench-master"))
    work_root = _expand_path(paths_raw.get("work_root", str(config_path.parent.parent / "workdirs")))
    vbench_cache = _expand_path(paths_raw.get("vbench_cache_dir", str(Path(work_root) / "cache" / "vbench")))
    vbench2_cache = _expand_path(paths_raw.get("vbench2_cache_dir", str(Path(work_root) / "cache" / "vbench2")))

    cfg = BenchConfig(
        paths=PathsConfig(
            vbench_repo_root=repo_root,
            work_root=work_root,
            vbench_cache_dir=vbench_cache,
            vbench2_cache_dir=vbench2_cache,
        ),
        runtime=RuntimeConfig(**runtime_raw),
        continuation=ContinuationConfig(**continuation_raw),
        weights_paths=_normalize_map(raw.get("weights_paths")),
        dataset_paths=_normalize_map(raw.get("dataset_paths")),
        extra_env={k: str(v) for k, v in (raw.get("extra_env") or {}).items()},
    )
    return cfg

