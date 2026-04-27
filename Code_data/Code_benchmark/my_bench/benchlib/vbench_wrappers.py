from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import torch

from .config import BenchConfig
from .manifest import BenchSample
from .staging import stage_custom_vbench_dataset


SHORT_CORE_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "temporal_flickering",
    "dynamic_degree",
    "imaging_quality",
    "aesthetic_quality",
    "overall_consistency",
    "temporal_style",
]

I2V_CORE_DIMENSIONS = [
    "i2v_subject",
    "i2v_background",
    "camera_motion",
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "imaging_quality",
    "aesthetic_quality",
]

LONG_CORE_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "temporal_flickering",
    "dynamic_degree",
    "imaging_quality",
    "aesthetic_quality",
]


def _bootstrap_vbench_repo(config: BenchConfig) -> None:
    repo_root = Path(config.paths.vbench_repo_root)
    vbench2_root = repo_root / "VBench-2.0"
    for candidate in [repo_root, vbench2_root]:
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

    os.environ["VBENCH_CACHE_DIR"] = config.paths.vbench_cache_dir
    os.environ["VBENCH2_CACHE_DIR"] = config.paths.vbench2_cache_dir
    for key, value in config.extra_env.items():
        os.environ[key] = value


def _device_from_config(config: BenchConfig) -> torch.device:
    want = config.runtime.device
    if want.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(want)


def _write_metadata(output_dir: str, payload: dict[str, Any]) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _cleanup(path: str, enabled: bool) -> None:
    if enabled and Path(path).exists():
        shutil.rmtree(path)


def run_vbench_short(
    config: BenchConfig,
    samples: list[BenchSample],
    output_dir: str,
    dimensions: list[str] | None = None,
    run_name: str = "vbench_short",
) -> str:
    _bootstrap_vbench_repo(config)
    from vbench import VBench

    output_dir = str(Path(output_dir).expanduser().resolve())
    staging = stage_custom_vbench_dataset(
        samples=samples,
        staging_root=str(Path(output_dir) / "staging"),
        use_symlink=config.runtime.use_symlink,
        with_images=False,
    )
    dimensions = dimensions or SHORT_CORE_DIMENSIONS

    bench = VBench(
        device=_device_from_config(config),
        full_info_dir=str(Path(config.paths.vbench_repo_root) / "vbench" / "VBench_full_info.json"),
        output_path=output_dir,
    )
    bench.evaluate(
        videos_path=staging.video_dir,
        name=run_name,
        prompt_list=staging.prompt_map,
        dimension_list=dimensions,
        local=config.runtime.load_ckpt_from_local,
        read_frame=config.runtime.read_frame,
        mode="custom_input",
        imaging_quality_preprocessing_mode=config.runtime.imaging_quality_preprocessing_mode,
    )
    _write_metadata(
        output_dir,
        {
            "suite": "vbench_short",
            "dimensions": dimensions,
            "samples": [sample.sample_id for sample in samples],
            "staging_dir": staging.root_dir,
        },
    )
    _cleanup(staging.root_dir, config.runtime.cleanup_staging)
    return str(Path(output_dir) / f"{run_name}_eval_results.json")


def run_vbench_i2v(
    config: BenchConfig,
    samples: list[BenchSample],
    output_dir: str,
    dimensions: list[str] | None = None,
    run_name: str = "vbench_i2v",
    resolution: str = "1-1",
) -> str:
    _bootstrap_vbench_repo(config)
    from vbench2_beta_i2v import VBenchI2V

    output_dir = str(Path(output_dir).expanduser().resolve())
    staging = stage_custom_vbench_dataset(
        samples=samples,
        staging_root=str(Path(output_dir) / "staging"),
        use_symlink=config.runtime.use_symlink,
        with_images=True,
    )
    dimensions = dimensions or I2V_CORE_DIMENSIONS

    bench = VBenchI2V(
        device=_device_from_config(config),
        full_info_dir=str(Path(config.paths.vbench_repo_root) / "vbench2_beta_i2v" / "vbench2_i2v_full_info.json"),
        output_path=output_dir,
    )
    bench.evaluate(
        videos_path=staging.video_dir,
        name=run_name,
        dimension_list=dimensions,
        custom_image_folder=staging.image_dir,
        mode="custom_input",
        local=config.runtime.load_ckpt_from_local,
        read_frame=config.runtime.read_frame,
        resolution=resolution,
        imaging_quality_preprocessing_mode=config.runtime.imaging_quality_preprocessing_mode,
    )
    _write_metadata(
        output_dir,
        {
            "suite": "vbench_i2v",
            "dimensions": dimensions,
            "samples": [sample.sample_id for sample in samples],
            "staging_dir": staging.root_dir,
            "resolution": resolution,
            "image_source": "last_context_frame_or_image_path",
        },
    )
    _cleanup(staging.root_dir, config.runtime.cleanup_staging)
    return str(Path(output_dir) / f"{run_name}_eval_results.json")


def run_vbench_long(
    config: BenchConfig,
    samples: list[BenchSample],
    output_dir: str,
    dimensions: list[str] | None = None,
    run_name: str = "vbench_long",
) -> str:
    _bootstrap_vbench_repo(config)
    from vbench2_beta_long import VBenchLong

    output_dir = str(Path(output_dir).expanduser().resolve())
    staging = stage_custom_vbench_dataset(
        samples=samples,
        staging_root=str(Path(output_dir) / "staging"),
        use_symlink=config.runtime.use_symlink,
        with_images=False,
    )
    dimensions = dimensions or LONG_CORE_DIMENSIONS
    long_root = Path(config.paths.vbench_repo_root) / "vbench2_beta_long"

    bench = VBenchLong(
        device=_device_from_config(config),
        full_info_dir=str(long_root / "VBench_full_info.json"),
        output_path=output_dir,
    )
    bench.evaluate(
        videos_path=staging.video_dir,
        name=run_name,
        prompt_list=staging.prompt_map,
        dimension_list=dimensions,
        local=config.runtime.load_ckpt_from_local,
        read_frame=config.runtime.read_frame,
        mode="long_custom_input",
        imaging_quality_preprocessing_mode=config.runtime.imaging_quality_preprocessing_mode,
        use_semantic_splitting=False,
        bg_clip2clip_feat_extractor="clip",
        sb_clip2clip_feat_extractor="dino",
        w_inclip=1.0,
        w_clip2clip=0.0,
        slow_fast_eval_config=str(long_root / "configs" / "slow_fast_params.yaml"),
        clip_length_config="clip_length_mix.yaml",
        dev_flag=False,
        sb_mapping_file_path=str(long_root / "configs" / "subject_mapping_table.yaml"),
        bg_mapping_file_path=str(long_root / "configs" / "background_mapping_table.yaml"),
        num_of_samples_per_prompt=1,
        static_filter_flag=False,
    )
    _write_metadata(
        output_dir,
        {
            "suite": "vbench_long",
            "dimensions": dimensions,
            "samples": [sample.sample_id for sample in samples],
            "staging_dir": staging.root_dir,
            "mode": "long_custom_input",
        },
    )
    _cleanup(staging.root_dir, config.runtime.cleanup_staging)
    return str(Path(output_dir) / f"{run_name}_eval_results.json")

