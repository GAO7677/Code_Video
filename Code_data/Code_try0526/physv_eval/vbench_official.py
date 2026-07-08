from __future__ import annotations

import json
import os
import sys
import importlib
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from .case_inputs import EvalCase, coerce_eval_case
from .paths import AGENT_OUTPUT_ROOT, CKPT_ROOT, TORCH_HOME_ROOT, VBENCH_FULL_INFO, VBENCH_ROOT
from .records import stable_path_id


_CUSTOM_DIMENSIONS = {
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
}


def _normalize_official_result(
    dimension: str,
    official_payload: dict[str, Any],
) -> dict[str, Any]:
    bucket = official_payload.get(dimension)
    if not isinstance(bucket, list) or len(bucket) != 2:
        raise RuntimeError(f"Unexpected VBench result format for {dimension}: {type(bucket)}")

    aggregate = bucket[0]
    per_video = bucket[1]
    if not isinstance(per_video, list):
        per_video = []

    score = float(aggregate) if aggregate is not None else None
    normalized: dict[str, Any] = {
        "score": score,
        "dimension": dimension,
        "metric_direction": "higher_is_better",
        "official": True,
        "method": "vbench_official_custom_input",
        "supported_dimensions": sorted(_CUSTOM_DIMENSIONS),
        "raw_dimension_score": score,
        "raw_results": per_video,
    }
    return normalized


def _build_custom_prompt_entries(
    videos: Sequence[tuple[Path, str | None]],
    *,
    dimension: str,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for video_path, caption in videos:
        entry: dict[str, Any] = {
            "prompt_en": caption or video_path.stem,
            "dimension": [dimension],
            "video_list": [str(video_path)],
        }
        entries.append(entry)
    return entries


class OfficialVBenchRunner:
    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        full_json_dir: Path | None = None,
        output_root: Path | None = None,
        cache_dir: Path | None = None,
        device: str = "cuda",
        load_ckpt_from_local: bool = False,
        read_frame: bool = False,
        imaging_quality_preprocessing_mode: str = "longer",
    ) -> None:
        self.repo_root = (repo_root or VBENCH_ROOT).resolve()
        self.full_json_dir = (full_json_dir or VBENCH_FULL_INFO).resolve()
        self.output_root = (output_root or (AGENT_OUTPUT_ROOT / "vbench_single_case")).resolve()
        self.cache_dir = (cache_dir or (CKPT_ROOT / "vbench")).resolve()
        self.device = device
        self.load_ckpt_from_local = load_ckpt_from_local
        self.read_frame = read_frame
        self.imaging_quality_preprocessing_mode = imaging_quality_preprocessing_mode

    def _lazy_imports(self) -> Any:
        repo_root_str = str(self.repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)

        import torch
        from vbench import VBench

        return torch, VBench

    def _prepare_env(self) -> None:
        os.environ.setdefault("PYTHONNOUSERSITE", "1")
        os.environ.setdefault("VBENCH_CACHE_DIR", str(self.cache_dir))
        os.environ.setdefault("TORCH_HOME", str(TORCH_HOME_ROOT))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def _evaluate_from_full_info(
        self,
        *,
        full_info_entries: list[dict[str, Any]],
        dimension: str,
        output_path: Path | None = None,
        run_name: str | None = None,
    ) -> tuple[dict[str, Any], Path, Path, Path]:
        if dimension not in _CUSTOM_DIMENSIONS:
            raise ValueError(
                f"VBench single-case custom input only supports {sorted(_CUSTOM_DIMENSIONS)}, got {dimension!r}"
            )

        self._prepare_env()
        _, _ = self._lazy_imports()

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        resolved_run_name = run_name or f"{dimension}_batch_{stamp}"
        run_output = (output_path or (self.output_root / resolved_run_name / dimension)).resolve()
        run_output.mkdir(parents=True, exist_ok=True)

        full_info_json = run_output / f"{resolved_run_name}_full_info.json"
        full_info_json.write_text(json.dumps(full_info_entries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        repo_root_str = str(self.repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)
        import torch
        from vbench.utils import init_submodules

        dimension_module = importlib.import_module(f"vbench.{dimension}")
        evaluate_func = getattr(dimension_module, f"compute_{dimension}")
        submodules_dict = init_submodules(
            [dimension],
            local=self.load_ckpt_from_local,
            read_frame=self.read_frame,
        )
        results = evaluate_func(
            str(full_info_json),
            torch.device(self.device),
            submodules_dict[dimension],
            imaging_quality_preprocessing_mode=self.imaging_quality_preprocessing_mode,
        )
        payload = {dimension: list(results)}
        result_json = run_output / f"{resolved_run_name}_eval_results.json"
        result_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        normalized = _normalize_official_result(dimension, payload)
        return normalized, result_json, full_info_json, run_output

    def score(
        self,
        video_path: Path,
        *,
        dimension: str,
        caption: str | None = None,
        output_path: Path | None = None,
    ) -> dict[str, Any]:
        if dimension not in _CUSTOM_DIMENSIONS:
            raise ValueError(
                f"VBench single-case custom input only supports {sorted(_CUSTOM_DIMENSIONS)}, got {dimension!r}"
            )

        self._prepare_env()
        sample_id = stable_path_id(video_path)
        run_name = f"{dimension}_{sample_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        normalized, result_json, full_info_json, run_output = self._evaluate_from_full_info(
            full_info_entries=_build_custom_prompt_entries([(video_path, caption)], dimension=dimension),
            dimension=dimension,
            output_path=output_path or (self.output_root / sample_id / dimension),
            run_name=run_name,
        )
        normalized.update(
            {
                "video": str(video_path),
                "caption_used": caption,
                "result_json": str(result_json),
                "full_info_json": str(full_info_json),
                "output_path": str(run_output),
                "cache_dir": str(self.cache_dir),
                "device": self.device,
                "mode": "custom_full_info",
            }
        )
        return normalized

    def score_batch(
        self,
        cases: Sequence[EvalCase | Path | str | dict[str, Any]],
        *,
        dimension: str,
        output_path: Path | None = None,
        run_name: str | None = None,
    ) -> dict[str, Any]:
        normalized_cases = [coerce_eval_case(case) for case in cases]
        entries = _build_custom_prompt_entries(
            [(case.video_path, case.caption) for case in normalized_cases],
            dimension=dimension,
        )
        normalized, result_json, full_info_json, run_output = self._evaluate_from_full_info(
            full_info_entries=entries,
            dimension=dimension,
            output_path=output_path,
            run_name=run_name,
        )
        normalized.update(
            {
                "videos": [str(case.video_path) for case in normalized_cases],
                "captions_used": [case.caption for case in normalized_cases],
                "result_json": str(result_json),
                "full_info_json": str(full_info_json),
                "output_path": str(run_output),
                "cache_dir": str(self.cache_dir),
                "device": self.device,
                "mode": "custom_full_info",
            }
        )
        return normalized

    def score_case(
        self,
        case: EvalCase | Path | str | dict[str, Any],
        *,
        dimension: str,
        caption: str | None = None,
        output_path: Path | None = None,
    ) -> dict[str, Any]:
        normalized = coerce_eval_case(case, caption=caption)
        return self.score(
            normalized.video_path,
            dimension=dimension,
            caption=normalized.caption,
            output_path=output_path,
        )


def score_single_case(
    case: EvalCase | Path | str | dict[str, Any],
    *,
    dimension: str,
    caption: str | None = None,
    output_path: Path | None = None,
    runner: OfficialVBenchRunner | None = None,
) -> dict[str, Any]:
    active_runner = runner or OfficialVBenchRunner()
    return active_runner.score_case(
        case,
        dimension=dimension,
        caption=caption,
        output_path=output_path,
    )
