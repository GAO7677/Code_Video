from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .case_inputs import EvalCase, coerce_eval_case
from .paths import AGENT_OUTPUT_ROOT, CKPT_ROOT, TORCH_HOME_ROOT, VBENCH2_FULL_INFO, VBENCH2_ROOT
from .records import stable_path_id


_CUSTOM_DIMENSIONS = {
    "Human_Anatomy",
    "Human_Identity",
    "Human_Clothes",
    "Diversity",
    "Multi-View_Consistency",
}


def _normalize_official_result(
    dimension: str,
    official_payload: dict[str, Any],
) -> dict[str, Any]:
    bucket = official_payload.get(dimension)
    if not isinstance(bucket, list) or len(bucket) != 2:
        raise RuntimeError(f"Unexpected VBench-2.0 result format for {dimension}: {type(bucket)}")

    aggregate = bucket[0]
    per_video = bucket[1]
    if not isinstance(per_video, list):
        per_video = []

    score = float(aggregate) if aggregate is not None else None
    return {
        "score": score,
        "dimension": dimension,
        "metric_direction": "higher_is_better",
        "official": True,
        "method": "vbench2_official_custom_input",
        "supported_dimensions": sorted(_CUSTOM_DIMENSIONS),
        "raw_dimension_score": score,
        "raw_results": per_video,
    }


class OfficialVBench2Runner:
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
    ) -> None:
        self.repo_root = (repo_root or VBENCH2_ROOT).resolve()
        self.full_json_dir = (full_json_dir or VBENCH2_FULL_INFO).resolve()
        self.output_root = (output_root or (AGENT_OUTPUT_ROOT / "vbench2_single_case")).resolve()
        self.cache_dir = (cache_dir or (CKPT_ROOT / "vbench2")).resolve()
        self.device = device
        self.load_ckpt_from_local = load_ckpt_from_local
        self.read_frame = read_frame

    def _lazy_imports(self) -> Any:
        repo_root_str = str(self.repo_root)
        if repo_root_str not in sys.path:
            sys.path.insert(0, repo_root_str)

        import torch
        from vbench2 import VBench2

        return torch, VBench2

    def _prepare_env(self) -> None:
        os.environ.setdefault("PYTHONNOUSERSITE", "1")
        os.environ.setdefault("VBENCH2_CACHE_DIR", str(self.cache_dir))
        os.environ.setdefault("TORCH_HOME", str(TORCH_HOME_ROOT))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)

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
                f"VBench2 single-case custom input only supports {sorted(_CUSTOM_DIMENSIONS)}, got {dimension!r}"
            )

        self._prepare_env()
        torch, VBench2 = self._lazy_imports()

        sample_id = stable_path_id(video_path)
        run_name = f"{dimension}_{sample_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_output = (output_path or (self.output_root / sample_id / dimension)).resolve()
        run_output.mkdir(parents=True, exist_ok=True)

        model = VBench2(torch.device(self.device), str(self.full_json_dir), str(run_output))
        prompt_list = [caption] if caption else []
        input_path = video_path
        if dimension == "Diversity":
            input_path = video_path if video_path.is_dir() else video_path.parent
        model.evaluate(
            videos_path=str(input_path),
            name=run_name,
            prompt_list=prompt_list,
            dimension_list=[dimension],
            local=self.load_ckpt_from_local,
            read_frame=self.read_frame,
            mode="custom_input",
        )

        result_json = run_output / f"{run_name}_eval_results.json"
        if not result_json.is_file():
            raise FileNotFoundError(f"Expected VBench-2.0 result file not found: {result_json}")
        payload = json.loads(result_json.read_text(encoding="utf-8"))

        normalized = _normalize_official_result(dimension, payload)
        normalized.update(
            {
                "video": str(video_path),
                "caption_used": caption,
                "result_json": str(result_json),
                "full_info_json": str(run_output / f"{run_name}_full_info.json"),
                "output_path": str(run_output),
                "cache_dir": str(self.cache_dir),
                "device": self.device,
                "mode": "custom_input",
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
    runner: OfficialVBench2Runner | None = None,
) -> dict[str, Any]:
    active_runner = runner or OfficialVBench2Runner()
    return active_runner.score_case(
        case,
        dimension=dimension,
        caption=caption,
        output_path=output_path,
    )
