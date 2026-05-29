from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

from .paths import PHYJUDGE_ADAPTER, PHYJUDGE_INFER


GENERAL_METRICS = ("SA", "PTV", "persistence")

MECHANICS_LAWS = (
    "gravity",
    "inertia",
    "momentum",
    "impenetrability",
    "collision",
    "material",
)

FLUID_LAWS = (
    "buoyancy",
    "displacement",
    "flow_dynamics",
    "boundary_interaction",
    "fluid_continuity",
)

OPTICS_LAWS = ("reflection", "shadow")


def _mean_or_none(values: list[int]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


class OfficialPhyGroundRunner:
    """Batch-friendly runner that reuses the released phyjudge infer.py logic.

    The goal is numerical parity with the official single-case script while
    avoiding repeated model reloads for batch evaluation.
    """

    def __init__(
        self,
        *,
        adapter_dir: Path = PHYJUDGE_ADAPTER,
        infer_script: Path = PHYJUDGE_INFER,
        dtype: str = "bfloat16",
        device_map: str = "auto",
        fps: float = 2.0,
        max_pixels: int = 360 * 640,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        cuda_visible_devices: str | None = None,
    ) -> None:
        self.adapter_dir = Path(adapter_dir)
        self.infer_script = Path(infer_script)
        self.dtype = dtype
        self.device_map = device_map
        self.fps = fps
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.cuda_visible_devices = cuda_visible_devices

        self._module: Any | None = None
        self._processor = None
        self._model = None
        self._adapter_dir = None
        self._prompt_cfg = None
        self._device = None

    def _load_module(self) -> Any:
        if self._module is not None:
            return self._module
        if self.cuda_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.cuda_visible_devices)
        os.environ.setdefault("PYTHONNOUSERSITE", "1")
        spec = importlib.util.spec_from_file_location("phyjudge_infer_local", self.infer_script)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load spec from {self.infer_script}")
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("phyjudge_infer_local", module)
        spec.loader.exec_module(module)
        self._module = module
        return module

    def _load_once(self) -> None:
        if self._model is not None:
            return
        module = self._load_module()
        dtype = module.dtype_from_name(self.dtype)
        processor, model, adapter_dir = module.load_model(
            str(self.adapter_dir),
            dtype=dtype,
            device_map=self.device_map,
        )
        prompt_cfg = module.load_yaml(adapter_dir / "subq+human.yaml")
        self._processor = processor
        self._model = model
        self._adapter_dir = adapter_dir
        self._prompt_cfg = prompt_cfg
        self._device = next(model.parameters()).device

    def score_one(
        self,
        video_path: Path,
        caption: str,
        *,
        metric: str | None = None,
        law: str | None = None,
        criteria: str | None = None,
    ) -> dict[str, Any]:
        self._load_once()
        module = self._module
        system_prompt, user_prompt, score_key = module.build_prompt(
            self._prompt_cfg,
            caption,
            metric=metric,
            law=law,
            criteria=criteria,
        )
        messages = module.build_messages(system_prompt, user_prompt, Path(video_path))
        inputs = module.prepare_inputs(
            self._processor,
            messages,
            self._device,
            fps=self.fps,
            max_pixels=self.max_pixels,
        )
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "temperature": self.temperature if self.temperature > 0 else None,
        }
        generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
        with module.torch.inference_mode():
            generated_ids = self._model.generate(**inputs, **generation_kwargs)
        raw = module.decode_generated(self._processor, inputs, generated_ids)
        score = module.parse_score(raw, score_key)
        return {
            "key": score_key,
            "score": score,
            "raw": raw,
            "metric": metric,
            "law": law,
            "caption": caption,
            "video": str(video_path),
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "fps": self.fps,
            "max_pixels": self.max_pixels,
        }

    def score_bundle(
        self,
        video_path: Path,
        caption: str,
        *,
        metrics: list[str] | None = None,
        laws: list[str] | None = None,
        criteria_overrides: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        metric_names = metrics or list(GENERAL_METRICS)
        law_names = laws or []
        criteria_overrides = criteria_overrides or {}

        general_scores: dict[str, int | None] = {}
        general_raw: dict[str, str] = {}
        for metric_name in metric_names:
            result = self.score_one(video_path, caption, metric=metric_name)
            general_scores[metric_name] = result["score"]
            general_raw[metric_name] = result["raw"]

        law_scores: dict[str, int | None] = {}
        law_raw: dict[str, str] = {}
        for law_name in law_names:
            result = self.score_one(
                video_path,
                caption,
                law=law_name,
                criteria=criteria_overrides.get(law_name),
            )
            law_scores[law_name] = result["score"]
            law_raw[law_name] = result["raw"]

        general_values = [value for value in general_scores.values() if isinstance(value, int)]
        law_values = [value for value in law_scores.values() if isinstance(value, int)]
        return {
            "general": general_scores,
            "general_avg": _mean_or_none(general_values),
            "physical_laws": law_scores,
            "physical_avg": _mean_or_none(law_values),
            "coverage": (len(law_values) / len(law_names)) if law_names else None,
            "raw": {
                "general": general_raw,
                "physical_laws": law_raw,
            },
            "method": "official phyjudge infer.py batched-equivalent",
            "adapter_dir": str(self.adapter_dir),
            "infer_script": str(self.infer_script),
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "fps": self.fps,
            "max_pixels": self.max_pixels,
        }


def resolve_phyground_caption(payload: dict[str, Any], fallback: str) -> str:
    for key in ("caption", "description", "prompt", "clip_name", "scenario", "name"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


def infer_phyground_laws(payload: dict[str, Any], *, group_id: str, video_stem: str) -> list[str]:
    caption = json.dumps(payload, ensure_ascii=False).lower()
    if group_id in {"B1", "B2", "B3"}:
        laws = list(MECHANICS_LAWS)
        if "nomiss" in video_stem:
            laws = [law for law in laws if law != "collision"]
        return laws
    if group_id == "C" and video_stem.startswith("sim_"):
        laws = list(MECHANICS_LAWS)
        if "nomiss" in video_stem:
            laws = [law for law in laws if law != "collision"]
        return laws
    if any(token in caption for token in ("water", "liquid", "pour", "fluid")):
        return list(FLUID_LAWS)
    if any(token in caption for token in ("shadow", "reflection", "mirror")):
        return list(OPTICS_LAWS)
    return []
