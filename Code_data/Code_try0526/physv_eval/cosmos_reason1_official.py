from __future__ import annotations

import random
import re
import sys
from pathlib import Path
from typing import Any

import yaml

from .case_inputs import EvalCase, coerce_eval_case
from .paths import COSMOS_REASON1_MODEL, COSMOS_REASON1_ROOT, VPHY_PYTHON


DEFAULT_PROMPT_PATH = (
    COSMOS_REASON1_ROOT.parent
    / "cosmos-cookbook"
    / "docs"
    / "recipes"
    / "post_training"
    / "reason1"
    / "physical-plausibility-check"
    / "assets"
    / "video_reward.yaml"
)


def parse_response(response: str) -> dict[str, Any] | None:
    match = re.search(r"<answer>\s*(\d+)\s*</answer>", response)
    if not match:
        return None
    try:
        answer_int = int(match.group(1))
    except ValueError:
        return {"answer": None, "raw": response}
    return {"answer": answer_int, "raw": response}


class OfficialCosmosReason1Runner:
    """Batch-friendly runner that keeps the official prompt and parse logic."""

    def __init__(
        self,
        *,
        model_path: Path = COSMOS_REASON1_MODEL,
        prompt_path: Path = DEFAULT_PROMPT_PATH,
        fps: int = 16,
        total_pixels: int = 8192 * 28 * 28,
        max_new_tokens: int = 256,
        temperature: float = 0.6,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.05,
        seed: int = 1,
        max_attempts: int = 10,
    ) -> None:
        self.model_path = str(model_path)
        self.prompt_path = Path(prompt_path)
        self.fps = fps
        self.total_pixels = total_pixels
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.seed = seed
        self.max_attempts = max(1, int(max_attempts))

        self._torch = None
        self._transformers = None
        self._process_vision_info = None
        self._processor = None
        self._model = None
        self._prompt_cfg = None
        self._model_device = None

    def _lazy_imports(self) -> None:
        if self._torch is not None:
            return
        import torch
        import transformers
        try:
            from qwen_vl_utils import process_vision_info
        except Exception:
            vphy_site_packages = VPHY_PYTHON.parent.parent / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
            if vphy_site_packages.is_dir():
                site_path = str(vphy_site_packages)
                if site_path not in sys.path:
                    sys.path.append(site_path)
            from qwen_vl_utils import process_vision_info

        self._torch = torch
        self._transformers = transformers
        self._process_vision_info = process_vision_info

    def _load_once(self) -> None:
        if self._model is not None:
            return
        self._lazy_imports()
        self._prompt_cfg = yaml.safe_load(self.prompt_path.read_text(encoding="utf-8"))
        self._processor = self._transformers.AutoProcessor.from_pretrained(self.model_path)
        model_cls = self._transformers.Qwen2_5_VLForConditionalGeneration
        model_kwargs: dict[str, Any] = {"dtype": "auto"}
        try:
            self._model = model_cls.from_pretrained(
                self.model_path,
                device_map="auto",
                **model_kwargs,
            )
            self._model_device = self._model.device
            return
        except ValueError as exc:
            if "requires `accelerate`" not in str(exc):
                raise

        self._model = model_cls.from_pretrained(
            self.model_path,
            **model_kwargs,
        )
        target_device = "cuda" if self._torch.cuda.is_available() else "cpu"
        self._model = self._model.to(target_device)
        self._model_device = self._torch.device(target_device)

    def _build_messages(self, video_path: Path) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": self._prompt_cfg["system_prompt"]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(video_path),
                        "fps": self.fps,
                        "total_pixels": self.total_pixels,
                    },
                    {"type": "text", "text": self._prompt_cfg["user_prompt"]},
                ],
            },
        ]

    def _generate_raw(self, inputs: Any, *, max_new_tokens: int) -> str:
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
        )
        trimmed = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, outputs, strict=False)
        ]
        return self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def score(self, video_path: Path) -> dict[str, Any]:
        self._load_once()
        self._torch.manual_seed(self.seed)
        random.seed(self.seed)

        messages = self._build_messages(video_path)
        prompt = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs, video_kwargs = self._process_vision_info(
            messages,
            return_video_kwargs=True,
        )
        for key, value in list(video_kwargs.items()):
            if isinstance(value, list) and len(value) == 1:
                video_kwargs[key] = value[0]

        inputs = self._processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        model_device = self._model_device if self._model_device is not None else self._model.device
        inputs = inputs.to(model_device)
        retry_max_new_tokens = max(self.max_new_tokens * 2, 512)
        attempt_token_limits = [self.max_new_tokens] + [retry_max_new_tokens] * max(0, self.max_attempts - 1)
        attempts: list[tuple[int, int, str, dict[str, Any] | None]] = []
        for attempt_index, max_new_tokens in enumerate(attempt_token_limits, start=1):
            raw = self._generate_raw(inputs, max_new_tokens=max_new_tokens)
            parsed = parse_response(raw)
            attempts.append((attempt_index, max_new_tokens, raw, parsed))
            if parsed is not None:
                break

        used_attempt_index, used_max_new_tokens, raw, parsed = attempts[-1]
        score = None if parsed is None else parsed.get("answer")
        return {
            "score": score,
            "raw": raw,
            "method": "official cosmos-reason1 prompt batched-equivalent",
            "model": self.model_path,
            "prompt_path": str(self.prompt_path),
            "fps": self.fps,
            "total_pixels": self.total_pixels,
            "max_new_tokens": used_max_new_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "seed": self.seed,
            "attempt_count": used_attempt_index,
            "max_attempts": self.max_attempts,
        }

    def score_case(self, case: EvalCase | Path | str | dict[str, Any]) -> dict[str, Any]:
        normalized = coerce_eval_case(case)
        return self.score(normalized.video_path)


def score_single_case(
    case: EvalCase | Path | str | dict[str, Any],
    *,
    runner: OfficialCosmosReason1Runner | None = None,
) -> dict[str, Any]:
    active_runner = runner or OfficialCosmosReason1Runner()
    return active_runner.score_case(case)
