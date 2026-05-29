from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

import yaml

from .paths import COSMOS_REASON1_MODEL, COSMOS_REASON1_ROOT


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

        self._torch = None
        self._transformers = None
        self._process_vision_info = None
        self._processor = None
        self._model = None
        self._prompt_cfg = None

    def _lazy_imports(self) -> None:
        if self._torch is not None:
            return
        import torch
        import transformers
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
        self._model = self._transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map="auto",
        )

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
        inputs = inputs.to(self._model.device)
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
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
        raw = self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        parsed = parse_response(raw)
        score = None if parsed is None else parsed.get("answer")
        return {
            "score": score,
            "raw": raw,
            "method": "official cosmos-reason1 prompt batched-equivalent",
            "model": self.model_path,
            "prompt_path": str(self.prompt_path),
            "fps": self.fps,
            "total_pixels": self.total_pixels,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "seed": self.seed,
        }
