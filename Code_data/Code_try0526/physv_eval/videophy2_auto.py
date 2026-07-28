from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

from .datasets import GROUP_SPECS, iter_group_jsons
from .official_pdi import resolve_text_query as resolve_pdi_text_query
from .paths import VIDEOPHY_ROOT, VIDEOPHY2_CKPT
from .paths import DATA_ROOT
from .case_inputs import EvalCase, coerce_eval_case
from .records import (
    load_payload,
    resolve_video_path,
    save_payload,
    set_videophy2_auto,
)


SCORE_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official VideoPhy-2 AutoEval.")
    parser.add_argument("--task", default="pc", choices=["sa", "pc", "rule"])
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--caption", default=None)
    parser.add_argument("--rule", default=None)
    parser.add_argument("--context-video", type=Path, default=None)
    parser.add_argument("--input-csv", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--json-path", action="append", type=Path, default=None)
    parser.add_argument("--groups", nargs="+", choices=list(GROUP_SPECS), default=None)
    parser.add_argument("--checkpoint", type=Path, default=VIDEOPHY2_CKPT)
    parser.add_argument("--repo-root", type=Path, default=VIDEOPHY_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def _resolve_text_query(payload: dict[str, Any], keys: list[str]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _resolve_group_c_sim_original(sample_key: str) -> Path:
    for candidate in [
        DATA_ROOT / "videos" / "ball_block" / f"{sample_key}.json",
        DATA_ROOT / "videos" / "jepa_sensitivity" / f"{sample_key}.json",
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No C-group sim original json found for {sample_key!r}")


def resolve_videophy2_sa_query(video_path: Path, payload: dict[str, Any]) -> str:
    direct = _resolve_text_query(payload, ["caption", "text_prompt", "prompt", "target_object"])
    if direct is not None:
        return direct

    stem = video_path.stem
    if "shuffle_test" in video_path.parts:
        if stem.startswith("gt_"):
            return resolve_pdi_text_query(video_path, payload)
        if stem.startswith("sim_"):
            sample_key = stem[len("sim_") :]
            for suffix in ("_original", "_shuffled"):
                if sample_key.endswith(suffix):
                    sample_key = sample_key[: -len(suffix)]
            source_payload = load_payload(_resolve_group_c_sim_original(sample_key))
            direct = _resolve_text_query(source_payload, ["caption", "text_prompt", "prompt", "target_object"])
            if direct is not None:
                return direct

    fallback = _resolve_text_query(payload, ["description", "scenario", "experiment"])
    if fallback is not None:
        return fallback
    return "ball"


class VideoPhy2Runner:
    def __init__(
        self,
        *,
        checkpoint: Path = VIDEOPHY2_CKPT,
        repo_root: Path = VIDEOPHY_ROOT,
        device: str = "cuda",
        dtype: str = "bfloat16",
        num_frames: int = 32,
    ) -> None:
        self.checkpoint = checkpoint
        self.repo_root = repo_root
        self.device = device
        self.dtype = dtype
        self.num_frames = num_frames
        self._torch = None
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._prompts = None

    def _torch_dtype(self) -> Any:
        torch = self._torch
        if self.dtype == "bfloat16":
            return torch.bfloat16
        if self.dtype == "float16":
            return torch.float16
        return torch.float32

    def _ensure_checkpoint_complete(self) -> None:
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"VideoPhy-2 checkpoint directory not found: {self.checkpoint}")
        index_path = self.checkpoint / "pytorch_model.bin.index.json"
        if index_path.exists():
            index_payload = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index_payload.get("weight_map", {})
            missing = sorted({name for name in weight_map.values() if not (self.checkpoint / name).exists()})
            if missing:
                preview = ", ".join(missing[:6])
                raise FileNotFoundError(
                    f"VideoPhy-2 checkpoint is incomplete under {self.checkpoint}. "
                    f"Missing {len(missing)} shard(s), e.g. {preview}"
                )

    def _lazy_imports(self) -> None:
        if self._torch is not None:
            return
        self._ensure_checkpoint_complete()
        videophy2_root = self.repo_root / "VIDEOPHY2"
        if not videophy2_root.exists():
            raise FileNotFoundError(f"VIDEOPHY2 source directory not found: {videophy2_root}")
        if str(videophy2_root) not in sys.path:
            sys.path.insert(0, str(videophy2_root))

        import torch
        import transformers.modeling_utils as modeling_utils
        import transformers.utils.import_utils as import_utils
        from transformers import LlamaTokenizer
        from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
        from mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
        from template import PROMPT_PHYSICS, PROMPT_RULE, PROMPT_SA

        # Local trusted checkpoint under /data/gaoya/ckpt may still ship .bin shards.
        # Newer transformers block torch<2.6 torch.load() by default; disable that
        # guard only for this in-repo trusted evaluation path.
        def _allow_local_torch_load(*args, **kwargs):
            return None

        if hasattr(import_utils, "check_torch_load_is_safe"):
            import_utils.check_torch_load_is_safe = _allow_local_torch_load
        if hasattr(modeling_utils, "check_torch_load_is_safe"):
            modeling_utils.check_torch_load_is_safe = _allow_local_torch_load

        self._torch = torch
        self._llama_tokenizer = LlamaTokenizer
        self._model_cls = MplugOwlForConditionalGeneration
        self._image_processor_cls = MplugOwlImageProcessor
        self._processor_cls = MplugOwlProcessor
        self._prompts = {
            "sa": PROMPT_SA,
            "pc": PROMPT_PHYSICS,
            "rule": PROMPT_RULE,
        }

    def _load_model(self) -> None:
        if self._model is not None:
            return
        self._lazy_imports()
        tokenizer = self._llama_tokenizer.from_pretrained(self.checkpoint)
        image_processor = self._image_processor_cls.from_pretrained(self.checkpoint)
        processor = self._processor_cls(image_processor, tokenizer)
        model = self._model_cls.from_pretrained(
            self.checkpoint,
            torch_dtype=self._torch_dtype(),
        )
        model = model.to(self.device).to(self._torch_dtype())
        model.eval()
        self._tokenizer = tokenizer
        self._processor = processor
        self._model = model

    def _prompt_for(self, task: str, *, caption: str | None, rule: str | None) -> str:
        if task == "sa":
            if not caption:
                raise ValueError("VideoPhy-2 task=sa requires a caption")
            return self._prompts["sa"].format(caption=caption)
        if task == "pc":
            return self._prompts["pc"]
        if task == "rule":
            if not rule:
                raise ValueError("VideoPhy-2 task=rule requires a rule")
            return self._prompts["rule"].format(rule=rule)
        raise ValueError(f"Unsupported task: {task}")

    def _parse_score(self, raw_output: str) -> int:
        output = raw_output.strip()
        answer_span = output.rsplit("AI:", 1)[-1].strip() or output
        lowered = answer_span.lower()

        for word, value in SCORE_WORDS.items():
            if re.search(rf"\b{word}\b", lowered):
                return value
        digit_match = re.search(r"\b([0-5])\b", lowered)
        if digit_match:
            return int(digit_match.group(1))

        lowered_full = output.lower()
        for word, value in SCORE_WORDS.items():
            if re.search(rf"\b{word}\b", lowered_full):
                return value
        digit_match = re.search(r"\b([0-5])\b", lowered_full)
        if digit_match:
            return int(digit_match.group(1))
        return 0

    def score_video(
        self,
        video_path: Path,
        *,
        task: str = "pc",
        caption: str | None = None,
        rule: str | None = None,
    ) -> dict[str, Any]:
        self._load_model()
        prompt = self._prompt_for(task, caption=caption, rule=rule)
        inputs = self._processor(
            text=[prompt],
            videos=[str(video_path)],
            num_frames=self.num_frames,
            return_tensors="pt",
        )
        inputs = {key: value.to(self._torch_dtype()) if getattr(value, "dtype", None) == self._torch.float32 else value for key, value in inputs.items()}
        inputs = {key: value.to(self._model.device) for key, value in inputs.items()}
        input_length = int(inputs["input_ids"].shape[-1])
        generate_kwargs = {
            "do_sample": False,
            "top_k": 1,
            "temperature": 0.001,
        }
        if input_length < 256:
            generate_kwargs["max_length"] = 256
            generation_limit = {"mode": "official_max_length", "value": 256}
        else:
            # Official max_length=256 cannot generate when the prompt is already
            # longer than 256 tokens. Preserve the full prompt and bound only the
            # short score response for this compatibility fallback.
            generate_kwargs["max_new_tokens"] = 8
            generation_limit = {"mode": "long_prompt_max_new_tokens", "value": 8}
        with self._torch.inference_mode():
            generated = self._model.generate(**inputs, **generate_kwargs)
        raw_output = self._tokenizer.decode(generated.tolist()[0], skip_special_tokens=True)
        return {
            "task": task,
            "score": self._parse_score(raw_output),
            "raw_output": raw_output,
            "num_frames": self.num_frames,
            "checkpoint": str(self.checkpoint),
            "input_token_length": input_length,
            "generation_limit": generation_limit,
        }

    def score_case(
        self,
        case: EvalCase | Path | str | dict[str, Any],
        *,
        task: str = "pc",
        caption: str | None = None,
        rule: str | None = None,
    ) -> dict[str, Any]:
        normalized = coerce_eval_case(case, caption=caption, rule=rule)
        resolved_caption = normalized.caption
        resolved_rule = normalized.rule
        if task == "sa" and not resolved_caption:
            payload = normalized.metadata or {}
            resolved_caption = resolve_videophy2_sa_query(normalized.video_path, payload)
        return self.score_video(
            normalized.video_path,
            task=task,
            caption=resolved_caption,
            rule=resolved_rule,
        )


def score_single_case(
    case: EvalCase | Path | str | dict[str, Any],
    *,
    task: str = "pc",
    caption: str | None = None,
    rule: str | None = None,
    runner: VideoPhy2Runner | None = None,
) -> dict[str, Any]:
    active_runner = runner or VideoPhy2Runner()
    return active_runner.score_case(case, task=task, caption=caption, rule=rule)


def _iter_selected_jsons(groups: list[str] | None, json_paths: list[Path] | None) -> list[Path]:
    selected: list[Path] = []
    if groups:
        for group_id in groups:
            selected.extend(iter_group_jsons(group_id))
    if json_paths:
        selected.extend(json_paths)
    seen: set[Path] = set()
    deduped: list[Path] = []
    for path in selected:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(path)
    return deduped


def _run_single_video(args: argparse.Namespace, runner: VideoPhy2Runner) -> None:
    if args.video is None:
        return
    result = runner.score_video(args.video, task=args.task, caption=args.caption, rule=args.rule)
    print(json.dumps({"video": str(args.video), **result}, ensure_ascii=False, indent=2))


def _run_single_case(args: argparse.Namespace, runner: VideoPhy2Runner) -> None:
    if args.video is None:
        return
    case = EvalCase(
        video_path=args.video,
        caption=args.caption,
        rule=args.rule,
        context_video_path=args.context_video,
    )
    result = runner.score_case(case, task=args.task)
    print(json.dumps({"video": str(args.video), **result}, ensure_ascii=False, indent=2))


def _run_csv(args: argparse.Namespace, runner: VideoPhy2Runner) -> None:
    if args.input_csv is None:
        return
    output_csv = args.output_csv or args.input_csv.with_name(f"{args.input_csv.stem}.videophy2_{args.task}.csv")
    with args.input_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    for extra in ["score", "raw_output"]:
        if extra not in fieldnames:
            fieldnames.append(extra)

    for row in rows:
        video = row.get("videopath")
        if not video:
            raise ValueError("CSV must contain a 'videopath' column")
        result = runner.score_video(
            Path(video),
            task=args.task,
            caption=row.get("caption"),
            rule=row.get("rule"),
        )
        row["score"] = result["score"]
        row["raw_output"] = result["raw_output"]

    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(output_csv)


def _run_json_mode(args: argparse.Namespace, runner: VideoPhy2Runner) -> None:
    selected = _iter_selected_jsons(args.groups, args.json_path)
    if not selected:
        return

    for index, json_path in enumerate(selected, start=1):
        payload = load_payload(json_path)
        video_path = resolve_video_path(json_path, payload)
        result_bucket = payload.get("metric_results", {}).get("videophy2_auto", {})
        if args.task == "pc" and not args.refresh and result_bucket.get("pc_score") is not None:
            print(f"[{index}/{len(selected)}] skip {json_path.name}: existing pc_score", flush=True)
            continue
        if args.task == "sa" and not args.refresh and result_bucket.get("sa_score") is not None:
            print(f"[{index}/{len(selected)}] skip {json_path.name}: existing sa_score", flush=True)
            continue
        caption = resolve_videophy2_sa_query(video_path, payload) if args.task == "sa" else None
        result = runner.score_video(video_path, task=args.task, caption=caption)
        set_videophy2_auto(payload, result)
        save_payload(json_path, payload)
        print(f"[{index}/{len(selected)}] {json_path.name}: {args.task}={result['score']}", flush=True)


def main() -> None:
    args = parse_args()
    runner = VideoPhy2Runner(
        checkpoint=args.checkpoint,
        repo_root=args.repo_root,
        device=args.device,
        dtype=args.dtype,
        num_frames=args.num_frames,
    )
    if args.video is not None:
        _run_single_case(args, runner)
        return
    if args.input_csv is not None:
        _run_csv(args, runner)
        return
    _run_json_mode(args, runner)


if __name__ == "__main__":
    main()
