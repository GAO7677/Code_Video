from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


_DEFAULT_VIDEO_KEYS = (
    "video",
    "video_path",
    "output_video",
    "paths.output_video_path",
)
_DEFAULT_CONTEXT_KEYS = (
    "context_video",
    "context_video_path",
    "paths.context_video_path",
)
_DEFAULT_CAPTION_KEYS = (
    "caption",
    "text_prompt",
    "prompt",
    "description",
    "scenario",
    "experiment",
    "target_object",
    "clip_name",
    "name",
)
_DEFAULT_RULE_KEYS = ("rule", "physical_law", "law")


@dataclass(frozen=True)
class EvalCase:
    video_path: Path
    caption: str | None = None
    rule: str | None = None
    context_video_path: Path | None = None
    metadata: dict[str, Any] | None = None


def _get_nested(payload: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def first_text(payload: Mapping[str, Any], keys: Sequence[str], *, default: str | None = None) -> str | None:
    for key in keys:
        value = _get_nested(payload, key) if "." in key else payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return default


def first_path(
    payload: Mapping[str, Any],
    keys: Sequence[str],
    *,
    base_dir: Path | None = None,
) -> Path | None:
    for key in keys:
        value = _get_nested(payload, key) if "." in key else payload.get(key)
        if value in (None, ""):
            continue
        path = Path(value)
        if path.exists():
            return path
        if base_dir is not None:
            candidate = (base_dir / path).resolve()
            if candidate.exists():
                return candidate
    return None


def coerce_eval_case(
    case: EvalCase | Path | str | Mapping[str, Any],
    *,
    caption: str | None = None,
    rule: str | None = None,
    context_video_path: Path | str | None = None,
    metadata: dict[str, Any] | None = None,
    video_keys: Sequence[str] = _DEFAULT_VIDEO_KEYS,
    caption_keys: Sequence[str] = _DEFAULT_CAPTION_KEYS,
    rule_keys: Sequence[str] = _DEFAULT_RULE_KEYS,
    context_keys: Sequence[str] = _DEFAULT_CONTEXT_KEYS,
) -> EvalCase:
    if isinstance(case, EvalCase):
        base = case
    elif isinstance(case, Mapping):
        payload = dict(case)
        base_dir_value = payload.get("json_path") or payload.get("_json_path")
        base_dir = Path(base_dir_value).parent if isinstance(base_dir_value, str) and base_dir_value else None
        video_path = first_path(payload, video_keys, base_dir=base_dir)
        if video_path is None:
            raise ValueError("case payload does not contain a usable video path")
        base = EvalCase(
            video_path=video_path,
            caption=first_text(payload, caption_keys),
            rule=first_text(payload, rule_keys),
            context_video_path=first_path(payload, context_keys, base_dir=base_dir),
            metadata=payload,
        )
    else:
        base = EvalCase(video_path=Path(case))

    resolved_caption = caption if caption is not None else base.caption
    resolved_rule = rule if rule is not None else base.rule
    if context_video_path is not None:
        resolved_context = Path(context_video_path)
    else:
        resolved_context = base.context_video_path

    return EvalCase(
        video_path=base.video_path,
        caption=resolved_caption,
        rule=resolved_rule,
        context_video_path=resolved_context,
        metadata=metadata if metadata is not None else base.metadata,
    )
