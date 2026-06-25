from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..case_inputs import EvalCase, coerce_eval_case
from ..records import load_payload


def load_eval_case(
    *,
    input_json: Path | None = None,
    video: Path | None = None,
    caption: str | None = None,
    rule: str | None = None,
    context_video: Path | None = None,
) -> EvalCase:
    if input_json is not None:
        payload = load_payload(input_json)
        payload["_json_path"] = str(input_json)
        return coerce_eval_case(
            payload,
            caption=caption,
            rule=rule,
            context_video_path=context_video,
        )
    if video is None:
        raise ValueError("Either input_json or video must be provided")
    return EvalCase(
        video_path=video,
        caption=caption,
        rule=rule,
        context_video_path=context_video,
    )


def result_record(case: EvalCase, result: dict[str, Any]) -> dict[str, Any]:
    record = {
        "video": str(case.video_path),
        "caption": case.caption,
        "rule": case.rule,
        "context_video": str(case.context_video_path) if case.context_video_path else None,
    }
    record.update(result)
    return record


def emit_result(record: dict[str, Any], *, output_json: Path | None = None) -> None:
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))
