#!/usr/bin/env python3
"""JSON-native Stage1B inference with context-aware text guidance controls."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v as base,
)


VALID_MODES = {
    "text_video_baseline",
    "positive_text_off",
    "negative_text_off",
    "video_only",
    "low_text_cfg",
    "anti_duplicate_prompt",
    "adaptive_context_guard",
}

ANTI_DUPLICATE_PREFIX = (
    "Use only the physical objects already visible in the context frames. "
    "Preserve their identity, count, color, and shape. "
    "Do not introduce or duplicate any object. "
)
ANTI_DUPLICATE_NEGATIVE = (
    "additional objects, duplicated objects, duplicate objects, extra balls, "
    "extra blocks, newly appearing objects not present in the context, "
    "object identity changes, objects disappearing"
)

_CATEGORY_ALIASES = {
    "ball": ("tennis ball", "tennis balls", "ball", "balls", "sphere", "spheres"),
    "block": ("block", "blocks", "cube", "cubes"),
    "capsule": ("capsule", "capsules"),
    "cylinder": ("cylinder", "cylinders"),
}
_NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
}


def _pop_option_values(argv: list[str], names: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    index = 1
    while index < len(argv):
        token = argv[index]
        matched_name = next((name for name in names if token == name), None)
        if matched_name is not None:
            if index + 1 >= len(argv):
                raise SystemExit(f"{matched_name} requires a value")
            values.append(argv[index + 1])
            del argv[index : index + 2]
            continue
        matched_name = next(
            (name for name in names if token.startswith(f"{name}=")), None
        )
        if matched_name is not None:
            values.append(token.split("=", 1)[1])
            del argv[index]
            continue
        index += 1
    return values


def _set_option_value(argv: list[str], option: str, value: str) -> None:
    for index, token in enumerate(argv):
        if token == option:
            if index + 1 >= len(argv):
                raise SystemExit(f"{option} requires a value")
            argv[index + 1] = value
            return
        if token.startswith(f"{option}="):
            argv[index] = f"{option}={value}"
            return
    argv.extend([option, value])


def _resolve_single_option(
    names: tuple[str, ...],
    *,
    default: str,
) -> str:
    values = _pop_option_values(sys.argv, names)
    if not values:
        return default
    normalized = [str(value).strip() for value in values]
    if len(set(normalized)) != 1:
        raise SystemExit(f"conflicting values for {names}: {normalized}")
    return normalized[-1]


GUIDANCE_MODE = _resolve_single_option(
    ("--context-guidance-mode", "--condition-mode"),
    default=os.environ.get("CONTEXT_GUIDANCE_MODE", "adaptive_context_guard"),
).lower()
if GUIDANCE_MODE not in VALID_MODES:
    raise SystemExit(
        f"unsupported context guidance mode {GUIDANCE_MODE!r}; "
        f"expected one of {sorted(VALID_MODES)}"
    )

GUARD_CFG_SCALE = float(
    _resolve_single_option(
        ("--context-guard-cfg-scale",),
        default=os.environ.get("CONTEXT_GUARD_CFG_SCALE", "2.5"),
    )
)
GROUNDING_SCORE_THRESHOLD = float(
    _resolve_single_option(
        ("--context-guard-score-threshold",),
        default=os.environ.get("CONTEXT_GUARD_SCORE_THRESHOLD", "0.20"),
    )
)
ANALYSIS_JSON_PATHS = _pop_option_values(
    sys.argv, ("--context-guidance-analyze-json",)
)

_set_option_value(sys.argv, "--method-suffix", GUIDANCE_MODE)


def _infer_count(prefix: str | None, noun: str) -> int:
    if prefix:
        token = prefix.strip().lower().split()[0]
        if token.isdigit():
            return max(1, int(token))
        if token in _NUMBER_WORDS:
            return _NUMBER_WORDS[token]
    return 2 if noun.lower().endswith("s") else 1


def _extract_prompt_object_counts(prompt: str) -> dict[str, int]:
    text = re.sub(r"[^a-z0-9-]+", " ", str(prompt).lower()).strip()
    counts: dict[str, int] = {}
    for category, aliases in _CATEGORY_ALIASES.items():
        alias_pattern = "|".join(
            re.escape(alias) for alias in sorted(aliases, key=len, reverse=True)
        )
        pattern = re.compile(
            rf"(?:(?P<prefix>\b(?:a|an|one|two|three|four|\d+)\b"
            rf"(?:\s+[a-z0-9-]+){{0,4}}\s+))?"
            rf"(?P<noun>\b(?:{alias_pattern})\b)"
        )
        inferred = [
            _infer_count(match.group("prefix"), match.group("noun"))
            for match in pattern.finditer(text)
        ]
        if inferred:
            # Repeated mentions usually refer to the same object, so use max rather than sum.
            counts[category] = max(inferred)
    return counts


def _phrase_categories(phrase: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9-]+", " ", str(phrase).lower()).strip()
    categories: set[str] = set()
    for category, aliases in _CATEGORY_ALIASES.items():
        if any(re.search(rf"\b{re.escape(alias)}\b", normalized) for alias in aliases):
            categories.add(category)
    return categories


def analyze_context_coverage(
    prompt: str,
    object_debug: dict[str, object] | None,
    *,
    score_threshold: float = GROUNDING_SCORE_THRESHOLD,
) -> dict[str, object]:
    expected_counts = _extract_prompt_object_counts(prompt)
    grounding = {}
    if isinstance(object_debug, dict):
        maybe_grounding = object_debug.get("grounding")
        if isinstance(maybe_grounding, dict):
            grounding = maybe_grounding

    phrases = [str(value) for value in grounding.get("object_phrases", [])]
    raw_scores = grounding.get("object_scores", [])
    scores = [float(value) for value in raw_scores] if isinstance(raw_scores, list) else []
    reliable_detections: list[dict[str, object]] = []
    for index, phrase in enumerate(phrases):
        score = scores[index] if index < len(scores) else None
        if score is not None and score < float(score_threshold):
            continue
        reliable_detections.append(
            {
                "index": index,
                "phrase": phrase,
                "score": score,
                "categories": sorted(_phrase_categories(phrase)),
            }
        )

    unmatched_detection_ids = set(range(len(reliable_detections)))
    exact_matches: dict[str, int] = {category: 0 for category in expected_counts}
    for category, expected_count in sorted(
        expected_counts.items(),
        key=lambda item: sum(
            item[0] in detection["categories"] for detection in reliable_detections
        ),
    ):
        for detection_id, detection in enumerate(reliable_detections):
            if exact_matches[category] >= expected_count:
                break
            if detection_id not in unmatched_detection_ids:
                continue
            if category not in detection["categories"]:
                continue
            exact_matches[category] += 1
            unmatched_detection_ids.remove(detection_id)

    expected_total = sum(expected_counts.values())
    exact_match_total = sum(exact_matches.values())
    missing_total = max(0, expected_total - exact_match_total)
    coverage_complete = expected_total > 0 and missing_total == 0
    residual_fallback_used = False

    # GroundingDINO sometimes labels one visible rigid object by shape rather than
    # semantic class (the brown ball in PhysicIQ 025 is "cylinder capsule").
    # Only allow a residual slot to repair this when another requested category
    # matched exactly. This keeps the two pipes in PhysicIQ 057 from being counted
    # as the two offscreen balls.
    if (
        not coverage_complete
        and len(expected_counts) >= 2
        and exact_match_total > 0
        and missing_total <= 1
        and len(reliable_detections) >= expected_total
        and len(unmatched_detection_ids) >= missing_total
    ):
        coverage_complete = True
        residual_fallback_used = True

    if coverage_complete and residual_fallback_used:
        coverage_basis = "exact_plus_one_residual_slot"
    elif coverage_complete:
        coverage_basis = "exact"
    elif expected_total == 0:
        coverage_basis = "no_supported_prompt_objects"
    else:
        coverage_basis = "incomplete"

    return {
        "expected_counts": expected_counts,
        "expected_total": expected_total,
        "reliable_detections": reliable_detections,
        "reliable_detection_count": len(reliable_detections),
        "score_threshold": float(score_threshold),
        "exact_matches": exact_matches,
        "exact_match_total": exact_match_total,
        "missing_total": missing_total,
        "coverage_complete": coverage_complete,
        "coverage_basis": coverage_basis,
        "residual_fallback_used": residual_fallback_used,
    }


def _join_negative_prompt(original: str | None, addition: str) -> str:
    if original is None or not str(original).strip():
        return addition
    return f"{str(original).rstrip(' ,')}, {addition}"


def _effective_conditioning(
    *,
    prompt: str,
    negative_prompt: str | None,
    cfg_scale: float,
    coverage: dict[str, object],
) -> dict[str, object]:
    effective_prompt = str(prompt)
    effective_negative = negative_prompt
    effective_cfg = float(cfg_scale)
    guard_applied = False
    reason = "mode_does_not_apply_guard"

    if GUIDANCE_MODE == "positive_text_off":
        effective_prompt = ""
        reason = "positive_text_removed_negative_text_preserved"
    elif GUIDANCE_MODE == "negative_text_off":
        effective_negative = ""
        reason = "negative_text_removed_positive_text_preserved"
    elif GUIDANCE_MODE == "video_only":
        effective_prompt = ""
        effective_negative = ""
        reason = "positive_and_negative_text_removed"
    elif GUIDANCE_MODE == "low_text_cfg":
        effective_cfg = float(GUARD_CFG_SCALE)
        reason = "unconditional_low_text_cfg"
    elif GUIDANCE_MODE == "anti_duplicate_prompt":
        effective_prompt = ANTI_DUPLICATE_PREFIX + effective_prompt
        effective_negative = _join_negative_prompt(
            effective_negative, ANTI_DUPLICATE_NEGATIVE
        )
        guard_applied = True
        reason = "unconditional_anti_duplicate_prompt"
    elif GUIDANCE_MODE == "adaptive_context_guard":
        if bool(coverage.get("coverage_complete", False)):
            effective_prompt = ANTI_DUPLICATE_PREFIX + effective_prompt
            effective_negative = _join_negative_prompt(
                effective_negative, ANTI_DUPLICATE_NEGATIVE
            )
            effective_cfg = float(GUARD_CFG_SCALE)
            guard_applied = True
            reason = "all_supported_prompt_objects_covered_by_context"
        else:
            reason = "prompt_objects_missing_or_coverage_uncertain"
    elif GUIDANCE_MODE == "text_video_baseline":
        reason = "baseline"

    return {
        "mode": GUIDANCE_MODE,
        "guard_applied": guard_applied,
        "reason": reason,
        "original_prompt": str(prompt),
        "effective_prompt": effective_prompt,
        "original_negative_prompt": negative_prompt,
        "effective_negative_prompt": effective_negative,
        "original_cfg_scale": float(cfg_scale),
        "effective_cfg_scale": effective_cfg,
        "guard_cfg_scale": float(GUARD_CFG_SCALE),
        "coverage": coverage,
    }


_original_run_single_case = base._run_single_case_in_process
_original_has_complete_output = base._has_complete_existing_output


def _run_single_case_with_context_guidance(*args, **kwargs):
    model = kwargs["model"]
    original_prompt = str(kwargs.get("input_caption", ""))
    original_negative = kwargs.get("negative_prompt")
    original_cfg = float(kwargs.get("cfg_scale", 5.0))
    state: dict[str, object] = {}

    original_object_builder = base.infer0705._build_object_context

    def _capture_object_context(*builder_args, **builder_kwargs):
        object_context, object_debug = original_object_builder(
            *builder_args, **builder_kwargs
        )
        coverage = analyze_context_coverage(original_prompt, object_debug)
        state["coverage"] = coverage
        state["conditioning"] = _effective_conditioning(
            prompt=original_prompt,
            negative_prompt=original_negative,
            cfg_scale=original_cfg,
            coverage=coverage,
        )
        return object_context, object_debug

    pipe_class = type(model.pipe)
    original_pipe_call = pipe_class.__call__

    def _guarded_pipe_call(self, *call_args, **call_kwargs):
        conditioning = state.get("conditioning")
        if not isinstance(conditioning, dict):
            coverage = analyze_context_coverage(original_prompt, None)
            conditioning = _effective_conditioning(
                prompt=original_prompt,
                negative_prompt=original_negative,
                cfg_scale=original_cfg,
                coverage=coverage,
            )
            state["coverage"] = coverage
            state["conditioning"] = conditioning
        call_kwargs["prompt"] = str(conditioning["effective_prompt"])
        call_kwargs["cfg_scale"] = float(conditioning["effective_cfg_scale"])
        effective_negative = conditioning["effective_negative_prompt"]
        if effective_negative is None:
            call_kwargs.pop("negative_prompt", None)
        else:
            call_kwargs["negative_prompt"] = str(effective_negative)
        state["pipe_call_count"] = int(state.get("pipe_call_count", 0)) + 1
        return original_pipe_call(self, *call_args, **call_kwargs)

    base.infer0705._build_object_context = _capture_object_context
    pipe_class.__call__ = _guarded_pipe_call
    try:
        result, logs = _original_run_single_case(*args, **kwargs)
    finally:
        pipe_class.__call__ = original_pipe_call
        base.infer0705._build_object_context = original_object_builder

    conditioning = state.get("conditioning")
    if not isinstance(conditioning, dict):
        coverage = analyze_context_coverage(original_prompt, result.get("object_debug"))
        conditioning = _effective_conditioning(
            prompt=original_prompt,
            negative_prompt=original_negative,
            cfg_scale=original_cfg,
            coverage=coverage,
        )
    conditioning["pipe_call_count"] = int(state.get("pipe_call_count", 0))
    result["context_guidance"] = conditioning
    result["input_caption_original"] = original_prompt
    result["input_caption_effective"] = conditioning["effective_prompt"]
    result["negative_prompt"] = conditioning["effective_negative_prompt"]
    result["guidance"] = conditioning["effective_cfg_scale"]
    model_args = result.get("model_args")
    if isinstance(model_args, dict):
        model_args["cfg_scale"] = conditioning["effective_cfg_scale"]
        model_args["negative_prompt"] = conditioning["effective_negative_prompt"]
    logs.append(
        "[context-guidance] "
        f"mode={GUIDANCE_MODE} "
        f"guard_applied={conditioning['guard_applied']} "
        f"coverage={conditioning['coverage']['coverage_basis']} "
        f"cfg={conditioning['original_cfg_scale']}->{conditioning['effective_cfg_scale']}"
    )
    return result, logs


def _has_complete_context_guidance_output(output_video: Path, output_json: Path) -> bool:
    if not _original_has_complete_output(output_video, output_json):
        return False
    try:
        with output_json.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    context_guidance = payload.get("context_guidance")
    return isinstance(context_guidance, dict) and context_guidance.get("mode") == GUIDANCE_MODE


def _run_analysis_only(paths: list[str]) -> None:
    outputs = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        prompt = str(
            payload.get("input_caption_original", payload.get("input_caption", ""))
        )
        coverage = analyze_context_coverage(prompt, payload.get("object_debug"))
        outputs.append({"path": str(path), "prompt": prompt, "coverage": coverage})
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


def main() -> None:
    if ANALYSIS_JSON_PATHS:
        _run_analysis_only(ANALYSIS_JSON_PATHS)
        return
    base._run_single_case_in_process = _run_single_case_with_context_guidance
    base._has_complete_existing_output = _has_complete_context_guidance_output
    base.main()


if __name__ == "__main__":
    main()
