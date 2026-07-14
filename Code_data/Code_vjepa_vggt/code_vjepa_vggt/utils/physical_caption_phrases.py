from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
_DETERMINERS = {
    "a", "an", "the", "one", "two", "three", "four", "five", "six",
    "several", "multiple", "another", "each", "both",
}
_EXPLICIT_COUNTS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "both": 2,
    "several": 2,
    "multiple": 2,
}
_BOUNDARIES = {
    "and", "or", "but", "while", "which", "that", "who", "where", "when",
    "on", "in", "at", "above", "below", "behind", "under", "over", "onto",
    "into", "from", "with", "without", "through", "across", "toward", "towards",
    "of", "as", "up", "down", "them", "it",
    "is", "are", "was", "were", "be", "being", "been", "has", "have", "had",
    "holding", "holds", "held", "hanging", "hangs", "suspended", "suspends",
    "sitting", "sits", "standing", "stands", "moving", "moves", "moved",
    "falling", "falls", "fell", "dropping", "drops", "dropped", "rolling",
    "rolls", "rotates", "rotating", "spins", "spinning", "collides", "hits",
    "strikes", "bounces", "slides", "pushes", "pulls", "places", "placed",
    "lowers", "raises", "releases", "released", "lets", "let", "go",
    "propped", "rests", "resting", "supports", "transfers", "causes", "continues",
}
_PHYSICAL_ACTIONS = {
    "hold", "holding", "holds", "held", "hang", "hanging", "suspended",
    "move", "moving", "moves", "fall", "falling", "falls", "fell", "drop",
    "dropping", "drops", "roll", "rolling", "rolls", "rotate", "rotates",
    "rotating", "spin", "spins", "spinning", "collide", "collides", "hit",
    "hits", "strike", "strikes", "bounce", "bounces", "slide", "slides",
    "push", "pushes", "pull", "pulls", "place", "places", "placed", "lower",
    "lowers", "raise", "raises", "release", "releases", "released", "let",
    "propped", "rest", "rests", "transfer", "transfers", "support", "supports",
    "occlude", "occludes", "cover", "covers", "pass", "passes",
}
_MOTION_MODIFIERS = {
    "moving", "falling", "rolling", "rotating", "spinning", "sliding",
    "suspended", "hanging", "stationary", "dropped", "released",
}
_NON_OBJECT_HEADS = {
    "camera", "shot", "movement", "scene", "background", "lighting", "view",
    "perspective", "video", "frame", "motion", "object", "objects",
}
_APPARATUS_HEADS = {
    "grabber", "grabbers", "tool", "tools", "arm", "arms",
}
_SUPPORT_HEADS = {
    "table", "tables", "floor", "floors", "wall", "walls", "platform",
    "platforms", "support", "supports", "cardstock", "paper", "surface", "surfaces",
}

# Verb forms terminate a candidate phrase, except when a motion modifier is the
# first word after a determiner (for example, "a rotating platform").
_BOUNDARIES.update(_PHYSICAL_ACTIONS)
_BOUNDARIES.update({"remain", "remains", "stay", "stays", "stationary", "still"})


@dataclass(frozen=True)
class PhysicalNounPhrase:
    text: str
    head: str
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    score: float
    instance_count: int
    mention_count: int
    nearest_action_distance: int | None
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return payload


def _lemma_head(word: str) -> str:
    value = word.lower()
    if value.endswith("ies") and len(value) > 4:
        return value[:-3] + "y"
    if value.endswith("ses") and len(value) > 4:
        return value[:-2]
    if value.endswith("s") and not value.endswith("ss") and len(value) > 3:
        return value[:-1]
    return value


def _candidate_after_start(tokens: list[str], start: int) -> tuple[int, int] | None:
    index = start + 1
    if index >= len(tokens):
        return None
    end = index
    while end < len(tokens) and end - index < 5:
        token = tokens[end].lower()
        if token in _BOUNDARIES:
            if end == index and token in _MOTION_MODIFIERS:
                end += 1
                continue
            if token == "of" and end == index + 1 and tokens[index].lower() in {"piece", "sheet", "pair", "stack"}:
                end += 1
                if end < len(tokens):
                    end += 1
                break
            break
        end += 1
    return (index, end) if end > index else None


def _raw_candidates(words: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for index, word in enumerate(words):
        if word.lower() in _DETERMINERS:
            span = _candidate_after_start(words, index)
            if span is not None:
                spans.append(span)
        if word.lower() == "and" and index + 1 < len(words):
            next_word = words[index + 1].lower()
            if next_word not in _DETERMINERS and next_word not in _BOUNDARIES:
                end = index + 2
                while end < len(words) and end - (index + 1) < 3 and words[end].lower() not in _BOUNDARIES:
                    end += 1
                spans.append((index + 1, end))
    return spans


def extract_physical_noun_phrases(
    caption: str,
    *,
    max_phrases: int = 4,
    min_score: float = 4.0,
) -> tuple[list[PhysicalNounPhrase], dict[str, object]]:
    matches = list(_TOKEN_RE.finditer(str(caption)))
    words = [match.group(0) for match in matches]
    lower = [word.lower() for word in words]
    action_indices = [index for index, word in enumerate(lower) if word in _PHYSICAL_ACTIONS]

    candidates: list[dict[str, object]] = []
    seen_spans: set[tuple[int, int]] = set()
    for start, end in _raw_candidates(words):
        if (start, end) in seen_spans or start >= end:
            continue
        seen_spans.add((start, end))
        phrase_words = words[start:end]
        head = _lemma_head(phrase_words[-1])
        if head in _NON_OBJECT_HEADS or head in {_lemma_head(item) for item in _APPARATUS_HEADS}:
            continue
        candidates.append(
            {
                "text": str(caption)[matches[start].start() : matches[end - 1].end()],
                "head": head,
                "start": start,
                "end": end,
                "char_start": matches[start].start(),
                "char_end": matches[end - 1].end(),
            }
        )

    head_counts = Counter(str(item["head"]) for item in candidates)
    best_by_surface: dict[tuple[str, str], PhysicalNounPhrase] = {}
    dropped: list[dict[str, object]] = []
    for item in candidates:
        start = int(item["start"])
        end = int(item["end"])
        head = str(item["head"])
        distances = [min(abs(start - action), abs(end - 1 - action)) for action in action_indices]
        nearest = min(distances) if distances else None
        score = 1.0
        reasons = ["noun_phrase"]
        if nearest is not None and nearest <= 3:
            score += 5.0
            reasons.append("physical_action_distance<=3")
        elif nearest is not None and nearest <= 7:
            score += 3.0
            reasons.append("physical_action_distance<=7")
        elif nearest is not None and nearest <= 12:
            score += 1.0
            reasons.append("physical_action_distance<=12")
        mention_count = int(head_counts[head])
        if mention_count > 1:
            score += min(4.0, 2.0 * (mention_count - 1))
            reasons.append(f"repeated_head={mention_count}")
        phrase_lower = {word.lower() for word in words[start:end]}
        if phrase_lower.intersection(_MOTION_MODIFIERS):
            score += 3.0
            reasons.append("motion_modifier")
        if head in {_lemma_head(value) for value in _SUPPORT_HEADS}:
            if nearest is not None and nearest <= 3:
                score += 2.0
                reasons.append("physically_involved_support")
            else:
                score -= 2.0
                reasons.append("static_support_penalty")
        phrase = PhysicalNounPhrase(
            text=str(item["text"]),
            head=head,
            char_start=int(item["char_start"]),
            char_end=int(item["char_end"]),
            token_start=start,
            token_end=end,
            score=float(score),
            instance_count=int(_EXPLICIT_COUNTS.get(lower[start - 1], 1)) if start > 0 else 1,
            mention_count=mention_count,
            nearest_action_distance=nearest,
            reasons=tuple(reasons),
        )
        surface_key = (head, phrase.text.lower())
        previous = best_by_surface.get(surface_key)
        if previous is None or (phrase.score, len(phrase.text)) > (previous.score, len(previous.text)):
            best_by_surface[surface_key] = phrase

    # Keep distinct same-class entities such as "red ball" and "blue ball",
    # while dropping later generic references such as "the ball".
    resolved: list[PhysicalNounPhrase] = []
    surface_phrases = list(best_by_surface.values())
    for phrase in surface_phrases:
        phrase_terms = {word.lower() for word in _TOKEN_RE.findall(phrase.text)}
        richer_reference = next(
            (
                other
                for other in surface_phrases
                if other.head == phrase.head
                and phrase_terms
                < {word.lower() for word in _TOKEN_RE.findall(other.text)}
            ),
            None,
        )
        if richer_reference is not None:
            dropped.append(
                {
                    **phrase.to_dict(),
                    "drop_reason": "generic_reference_to_richer_same_head_phrase",
                    "richer_reference": richer_reference.text,
                }
            )
            continue
        resolved.append(phrase)

    ranked = sorted(
        resolved,
        key=lambda item: (-item.score, item.char_start, item.text.lower()),
    )
    selected = [item for item in ranked if item.score >= float(min_score)][: max(1, int(max_phrases))]
    selected_keys = {(item.head, item.char_start) for item in selected}
    for item in ranked:
        if (item.head, item.char_start) not in selected_keys:
            dropped.append({**item.to_dict(), "drop_reason": "below_score_or_max_phrases"})
    return selected, {
        "mode": "physical_noun_phrases",
        "caption": str(caption),
        "max_phrases": int(max_phrases),
        "min_score": float(min_score),
        "selected": [item.to_dict() for item in selected],
        "dropped": dropped,
    }
