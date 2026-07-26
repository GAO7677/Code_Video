"""Canonical moving-ball Head categories used by grouped ablations."""

from __future__ import annotations


CATEGORY_TARGETS: dict[str, tuple[tuple[int, int], ...]] = {
    "S": ((0, 8), (5, 4), (11, 0), (17, 3), (19, 20), (29, 7)),
    "T": ((0, 13), (5, 9), (11, 22), (17, 8), (19, 11), (29, 22)),
    "P": ((0, 14), (5, 17), (11, 14), (17, 5), (19, 18), (29, 23)),
    "C": ((0, 5), (5, 0), (11, 2), (17, 23), (19, 15), (29, 18)),
    "G": ((0, 0), (5, 19), (11, 4), (17, 7), (19, 21), (29, 2)),
}


def targets_for_category(category: str) -> list[tuple[int, int]]:
    normalized = category.strip().upper()
    try:
        return list(CATEGORY_TARGETS[normalized])
    except KeyError as error:
        raise ValueError(
            f"Unknown category {category!r}; expected one of "
            f"{tuple(CATEGORY_TARGETS)}"
        ) from error

