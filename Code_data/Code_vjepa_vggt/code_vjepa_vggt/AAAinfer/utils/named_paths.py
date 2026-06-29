from __future__ import annotations

from pathlib import Path


def resolve_output_root(
    *,
    explicit_output_root: str | Path | None,
    base_output_root: str | Path,
    model_name: str,
) -> Path:
    if explicit_output_root is not None:
        return Path(explicit_output_root).expanduser().resolve()
    return Path(base_output_root).expanduser().resolve() / str(model_name).strip()


def resolve_runtime_root(
    *,
    explicit_runtime_root: str | Path | None,
    base_runtime_root: str | Path,
    model_name: str,
    suffix: str = "_runtime",
) -> Path:
    if explicit_runtime_root is not None:
        return Path(explicit_runtime_root).expanduser().resolve()
    return Path(base_runtime_root).expanduser().resolve() / f"{str(model_name).strip()}{suffix}"
