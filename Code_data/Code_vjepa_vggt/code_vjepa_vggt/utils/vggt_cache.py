from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class VGGTDenseCache:
    query_points: torch.Tensor | None
    tracks: torch.Tensor | None
    visibility: torch.Tensor | None
    confidence: torch.Tensor | None
    dense_patch_tokens: torch.Tensor
    patch_grid_hw: tuple[int, int]
    input_hw: tuple[int, int]
    patch_size: int
    frame_indices: list[int]
    source_video: str | None = None
    output_file: str | None = None
    aggregated_last_shape: list[int] | None = None
    patch_start_idx: int | None = None
    dtype: str | None = None
    used_model: bool = False
    pose_enc: torch.Tensor | None = None
    depth: torch.Tensor | None = None
    depth_conf: torch.Tensor | None = None
    world_points: torch.Tensor | None = None
    world_points_conf: torch.Tensor | None = None


def _candidate_cache_names(sample: dict[str, Any]) -> list[str]:
    names: list[str] = []
    video_path = sample.get("video_path")
    if isinstance(video_path, str) and video_path.strip():
        path = Path(video_path.strip())
        names.extend(
            [
                f"{path.stem}.vggt.pt",
                f"{path.name}.vggt.pt",
            ]
        )
        parts = [part for part in path.parts if part not in {"/", ""}]
        if len(parts) >= 2:
            names.append(f"{parts[-2]}__{path.stem}.vggt.pt")
    sample_name = sample.get("sample_name")
    if isinstance(sample_name, str) and sample_name.strip():
        names.append(f"{sample_name.strip()}.vggt.pt")
    stem = sample.get("stem")
    if isinstance(stem, str) and stem.strip():
        names.append(f"{stem.strip()}.vggt.pt")
    return list(dict.fromkeys(names))


def load_vggt_cache(
    sample: dict[str, Any],
    cache_root: str | Path,
    *,
    allow_missing: bool = True,
) -> VGGTDenseCache | None:
    cache_root = Path(cache_root).expanduser().resolve()
    candidates = _candidate_cache_names(sample)
    for rel_name in candidates:
        candidate = cache_root / rel_name
        if not candidate.is_file():
            continue
        payload = torch.load(candidate, map_location="cpu")
        dense_patch_tokens = payload.get("dense_patch_tokens")
        patch_grid_hw = payload.get("patch_grid_hw")
        input_hw = payload.get("input_hw")
        if dense_patch_tokens is None or patch_grid_hw is None or input_hw is None:
            raise RuntimeError(f"invalid VGGT cache payload: {candidate}")
        if not isinstance(dense_patch_tokens, torch.Tensor):
            dense_patch_tokens = torch.as_tensor(dense_patch_tokens)
        if dense_patch_tokens.ndim == 4:
            dense_patch_tokens = dense_patch_tokens.unsqueeze(0)
        return VGGTDenseCache(
            query_points=None,
            tracks=None,
            visibility=None,
            confidence=None,
            dense_patch_tokens=dense_patch_tokens,
            patch_grid_hw=(int(patch_grid_hw[0]), int(patch_grid_hw[1])),
            input_hw=(int(input_hw[0]), int(input_hw[1])),
            patch_size=int(payload.get("patch_size", 14)),
            frame_indices=[int(v) for v in payload.get("frame_indices", [])],
            source_video=payload.get("source_video"),
            output_file=payload.get("output_file"),
            aggregated_last_shape=list(payload.get("aggregated_last_shape", [])) or None,
            patch_start_idx=int(payload.get("patch_start_idx")) if payload.get("patch_start_idx") is not None else None,
            dtype=payload.get("dtype"),
            used_model=bool(payload.get("used_model", False)),
            pose_enc=payload.get("pose_enc"),
            depth=payload.get("depth"),
            depth_conf=payload.get("depth_conf"),
            world_points=payload.get("world_points"),
            world_points_conf=payload.get("world_points_conf"),
        )
    if allow_missing:
        return None
    raise FileNotFoundError(
        f"VGGT cache not found under {cache_root}. tried: {', '.join(candidates) if candidates else '<none>'}"
    )


def sample_dense_patch_tokens_at_query_points(
    dense_patch_tokens: torch.Tensor,
    query_points_xy: torch.Tensor,
    *,
    image_hw: tuple[int, int],
) -> torch.Tensor:
    if dense_patch_tokens.ndim != 5:
        raise ValueError(
            f"dense_patch_tokens must have shape [B,T,H,W,D], got {list(dense_patch_tokens.shape)}"
        )
    if query_points_xy.ndim == 2:
        query_points_xy = query_points_xy.unsqueeze(0).unsqueeze(0)
    elif query_points_xy.ndim == 3:
        query_points_xy = query_points_xy.unsqueeze(1)
    if query_points_xy.ndim != 4:
        raise ValueError(
            f"query_points_xy must have shape [B,T,N,2], [B,N,2], or [N,2], got {list(query_points_xy.shape)}"
        )
    batch, frames, grid_h, grid_w, dim = dense_patch_tokens.shape
    if query_points_xy.shape[0] != batch:
        if query_points_xy.shape[0] == 1:
            query_points_xy = query_points_xy.expand(batch, -1, -1, -1)
        else:
            raise ValueError(
                f"query batch {int(query_points_xy.shape[0])} does not match feature batch {batch}"
            )
    if query_points_xy.shape[1] != frames:
        if query_points_xy.shape[1] == 1:
            query_points_xy = query_points_xy.expand(-1, frames, -1, -1)
        else:
            raise ValueError(
                f"query frames {int(query_points_xy.shape[1])} do not match feature frames {frames}"
            )
    feature_map = dense_patch_tokens.permute(0, 1, 4, 2, 3).reshape(batch * frames, dim, grid_h, grid_w)
    h, w = int(image_hw[0]), int(image_hw[1])
    x = query_points_xy[..., 0] / max(float(w - 1), 1.0)
    y = query_points_xy[..., 1] / max(float(h - 1), 1.0)
    grid = torch.stack([x.clamp(0.0, 1.0) * 2.0 - 1.0, y.clamp(0.0, 1.0) * 2.0 - 1.0], dim=-1)
    grid = grid.reshape(batch * frames, query_points_xy.shape[2], 1, 2)
    sampled = torch.nn.functional.grid_sample(
        feature_map,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.squeeze(-1).permute(0, 2, 1).reshape(batch, frames, query_points_xy.shape[2], dim)
