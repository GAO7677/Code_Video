from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


CAPTIONS = {
    "simple_f1_sphere_bounce_roll": (
        "A sphere enters the scene, falls under gravity, bounces on the floor, "
        "and continues rolling."
    ),
    "simple_f1_capsule_slide_spin": (
        "A capsule enters with an initial velocity, slides under friction, and rotates."
    ),
    "simple_f1_capsule_low_spin_long_slide": (
        "A slowly rotating capsule enters and slides a long distance across the floor."
    ),
    "simple_f1_capsule_fast_spin_glancing": (
        "A fast-spinning capsule enters diagonally and rapidly changes orientation."
    ),
    "simple_f1_capsule_upright_tumble_slide": (
        "A tilted capsule enters, tumbles, and changes its axial orientation while sliding."
    ),
    "simple_f1_capsule_highspeed_lowspin": (
        "A capsule moves quickly across the scene with little rotation."
    ),
    "simple_f1_capsule_reverse_spin_sweep": (
        "A capsule sweeps across the scene while rotating in the opposite direction."
    ),
    "simple_f2_cylinder_hits_cylinder": (
        "A horizontal cylinder rolls into an upright cylinder, causing both cylinders "
        "to translate and rotate."
    ),
    "simple_f2_puck_hits_box": (
        "A flat puck slides diagonally into a block, transferring momentum and deflecting."
    ),
    "simple_f3_capsule_box_cylinder_chain": (
        "A capsule pushes a block, and the block then collides with a cylinder."
    ),
    "simple_f3_sphere_chain_reaction": (
        "A sphere hits the first block, which then drives a second block in a chain reaction."
    ),
    "simple_f4_ball_behind_pillars": (
        "A moving ball passes behind two stationary pillars and becomes visible again."
    ),
    "simple_f4_dual_sphere_cross_occlusion": (
        "Two balls enter from opposite sides, cross behind a pillar, and reappear."
    ),
    "simple_f5_cylinder_topple": (
        "A tilted cylinder on a narrow pedestal becomes unstable and topples under gravity."
    ),
    "simple_f5_sphere_drop_on_platform": (
        "A sphere drops onto a platform, rolls across it, and leaves the supporting surface."
    ),
}


class PyBulletTextOCVPDataset(Dataset):
    """Adapter from the existing 24-frame episode NPZ files to TextOCVP."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        num_frames: int = 10,
        image_hw: tuple[int, int] = (64, 112),
        limit: int | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.num_frames = int(num_frames)
        self.image_hw = tuple(int(v) for v in image_hw)
        split_root = self.root / split
        self.metadata_paths = sorted(split_root.glob("*.json"))
        if limit is not None:
            self.metadata_paths = self.metadata_paths[: int(limit)]
        if not self.metadata_paths:
            raise FileNotFoundError(f"no episode JSON files found under {split_root}")

    def __len__(self) -> int:
        return len(self.metadata_paths)

    def __getitem__(self, index: int) -> dict[str, object]:
        metadata_path = self.metadata_paths[index]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        archive_path = metadata_path.with_suffix(".npz")
        with np.load(archive_path, allow_pickle=False) as archive:
            frames = torch.from_numpy(archive["full_frames"]).float()

        if frames.ndim != 4 or int(frames.shape[1]) != 3:
            raise ValueError(f"expected full_frames [T,3,H,W], got {tuple(frames.shape)}")
        frame_ids = torch.linspace(0, frames.shape[0] - 1, self.num_frames).round().long()
        frames = frames[frame_ids].clamp(0.0, 1.0)
        frames = F.interpolate(
            frames,
            size=self.image_hw,
            mode="bilinear",
            align_corners=False,
        )

        template_key = str(metadata.get("template_key", ""))
        caption = CAPTIONS.get(template_key)
        if caption is None:
            raise KeyError(f"no English caption template for {template_key!r}")
        return {
            "video": frames,
            "caption": caption,
            "sample_id": str(metadata.get("sample_id", metadata_path.stem)),
            "template_key": template_key,
            "frame_ids": frame_ids,
        }

