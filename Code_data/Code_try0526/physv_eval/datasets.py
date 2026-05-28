from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .paths import A_OUTPUT, DATA_ROOT


@dataclass(frozen=True)
class GroupSpec:
    group_id: str
    title: str
    description: str
    json_root: Path
    pattern: str

    def iter_jsons(self) -> list[Path]:
        return sorted(self.json_root.glob(self.pattern))


GROUP_SPECS: dict[str, GroupSpec] = {
    "A": GroupSpec(
        group_id="A",
        title="PDI-Bench 生成视频",
        description="PDI-Bench 的 GT / Wan / VACE 视频，用于方法级比较。",
        json_root=A_OUTPUT,
        pattern="**/*.json",
    ),
    "B1": GroupSpec(
        group_id="B1",
        title="Ball-Block 物理参数",
        description="固定外观，只改恢复系数、摩擦和球质量。",
        json_root=DATA_ROOT / "videos" / "ball_block",
        pattern="*.json",
    ),
    "B2": GroupSpec(
        group_id="B2",
        title="JEPA 运动敏感性",
        description="固定外观，系统改变速度、质量、重力、碰撞与方向。",
        json_root=DATA_ROOT / "videos" / "jepa_sensitivity",
        pattern="*.json",
    ),
    "B3": GroupSpec(
        group_id="B3",
        title="外观敏感性",
        description="同一物理轨迹，只改渲染外观与光照。",
        json_root=DATA_ROOT / "videos" / "ball_block_appearance",
        pattern="*.json",
    ),
    "C": GroupSpec(
        group_id="C",
        title="帧序打乱 Sanity Check",
        description="只破坏时序，不改单帧内容。",
        json_root=DATA_ROOT / "videos" / "shuffle_test",
        pattern="*.json",
    ),
}


def iter_group_jsons(group_id: str) -> list[Path]:
    try:
        return GROUP_SPECS[group_id].iter_jsons()
    except KeyError as exc:
        raise ValueError(f"Unknown group: {group_id}") from exc
