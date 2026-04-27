from __future__ import annotations

import os
from pathlib import Path

from .config import BenchConfig


VBENCH_TARGETS = {
    "clip_vit_b32": ("vbench", "clip_model/ViT-B-32.pt"),
    "clip_vit_l14": ("vbench", "clip_model/ViT-L-14.pt"),
    "dino_vitb16": ("vbench", "dino_model/dino_vitbase16_pretrain.pth"),
    "dino_repo_dir": ("vbench", "dino_model/facebookresearch_dino_main"),
    "amt_s": ("vbench", "amt_model/amt-s.pth"),
    "raft_things": ("vbench", "raft_model/models/raft-things.pth"),
    "musiq_spaq": ("vbench", "pyiqa_model/musiq_spaq_ckpt-358bb6af.pth"),
    "umt_human_action": ("vbench", "umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth"),
    "grit_densecap": ("vbench", "grit_model/grit_b_densecap_objectdet.pth"),
    "tag2text_swin": ("vbench", "caption_model/tag2text_swin_14m.pth"),
    "viclip_pretrain": ("vbench", "ViCLIP/ViClip-InternVid-10M-FLT.pth"),
    "clip_vit_b32_vbench2": ("vbench2", "clip_model/ViT-B-32.pt"),
    "clip_vit_l14_vbench2": ("vbench2", "clip_model/ViT-L-14.pt"),
    "dino_vitb16_vbench2": ("vbench2", "dino_model/dino_vitbase16_pretrain.pth"),
    "dino_repo_dir_vbench2": ("vbench2", "dino_model/facebookresearch_dino_main"),
    "amt_s_vbench2": ("vbench2", "amt_model/amt-s.pth"),
    "raft_things_vbench2": ("vbench2", "raft_model/models/raft-things.pth"),
    "musiq_spaq_vbench2": ("vbench2", "pyiqa_model/musiq_spaq_ckpt-358bb6af.pth"),
}


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink():
            dst.unlink()
        elif dst.is_dir():
            raise RuntimeError(f"Refusing to replace existing real directory: {dst}")
        else:
            dst.unlink()
    os.symlink(src, dst)


def link_manual_weights(config: BenchConfig) -> list[tuple[str, str, str]]:
    linked: list[tuple[str, str, str]] = []
    for key, src_path in config.weights_paths.items():
        if key not in VBENCH_TARGETS:
            continue
        src = Path(src_path).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"Configured weight path does not exist: {key} -> {src}")

        cache_family, relative_dst = VBENCH_TARGETS[key]
        cache_root = config.paths.vbench_cache_dir if cache_family == "vbench" else config.paths.vbench2_cache_dir
        dst = Path(cache_root) / relative_dst
        _link(src, dst)
        linked.append((key, str(src), str(dst)))
    return linked
