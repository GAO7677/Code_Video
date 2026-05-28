#!/usr/bin/env python3
"""用 WMReward 官方 sliding-window V-JEPA2 surprise 评分，对比旧分数"""

import json, os, sys, copy
from pathlib import Path
import torch
from torchvision.transforms.functional import resize
import numpy as np

# WMReward utils.py expects a vjepa2/ subdir — point to torch.hub cache
VJEPA2_SRC = "/home/gaoya/.cache/torch/hub/facebookresearch_vjepa2_main"
sys.path.insert(0, VJEPA2_SRC)
sys.path.insert(0, os.path.join(VJEPA2_SRC, "src"))
sys.path.insert(0, "/home/gaoya/Code_Video/WMReward-main1/WMReward-main")
from utils import compute_vjepa_loss_sliding_window, get_video

DATA_DIR = Path("/data/gaoya/AAA_test_video/Dataset_physV/0526dp")
CKPT_DIR = Path("/data/gaoya/ckpt/Sylvest-vjepa2-vit-g")
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

VIDEO_DIRS = [
    DATA_DIR / "videos" / "ball_block",
    DATA_DIR / "videos" / "jepa_sensitivity",
]


def load_models():
    from models.vision_transformer import vit_giant_xformers_rope
    from models.predictor import vit_predictor

    encoder = vit_giant_xformers_rope(
        img_size=(384, 384), num_frames=16, patch_size=16, tubelet_size=2,
        use_sdpa=True, use_SiLU=False, wide_SiLU=True, uniform_power=False)
    predictor = vit_predictor(
        img_size=(384, 384), patch_size=16, num_frames=16, tubelet_size=2,
        use_mask_tokens=True, embed_dim=encoder.embed_dim, predictor_embed_dim=384,
        depth=12, num_heads=12, num_mask_tokens=10,
        uniform_power=False, use_sdpa=True, use_silu=False, wide_silu=True)
    ckpt = torch.load(str(CKPT_DIR / "vitg-384.pt"), map_location="cpu")

    def clean(d):
        out = {}
        for k, v in d.items():
            out[k.replace("module.", "").replace("backbone.", "")] = v
        return out

    encoder.load_state_dict(clean(ckpt["target_encoder"]), strict=False)
    predictor.load_state_dict(clean(ckpt["predictor"]), strict=False)
    target_encoder = copy.deepcopy(encoder)
    return encoder.eval().cuda(), target_encoder.eval().cuda(), predictor.eval().cuda()


def load_video_tensor(path, max_frames=49):
    video_np = get_video(path, max_frames=max_frames)
    video_tensor = torch.from_numpy(video_np).permute(3, 0, 1, 2).float()
    video_tensor = resize(video_tensor.permute(1, 0, 2, 3), [384, 384]).permute(1, 0, 2, 3)
    video_tensor = (video_tensor / 127.5) - 1.0
    return video_tensor.unsqueeze(0)


def main():
    print("Loading V-JEPA2 ViT-Giant 384...", flush=True)
    encoder, target_encoder, predictor = load_models()
    print("Model loaded.\n", flush=True)

    for vdir in VIDEO_DIRS:
        if not vdir.exists():
            continue
        videos = sorted(vdir.glob("*.mp4"))
        print(f"[{vdir.name}] {len(videos)} videos")

        for vp in videos:
            name = vp.stem
            jp = vdir / f"{name}.json"
            if not jp.exists():
                continue

            print(f"  {name}...", end=" ", flush=True)
            video_tensor = load_video_tensor(str(vp)).cuda()

            with torch.no_grad():
                loss = compute_vjepa_loss_sliding_window(
                    video_tensor=video_tensor, encoder=encoder, target_encoder=target_encoder,
                    predictor=predictor, img_size=384, window_size=16, loss_exp=2,
                    masking_mode="causal", context_frames=8, is_vae_output=True,
                    seed=42, stride=2, mode="mean")
            surprise = float(loss.item())
            similarity = 1.0 - surprise

            meta = json.loads(jp.read_text())
            meta["wmreward_jepa"] = {
                "surprise": surprise,
                "similarity": similarity,
                "method": "WMReward sliding-window (16f, stride 2, causal, cosine dist)",
            }

            old = meta.get("jepa", {}).get("jepa_score")
            if old:
                print(f"WMR={similarity:.4f} old={old:.4f} Δ={similarity-old:+.4f}")
            else:
                print(f"WMR={similarity:.4f}")

            jp.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    print("\nDone")


if __name__ == "__main__":
    main()
