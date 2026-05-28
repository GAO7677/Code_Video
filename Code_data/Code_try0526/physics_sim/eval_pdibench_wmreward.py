#!/usr/bin/env python3
"""对 PDI-Bench output 视频跑 WMReward JEPA 评估"""

import json, os, sys, copy
from pathlib import Path
import torch
from torchvision.transforms.functional import resize
import numpy as np

VJEPA2_SRC = "/home/gaoya/.cache/torch/hub/facebookresearch_vjepa2_main"
sys.path.insert(0, VJEPA2_SRC)
sys.path.insert(0, os.path.join(VJEPA2_SRC, "src"))
sys.path.insert(0, "/home/gaoya/Code_Video/WMReward-main1/WMReward-main")
from utils import compute_vjepa_loss_sliding_window, get_video

PDI_OUTPUT = Path("/data/gaoya/AAA_test_video/Output_try0526/PDI-Bench/output")
CKPT = Path("/data/gaoya/ckpt/Sylvest-vjepa2-vit-g/vitg-384.pt")
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def load_models():
    from models.vision_transformer import vit_giant_xformers_rope
    from models.predictor import vit_predictor
    encoder = vit_giant_xformers_rope(img_size=(384,384), num_frames=16, patch_size=16, tubelet_size=2,
                                       use_sdpa=True, use_SiLU=False, wide_SiLU=True, uniform_power=False)
    predictor = vit_predictor(img_size=(384,384), patch_size=16, num_frames=16, tubelet_size=2,
                               use_mask_tokens=True, embed_dim=encoder.embed_dim, predictor_embed_dim=384,
                               depth=12, num_heads=12, num_mask_tokens=10,
                               uniform_power=False, use_sdpa=True, use_silu=False, wide_silu=True)
    ckpt = torch.load(str(CKPT), map_location="cpu")
    def clean(d):
        out = {}
        for k, v in d.items():
            out[k.replace("module.","").replace("backbone.","")] = v
        return out
    encoder.load_state_dict(clean(ckpt["target_encoder"]), strict=False)
    predictor.load_state_dict(clean(ckpt["predictor"]), strict=False)
    target_encoder = copy.deepcopy(encoder)
    return encoder.eval().cuda(), target_encoder.eval().cuda(), predictor.eval().cuda()


def load_video_tensor(path, max_frames=49):
    video_np = get_video(path, max_frames=max_frames)
    video_tensor = torch.from_numpy(video_np).permute(3,0,1,2).float()
    video_tensor = resize(video_tensor.permute(1,0,2,3), [384,384]).permute(1,0,2,3)
    video_tensor = (video_tensor / 127.5) - 1.0
    return video_tensor.unsqueeze(0)


def main():
    print("Loading V-JEPA2 ViT-Giant 384...", flush=True)
    encoder, target_encoder, predictor = load_models()
    print("Model loaded.\n", flush=True)

    # Collect all mp4 files
    videos = list(PDI_OUTPUT.rglob("*.mp4"))
    print(f"Found {len(videos)} videos\n")

    for i, vp in enumerate(videos):
        rel = vp.relative_to(PDI_OUTPUT)
        print(f"[{i+1}/{len(videos)}] {rel}...", end=" ", flush=True)

        try:
            video_tensor = load_video_tensor(str(vp)).cuda()
            with torch.no_grad():
                loss = compute_vjepa_loss_sliding_window(
                    video_tensor=video_tensor, encoder=encoder, target_encoder=target_encoder,
                    predictor=predictor, img_size=384, window_size=16, loss_exp=2,
                    masking_mode="causal", context_frames=8, is_vae_output=True,
                    seed=42, stride=2, mode="mean")
            surprise = float(loss.item())
            similarity = 1.0 - surprise

            # Read existing JSON or create
            jp = vp.with_suffix(".json")
            meta = json.loads(jp.read_text()) if jp.exists() else {}
            meta["wmreward_jepa"] = {"surprise": surprise, "similarity": similarity}
            jp.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
            print(f"WMR={similarity:.4f}")
        except Exception as e:
            print(f"FAILED: {e}")

    print("\nDone")


if __name__ == "__main__":
    main()
