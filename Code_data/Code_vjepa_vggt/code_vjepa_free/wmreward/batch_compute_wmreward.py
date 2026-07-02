#!/usr/bin/env python3
import argparse
import copy
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader
from einops import rearrange
from torchvision import transforms


REPO_DIR = Path("/home/gaoya/.cache/torch/hub/facebookresearch_vjepa2_main")
if not REPO_DIR.exists():
    raise FileNotFoundError(f"Missing local vjepa2 hub cache: {REPO_DIR}")
sys.path.insert(0, str(REPO_DIR))

from src.masks.utils import apply_masks  # noqa: E402
from src.models.predictor import vit_predictor  # noqa: E402
from src.models.vision_transformer import vit_giant_xformers, vit_huge  # noqa: E402


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def set_deterministic(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def clean_backbone_key(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "")
        key = key.replace("backbone.", "")
        cleaned[key] = value
    return cleaned


def build_pt_video_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )


def get_video(path: str, max_frames: int = 49) -> np.ndarray:
    vr = VideoReader(path)
    num_frames = len(vr)
    frame_count = min(max_frames, num_frames)
    frame_idx = np.linspace(0, num_frames - 1, frame_count, dtype=int)
    return vr.get_batch(frame_idx).asnumpy()


def load_video_as_tensor(video_path: str, max_frames: int, img_size: int) -> torch.Tensor:
    video_np = get_video(video_path, max_frames=max_frames)
    video = torch.from_numpy(video_np).float()  # [T, H, W, C]
    video = video.permute(0, 3, 1, 2) / 255.0  # [T, C, H, W]
    resize = transforms.Resize((img_size, img_size), antialias=True)
    video = resize(video)
    video = (video * 2.0) - 1.0
    video = video.permute(1, 0, 2, 3).contiguous()  # [C, T, H, W]
    return video.unsqueeze(0)


def choose_window_params(num_frames: int, base_window: int, base_context: int, base_stride: int):
    if num_frames < 2:
        raise ValueError(f"Video has too few frames: {num_frames}")
    window_size = min(base_window, num_frames)
    if window_size < 2:
        raise ValueError(f"window_size became invalid: {window_size}")
    if window_size % 2 == 1:
        window_size -= 1
    if window_size < 2:
        raise ValueError(f"window_size became invalid after even adjustment: {window_size}")

    context_frames = min(base_context, window_size - 2)
    if context_frames < 2:
        context_frames = max(2, window_size // 2)
    if context_frames >= window_size:
        context_frames = window_size - 2
    if context_frames % 2 == 1:
        context_frames -= 1
    if context_frames < 2:
        raise ValueError(
            f"context_frames became invalid for num_frames={num_frames}, window_size={window_size}"
        )

    stride = min(base_stride, window_size)
    stride = max(1, stride)
    return window_size, context_frames, stride


def generate_causal_masks(batch_size, img_size, frames_per_clip, encoder, context_frames, device):
    grid_size = img_size // encoder.patch_size
    grid_depth = frames_per_clip // encoder.tubelet_size
    context_depth = context_frames // encoder.tubelet_size
    future_steps = grid_depth - context_depth
    if future_steps <= 0:
        raise ValueError(
            f"context_frames={context_frames} too large for frames_per_clip={frames_per_clip}"
        )
    n_context = int(grid_size**2 * context_depth)
    n_pred = int(grid_size**2 * future_steps)
    ctxt_positions = torch.arange(n_context, device=device).unsqueeze(0).repeat(batch_size, 1)
    tgt_positions = torch.arange(n_pred, device=device).unsqueeze(0).repeat(batch_size, 1)
    tgt_positions += n_context
    return ctxt_positions, tgt_positions


def load_vjepa_models_local(model_name: str, checkpoint_path: str, device: torch.device):
    img_size = 384 if "384" in model_name else 256
    if model_name in {"vitg", "vit_giant", "vitg384", "vit_giant_384"}:
        encoder = vit_giant_xformers(
            img_size=(img_size, img_size),
            num_frames=64,
            tubelet_size=2,
            use_sdpa=True,
            use_SiLU=False,
            wide_SiLU=True,
            uniform_power=False,
            use_rope=True,
        )
    elif model_name in {"vith", "vit_huge"}:
        encoder = vit_huge(
            img_size=(img_size, img_size),
            num_frames=64,
            tubelet_size=2,
            use_sdpa=True,
            use_SiLU=False,
            wide_SiLU=True,
            uniform_power=False,
            use_rope=True,
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    encoder_state = state_dict.get("target_encoder") or state_dict["encoder"]
    encoder.load_state_dict(clean_backbone_key(encoder_state), strict=False)

    predictor = vit_predictor(
        img_size=(img_size, img_size),
        patch_size=encoder.patch_size,
        use_mask_tokens=True,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=384,
        num_frames=encoder.num_frames,
        tubelet_size=encoder.tubelet_size,
        depth=12,
        num_heads=12,
        num_mask_tokens=10,
        use_rope=True,
        uniform_power=False,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
    )
    predictor.load_state_dict(clean_backbone_key(state_dict["predictor"]), strict=False)

    encoder = encoder.to(device).eval()
    target_encoder = copy.deepcopy(encoder).to(device).eval()
    predictor = predictor.to(device).eval()
    return encoder, target_encoder, predictor, img_size


def preprocess_for_vjepa(video_tensor: torch.Tensor, img_size: int, dtype: torch.dtype, device: torch.device):
    transform = build_pt_video_transform(img_size)
    if video_tensor.dim() == 4:
        video_tensor = video_tensor.unsqueeze(0)
    video_255 = (video_tensor + 1.0) * 127.5
    processed = []
    for b in range(video_255.shape[0]):
        video_tchw = video_255[b].permute(1, 0, 2, 3).to(device)
        video_tchw = video_tchw / 255.0
        video_normalized = transform(video_tchw)
        processed.append(video_normalized.permute(1, 0, 2, 3).contiguous())
    return torch.stack(processed, dim=0).to(device=device, dtype=dtype)


@torch.no_grad()
def compute_vjepa_loss_sliding_window(
    video_tensor: torch.Tensor,
    encoder,
    target_encoder,
    predictor,
    img_size: int = 384,
    window_size: int = 16,
    context_frames: int = 8,
    stride: int = 2,
    mode: str = "mean",
):
    device = next(encoder.parameters()).device
    model_dtype = next(encoder.parameters()).dtype
    clips = preprocess_for_vjepa(video_tensor, img_size=img_size, dtype=model_dtype, device=device)
    pieces = clips.unfold(2, window_size, stride).permute(0, 2, -1, 1, 3, 4).contiguous()
    pieces = pieces.flatten(0, 1)
    pieces = rearrange(pieces, "b t c h w -> b c t h w")

    chunk_losses = []
    for chunk in pieces:
        chunk = chunk.unsqueeze(0)
        masks_enc, masks_pred = generate_causal_masks(
            batch_size=1,
            img_size=img_size,
            frames_per_clip=window_size,
            encoder=encoder,
            context_frames=context_frames,
            device=device,
        )

        h = target_encoder(chunk)
        h = torch.stack([F.layer_norm(hi, (hi.size(-1),)) for hi in h])
        z = encoder(chunk, masks_enc)
        z = predictor(z, masks_enc, masks_pred)
        z = F.layer_norm(z, (z.size(-1),))
        h_masked = apply_masks(h, masks_pred, concat=False)
        loss = 1 - F.cosine_similarity(z, h_masked[0], dim=-1).mean()
        chunk_losses.append(loss)

    if not chunk_losses:
        raise ValueError("No sliding windows were generated for this video.")

    losses = torch.stack(chunk_losses)
    if mode == "mean":
        loss = losses.mean()
    elif mode == "max":
        loss = losses.max()
    else:
        raise ValueError(f"Unsupported aggregation mode: {mode}")
    return loss, len(chunk_losses)


def find_videos_from_json(input_root: str):
    pairs = []
    seen_videos = set()
    for json_path in sorted(Path(input_root).rglob("*.json")):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        output_video = obj.get("output_video")
        if not isinstance(output_video, str):
            continue
        if output_video in seen_videos:
            continue
        seen_videos.add(output_video)
        pairs.append((str(json_path), output_video))
    return pairs


def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def write_rows(csv_path: str, rows):
    ensure_parent(csv_path)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "json_path",
                "video_path",
                "relative_path",
                "surprise_score",
                "similarity_score",
                "num_windows",
                "window_size",
                "context_frames",
                "stride",
                "checkpoint_path",
                "model_name",
                "status",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Batch compute WMReward surprise scores.")
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="vitg384")
    parser.add_argument("--window_size", type=int, default=16)
    parser.add_argument("--context_frames", type=int, default=8)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max_frames", type=int, default=49)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_name", type=str, default="wmreward_scores.csv")
    args = parser.parse_args()

    set_deterministic(args.seed)
    video_pairs = find_videos_from_json(args.input_root)
    if args.limit is not None:
        video_pairs = video_pairs[: args.limit]
    if not video_pairs:
        raise FileNotFoundError(f"No json.output_video entries found under {args.input_root}")

    os.makedirs(args.output_dir, exist_ok=True)
    output_csv = os.path.join(args.output_dir, args.output_name)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    encoder, target_encoder, predictor, img_size = load_vjepa_models_local(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        device=device,
    )

    rows = []
    for idx, (json_path, video_path) in enumerate(video_pairs, start=1):
        rel_path = os.path.relpath(video_path, args.input_root)
        print(f"[{idx}/{len(video_pairs)}] {rel_path}", flush=True)
        try:
            video_np = get_video(video_path, max_frames=args.max_frames)
            num_frames = int(video_np.shape[0])
            window_size, context_frames, stride = choose_window_params(
                num_frames=num_frames,
                base_window=args.window_size,
                base_context=args.context_frames,
                base_stride=args.stride,
            )
            video_tensor = load_video_as_tensor(
                video_path=video_path,
                max_frames=args.max_frames,
                img_size=img_size,
            ).to(device)
            loss, num_windows = compute_vjepa_loss_sliding_window(
                video_tensor=video_tensor,
                encoder=encoder,
                target_encoder=target_encoder,
                predictor=predictor,
                img_size=img_size,
                window_size=window_size,
                context_frames=context_frames,
                stride=stride,
                mode="mean",
            )
            surprise = float(loss.item())
            similarity = 1.0 - surprise
            print(
                f"  surprise={surprise:.6f} similarity={similarity:.6f} windows={num_windows} frames={num_frames} window={window_size} context={context_frames} stride={stride}",
                flush=True,
            )
            rows.append(
                {
                    "json_path": json_path,
                    "video_path": video_path,
                    "relative_path": rel_path,
                    "surprise_score": f"{surprise:.8f}",
                    "similarity_score": f"{similarity:.8f}",
                    "num_windows": num_windows,
                    "window_size": window_size,
                    "context_frames": context_frames,
                    "stride": stride,
                    "checkpoint_path": args.checkpoint_path,
                    "model_name": args.model_name,
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as exc:
            print(f"  ERROR: {exc}", flush=True)
            rows.append(
                {
                    "json_path": json_path,
                    "video_path": video_path,
                    "relative_path": rel_path,
                    "surprise_score": "",
                    "similarity_score": "",
                    "num_windows": "",
                    "window_size": "",
                    "context_frames": "",
                    "stride": "",
                    "checkpoint_path": args.checkpoint_path,
                    "model_name": args.model_name,
                    "status": "error",
                    "error": repr(exc),
                }
            )
        write_rows(output_csv, rows)

    print(f"Saved CSV to {output_csv}")


if __name__ == "__main__":
    main()
