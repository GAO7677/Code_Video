from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from torchvision.transforms.functional import resize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score BoN candidates with WMReward.")
    parser.add_argument("--wmreward-repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidates-root", type=Path, required=True)
    parser.add_argument("--rewards-root", type=Path, required=True)
    parser.add_argument("--selected-root", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--model", default="vitg384")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.wmreward_repo.resolve()))
    from utils import compute_vjepa_loss_sliding_window, get_video, load_vjepa_model_source

    entries = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    encoder, target_encoder, predictor, img_size = load_vjepa_model_source(args.model)
    args.rewards_root.mkdir(parents=True, exist_ok=True)
    args.selected_root.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        reward_path = args.rewards_root / f"{entry['benchmark_id']}.json"
        selected_path = args.selected_root / entry["generated_video_name"]
        if reward_path.is_file() and selected_path.is_file():
            print(f"[skip] {entry['benchmark_id']}")
            continue
        scores = []
        for index in range(args.candidates):
            video_path = args.candidates_root / entry["benchmark_id"] / f"candidate_{index:02d}.mp4"
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            video_np = get_video(str(video_path), max_frames=49)
            tensor = torch.from_numpy(video_np).permute(3, 0, 1, 2).float()
            tensor = resize(tensor.permute(1, 0, 2, 3), [img_size, img_size]).permute(1, 0, 2, 3)
            tensor = ((tensor / 127.5) - 1.0).unsqueeze(0).cuda()
            with torch.no_grad():
                surprise = compute_vjepa_loss_sliding_window(
                    video_tensor=tensor, encoder=encoder, target_encoder=target_encoder,
                    predictor=predictor, img_size=img_size, window_size=16,
                    loss_exp=2, masking_mode="causal", context_frames=8,
                    is_vae_output=True, seed=42, stride=8, mode="mean",
                ).item()
            scores.append({"candidate_index": index, "surprise": surprise,
                           "reward": 1.0 - surprise, "video": str(video_path)})
            del tensor
            torch.cuda.empty_cache()
        best = min(scores, key=lambda item: item["surprise"])
        shutil.copy2(best["video"], selected_path)
        reward_path.write_text(json.dumps({
            "benchmark_id": entry["benchmark_id"], "selection": "minimum_surprise",
            "best": best, "scores": scores, "selected_video": str(selected_path),
        }, indent=2) + "\n")
        print(f"[done] {entry['benchmark_id']} best={best['candidate_index']:02d}")


if __name__ == "__main__":
    main()
