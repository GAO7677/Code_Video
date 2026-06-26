from __future__ import annotations

import argparse
from pathlib import Path

import torch.backends.cudnn as cudnn

from code_vjepa_vggt.adapters.vggt_adapter import VGGTTrackAdapter
from code_vjepa_vggt.cache_vggt_dense_features import _cache_one_dataset_sample
from code_vjepa_vggt.data.phys_state_dataset import PhysStateEpisodeDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill VGGT cache files for the phys_state val split.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vggt-model-path", default="/data/gaoya/ckpt/facebook-VGGT-1B")
    parser.add_argument("--dataset-split", default="val")
    parser.add_argument("--dataset-start", type=int, default=0)
    parser.add_argument("--dataset-end", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cudnn.enabled = False
    cudnn.benchmark = False
    cudnn.deterministic = False

    dataset = PhysStateEpisodeDataset(
        root=args.dataset_root,
        split=str(args.dataset_split),
        resolution=(512, 896),
        num_context_frames=8,
        context_fraction=0.5,
        random_context_frames=False,
        seed=42,
    )
    start_idx = max(0, int(args.dataset_start))
    end_idx = len(dataset) if args.dataset_end is None else min(len(dataset), int(args.dataset_end))
    if start_idx >= end_idx:
        raise ValueError(f"invalid dataset range [{start_idx}, {end_idx}) for dataset size {len(dataset)}")

    adapter = VGGTTrackAdapter(
        model_path=str(args.vggt_model_path),
        num_queries=8,
        device=str(args.device),
        input_hw=(280, 504),
        trainable=False,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped = 0
    for idx in range(start_idx, end_idx):
        sample = dataset[idx]
        stem = Path(str(sample["video_path"])).stem
        output_path = output_dir / f"{stem}.vggt.pt"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            continue
        payload = _cache_one_dataset_sample(adapter, sample, output_path)
        generated += 1
        print(
            {
                "idx": idx,
                "source_video": payload["source_video"],
                "output_file": payload["output_file"],
            },
            flush=True,
        )

    print(
        {
            "dataset_split": args.dataset_split,
            "dataset_start": start_idx,
            "dataset_end": end_idx,
            "generated": generated,
            "skipped": skipped,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
