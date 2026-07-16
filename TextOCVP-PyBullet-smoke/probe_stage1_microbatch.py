#!/usr/bin/env python3
"""Probe a real SAVi forward/backward at the requested global micro batch."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


TEXTOCVP_ROOT = Path("/home/gaoya/Code_Video/TextOCVP-master")
sys.path.insert(0, str(TEXTOCVP_ROOT / "src"))
os.chdir(TEXTOCVP_ROOT)

from data.Stage1Indexed import Stage1Indexed  # noqa: E402
from models.SAVi import SAVi  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-root", type=Path, required=True)
    parser.add_argument("--dataset-mode", default="mixed")
    parser.add_argument("--micro-global-batch-size", type=int, required=True)
    return parser.parse_args()


def build_model():
    params = json.loads(
        (TEXTOCVP_ROOT / "src/configs/models/SAVi.json").read_text(encoding="utf-8")
    )
    params["num_slots"] = 8
    params["slot_dim"] = 256
    params["encoder"]["encoder_params"]["resolution"] = [216, 384]
    params["decoder"]["decoder_params"]["resolution"] = [216, 384]
    return SAVi(**copy.deepcopy(params))


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dataset = Stage1Indexed(
        index_root=args.index_root,
        dataset_mode=args.dataset_mode,
        split="train",
        num_frames=10,
        img_size=(216, 384),
        frame_stride=1,
        random_start=True,
        max_samples=args.micro_global_batch_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.micro_global_batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=dataset.collate_fn,
    )
    videos, _ = next(iter(loader))
    model = build_model().cuda()
    device_ids = list(range(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
    for device_id in device_ids:
        torch.cuda.reset_peak_memory_stats(device_id)
    videos = videos.cuda(non_blocking=True)
    output = model(x=videos, num_imgs=videos.shape[1], decode=True)
    loss = F.mse_loss(output["recons_imgs"].clamp(0, 1), videos.clamp(0, 1))
    loss.backward()
    memory = []
    for device_id in device_ids:
        memory.append(
            {
                "gpu": device_id,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device_id) / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device_id) / 2**30,
            }
        )
    print(
        json.dumps(
            {
                "status": "passed",
                "micro_global_batch_size": args.micro_global_batch_size,
                "per_gpu_batch_size": args.micro_global_batch_size // len(device_ids),
                "input_shape": list(videos.shape),
                "loss": float(loss.detach()),
                "memory": memory,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
