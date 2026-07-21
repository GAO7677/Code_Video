#!/usr/bin/env python3
import argparse
import gc
import json
from pathlib import Path
import random
import sys
import time

import torch
import torch.nn.functional as torch_f


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "third_party/dinov3"))
sys.path.insert(0, str(ROOT / "upstream"))


class CenterSliceSequence:
    def __init__(self, keys, size):
        self.keys = keys
        self.size = size

    def __call__(self, **sample):
        sequence_length = len(sample[self.keys[0]])
        if sequence_length <= self.size:
            return sample
        start = (sequence_length - self.size) // 2
        end = start + self.size
        for key in self.keys:
            if len(sample[key]) != sequence_length:
                raise ValueError(f"unaligned sequence length for {key}")
            sample[key] = sample[key][start:end]
        return sample


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=300)
    parser.add_argument("--index-seed", type=int, default=42)
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    return parser.parse_args()


def metric_values(metrics):
    result = {}
    for key, (value, valid) in metrics.items():
        selected = value[valid]
        if selected.numel() == 0:
            raise RuntimeError(f"metric {key} has no valid values")
        result[key] = float(selected.float().mean().item())
    return result


@torch.inference_mode()
def main():
    args = parse_args()
    if args.max_cases <= 0:
        raise ValueError("max-cases must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    from object_centric_bench.datum import YTVIS
    from object_centric_bench.learn import MetricWrap
    from object_centric_bench.model import ModelWrap
    from object_centric_bench.util import Config, build_from_config
    from object_centric_bench.util_model import interpolat_argmax_attent

    config_file = args.config_file.resolve()
    cfg = Config.fromfile(config_file)
    transform = build_from_config(cfg.dataset_v.transform)
    dataset = YTVIS(
        data_file=args.data_dir.resolve() / cfg.dataset_t.data_file,
        extra_keys=cfg.dataset_t.extra_keys,
        transform0=CenterSliceSequence(
            keys=["video", "segment"], size=cfg.train_clip_frames
        ),
        transform=transform,
        ts=None,
    )
    num_cases = min(args.max_cases, len(dataset))
    indices = sorted(random.Random(args.index_seed).sample(range(len(dataset)), num_cases))
    collate_fn = build_from_config(cfg.collate_fn_v)

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    model = ModelWrap(
        build_from_config(cfg.model), cfg.model_imap, cfg.model_omap
    ).to(device).eval()
    model.freez(cfg.freez, verbose=False)
    acc_fn = MetricWrap(detach=True, **build_from_config(cfg.acc_fn_v))
    amp_dtype = getattr(torch, args.amp_dtype)

    output_file = args.output_file.resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": str(config_file),
        "dataset": str(args.data_dir.resolve() / cfg.dataset_t.data_file),
        "split": "train",
        "sampling": {
            "num_cases": num_cases,
            "dataset_cases": len(dataset),
            "index_seed": args.index_seed,
            "indices": indices,
            "temporal": f"center_{cfg.train_clip_frames}_frames",
            "spatial": "validation_center_crop",
            "model_mode": "eval",
        },
        "amp_dtype": args.amp_dtype,
        "checkpoints": [],
    }

    for checkpoint in args.checkpoint:
        checkpoint = checkpoint.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        state_dict = torch.load(
            checkpoint, map_location="cpu", weights_only=True, mmap=True
        )
        model.load_state_dict(state_dict, strict=True)
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        started = time.time()
        records = []
        totals = {key: 0.0 for key in cfg.acc_fn_v}
        total_recon = 0.0
        frame_counts = []
        for position, index in enumerate(indices, start=1):
            batch = collate_fn([dataset[index]])
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=amp_dtype):
                output = model(batch={"video": batch["video"]})
                recon = (
                    output["recon"] - output["feature"].detach()
                ).square().mean()
                output["segment"] = torch_f.one_hot(
                    interpolat_argmax_attent(
                        output["attentd"].detach(), size=cfg.resolut0
                    ).long()
                ).bool()
                metrics = metric_values(acc_fn(batch=batch, output=output))

            recon_value = float(recon.float().item())
            total_recon += recon_value
            for key, value in metrics.items():
                totals[key] += value
            frames = int(batch["video"].shape[1])
            frame_counts.append(frames)
            records.append(
                {
                    "index": index,
                    "frames": frames,
                    "recon": recon_value,
                    **metrics,
                }
            )
            if position % 25 == 0 or position == num_cases:
                print(
                    f"[train-eval] checkpoint={checkpoint.name} "
                    f"cases={position}/{num_cases}",
                    flush=True,
                )

        summary = {
            "checkpoint": str(checkpoint),
            "recon": total_recon / num_cases,
            **{key: value / num_cases for key, value in totals.items()},
            "min_frames": min(frame_counts),
            "max_frames": max(frame_counts),
            "mean_frames": sum(frame_counts) / num_cases,
            "seconds": time.time() - started,
            "peak_memory_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
            "cases": records,
        }
        payload["checkpoints"].append(summary)
        output_file.write_text(json.dumps(payload, indent=2) + "\n")
        print(
            json.dumps({key: value for key, value in summary.items() if key != "cases"}),
            flush=True,
        )


if __name__ == "__main__":
    main()
