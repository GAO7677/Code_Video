from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from code_vjepa_vggt.trainers.context_video_trainer import ContextVideoTrainer
from code_vjepa_vggt.utils.config import load_yaml_config
from code_vjepa_vggt.utils.masks import collate_video_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/inspect_phys_state_vjepa_vggt.yaml",
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--output",
        default="/data/gaoya/AAA_test_video/0529/vjepa_vggt/tmp/smoke/smoke_train_shapes.json",
    )
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    trainer = ContextVideoTrainer(cfg, build_optimizer=False)
    sample = trainer.dataset[args.index]
    batch = collate_video_batch([sample])
    with torch.no_grad():
        debug = trainer._prepare_batch(batch)["debug"]
    report = {
        "status": "prepare_batch_smoke_ok",
        "note": "Smoke test ran through the new training preparation path, including variable-length context padding, optional SAM2 priors, JEPA/VGGT/VAE encoding, object token pooling, and fused Wan context construction.",
        "sample_index": args.index,
        "debug": debug,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
