#!/usr/bin/env python3
from argparse import Namespace
import argparse
import gc
import json
from pathlib import Path

import torch

import eval as official_eval


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    return parser.parse_args()


def model_specs(weights_dir):
    return [
        (
            "rsfq2_c-movi_c",
            ROOT / "eval-configs-ytvis-hq/rsfq2_c-movi_c_on_ytvis_hq.py",
            weights_dir / "rsfq2_c-movi_c",
            "discovery",
        ),
        (
            "rsfq2_c-movi_e",
            ROOT / "eval-configs-ytvis-hq/rsfq2_c-movi_e_on_ytvis_hq.py",
            weights_dir / "rsfq2_c-movi_e",
            "discovery",
        ),
        (
            "rsfq2_r-ytvis",
            ROOT / "eval-configs-ytvis-hq/rsfq2_r-ytvis_on_ytvis_hq.py",
            weights_dir / "rsfq2_r-ytvis",
            "discovery",
        ),
        (
            "rsfq2_r-ytvis_hq",
            ROOT / "config-randsfq/rsfq2_r-ytvis_hq.py",
            weights_dir / "rsfq2_r-ytvis_hq",
            "discovery",
        ),
        (
            "rsfq2_r_recogn-ytvis_hq",
            ROOT / "config-randsfq/rsfq2_r_recogn-ytvis_hq.py",
            weights_dir / "rsfq2_r_recogn-ytvis_hq",
            "recognition",
        ),
    ]


def to_jsonable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def main():
    args = parse_args()
    output_file = args.output_file.resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "inference_code": str((ROOT / "eval.py").resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "dataset": str((args.data_dir / "ytvis_hq/val.lmdb").resolve()),
        "results": [],
    }

    for model_name, config_file, checkpoint_dir, task in model_specs(
        args.weights_dir.resolve()
    ):
        checkpoints = sorted(checkpoint_dir.glob("*.pth"))
        if len(checkpoints) != 3:
            raise RuntimeError(
                f"expected three checkpoints for {model_name}, got {checkpoints}"
            )
        for checkpoint in checkpoints:
            print(f"[official-eval] start {model_name}/{checkpoint.name}", flush=True)
            eval_args = Namespace(
                cfg_file=config_file,
                data_dir=args.data_dir.resolve(),
                ckpt_file=checkpoint,
                is_viz=False,
                is_img=False,
                dump_log=False,
            )
            metrics = official_eval.main(eval_args)
            result = {
                "model": model_name,
                "task": task,
                "seed": int(checkpoint.name.split("-")[0]),
                "checkpoint": str(checkpoint),
                "config": str(config_file),
                "metrics": {key: to_jsonable(value) for key, value in metrics.items()},
            }
            payload["results"].append(result)
            output_file.write_text(json.dumps(payload, indent=2) + "\n")
            print(json.dumps(result), flush=True)
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
