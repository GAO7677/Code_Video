#!/usr/bin/env python3
"""Run a representative multi-case context sweep batch for one GPU."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


WAN_PYTHON = Path("/data/gaoya/miniconda3/envs/wan/bin/python")
TRAIN0419_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_train/train_0419")
VACE_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.1-VACE-1.3B")
BENCH_ROOT = Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V_nullcaption")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-json", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--context-lengths", default="16,32,38")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    payload = read_json(args.batch_json)
    cases = payload.get("cases", [])
    work_root = BENCH_ROOT / "tools" / "dataset_representative_context_sweeps"
    meta_root = work_root / "meta"
    generated_root = work_root / "generated"
    runtime_root = work_root / "runtime"
    meta_root.mkdir(parents=True, exist_ok=True)
    generated_root.mkdir(parents=True, exist_ok=True)
    runtime_root.mkdir(parents=True, exist_ok=True)

    for context_frames in [int(x.strip()) for x in args.context_lengths.split(",") if x.strip()]:
        list_path = meta_root / f"gpu{args.gpu}_context_{context_frames:02d}f.txt"
        meta_paths: list[str] = []
        for item in cases:
            dataset = str(item["dataset"])
            sample_id = str(item["sample_id"])
            meta_json_path = Path(str(item["meta_json_path"]))
            tmp_meta = meta_root / f"{dataset}__{sample_id}__context_{context_frames:02d}f.json"
            data = read_json(meta_json_path)
            data["caption"] = data.get("caption", "")
            tmp_meta.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            meta_paths.append(str(tmp_meta))
        list_path.write_text("\n".join(meta_paths) + "\n", encoding="utf-8")

        command = [
            str(WAN_PYTHON),
            str(TRAIN0419_ROOT / "batch_eval_vace.py"),
            "--vace_root",
            str(VACE_ROOT),
            "--meta_list_path",
            str(list_path),
            "--output_root",
            str(generated_root / f"batch_context_{context_frames:02d}f"),
            "--runtime_root",
            str(runtime_root / f"gpu{args.gpu}_context_{context_frames:02d}f"),
            "--model_name",
            f"representative_gpu{args.gpu}_ctx{context_frames:02d}f",
            "--mode",
            "v2v_clipref",
            "--device",
            "cuda:0",
            "--height",
            "544",
            "--width",
            "720",
            "--fps",
            "16",
            "--num_frames",
            "49",
            "--context_frames",
            str(context_frames),
            "--num_inference_steps",
            "50",
            "--cfg_scale",
            "5.0",
            "--seed",
            "42",
            "--overwrite",
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
