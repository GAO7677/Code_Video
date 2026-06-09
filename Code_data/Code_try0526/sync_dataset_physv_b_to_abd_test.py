#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SOURCE_ROOT = Path("/data/gaoya/AAA_test_video/Output_try0526/Dataset_physV_B_benchmark")
ABD_B_ROOT = Path("/data/gaoya/AAA_test_video/Output_try0526/ABD_test/B")
METHODS = ["wan22-5B-TI2V", "VACE_1p3B_TI2V", "VACE_1p3B_ctx08"]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    count = 0
    for method in METHODS:
        src_dir = SOURCE_ROOT / "output" / method
        if not src_dir.is_dir():
            continue
        dst_dir = ensure_dir(ABD_B_ROOT / method)
        for src_json in sorted(src_dir.glob("*.json")):
            payload = load_json(src_json)
            normalized = {
                "group": "B",
                "benchmark": "Dataset_physV_B_benchmark",
                "method_name": method,
                "case_key": payload["case_key"],
                "category": payload["category"],
                "input_prompt": payload.get("input_prompt"),
                "input_image": payload.get("input_image"),
                "input_context_video": payload.get("input_context_video"),
                "source_video": payload["source_video"],
                "output_video": payload["output_video"],
                "conditioning_mode": payload.get("conditioning_mode"),
                "context_frames": payload.get("context_frames"),
                "seed": payload.get("seed"),
                "fps": payload.get("fps"),
                "num_frames": payload.get("num_frames"),
                "num_inference_steps": payload.get("num_inference_steps"),
                "cfg_scale": payload.get("cfg_scale"),
                "width": payload.get("width"),
                "height": payload.get("height"),
                "negative_prompt": payload.get("negative_prompt"),
                "original_json": str(src_json),
            }
            write_json(dst_dir / src_json.name, normalized)
            count += 1
    print(f"sync complete: {count} json files")


if __name__ == "__main__":
    main()
