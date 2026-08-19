#!/usr/bin/env python3
"""Build exact-input visual assets for each Qwen3.8 multi-case result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

from build_qwen38_demo_visualization import make_contact_sheet, to_rgb_uint8, unpack_video_patches, write_video


DEFAULT_RESULTS = Path("/data/gaoya/agent-data/outputs/physv_qwen3_8/six_cases_gpu7_fla.jsonl")
DEFAULT_MODEL = "/data/gaoya/ckpt/Qwen-Qwen3.8-27B-FP8"
DEFAULT_OUTPUT = Path("/data/gaoya/agent-data/outputs/physv_qwen3_8/viewer_six_gpu7_fla")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def slug(case_id: str) -> str:
    return case_id.replace("/", "__")


def build_case(processor: Any, row: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    request = row["video_request"]
    messages = [{"role": "user", "content": [
        {"type": "video", "video": row["video"], **request},
        {"type": "text", "text": row["prompt"]},
    ]}]
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    _, video_inputs, video_kwargs = process_vision_info(
        messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
    )
    raw_video, metadata = video_inputs[0]
    inputs = processor(
        text=[prompt_text], videos=[raw_video], video_metadata=[metadata], padding=True,
        return_tensors="pt", **video_kwargs,
    )
    grid_thw = [int(value) for value in inputs["video_grid_thw"][0].tolist()]
    processed = unpack_video_patches(inputs["pixel_values_videos"], grid_thw, processor.video_processor)
    raw_rgb = to_rgb_uint8(raw_video)
    processed_rgb = to_rgb_uint8(processed, processor.video_processor)
    indices = [int(value) for value in metadata["frames_indices"]]
    source_fps = float(metadata["fps"])
    display_fps = len(indices) * source_fps / int(metadata["total_num_frames"])
    case_slug = slug(row["case_id"])
    case_dir = output_dir / case_slug
    write_video(raw_rgb, case_dir / "sampled_frames.mp4", display_fps)
    write_video(processed_rgb, case_dir / "processor_frames.mp4", display_fps)
    make_contact_sheet(raw_rgb, indices, source_fps, case_dir / "sampled_contact_sheet.jpg")
    return {
        "case_slug": case_slug,
        "source_video": row["video"],
        "assets": {
            "source": f"/media/{case_slug}/source",
            "sampled": f"/media/{case_slug}/sampled",
            "processor": f"/media/{case_slug}/processor",
            "sheet": f"/media/{case_slug}/sheet",
        },
        "audit": {
            "source_fps": source_fps,
            "source_total_frames": int(metadata["total_num_frames"]),
            "frames_indices": indices,
            "sampled_frame_count": len(indices),
            "input_replay_fps": display_fps,
            "raw_video_shape": list(raw_video.shape),
            "processor_frame_shape": list(processed.shape),
            "pixel_values_videos_shape": list(inputs["pixel_values_videos"].shape),
            "video_grid_thw": grid_thw,
            "input_ids_shape": list(inputs["input_ids"].shape),
            "video_backend": str(metadata.get("video_backend", "unknown")),
        },
    }


def main() -> None:
    args = parse_args()
    rows = load_rows(args.results)
    if len(rows) != 6 or any(row.get("status") != "ok" for row in rows):
        raise ValueError("Expected six successful inference rows before building the viewer")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True, trust_remote_code=True)
    public_rows = []
    for row in rows:
        public = dict(row)
        public.update(build_case(processor, row, args.output_dir))
        public_rows.append(public)
        print(f"built={row['case_id']}", flush=True)
    with (args.output_dir / "viewer_data.json").open("w", encoding="utf-8") as handle:
        json.dump({"cases": public_rows}, handle, ensure_ascii=False, indent=2)
    print(f"viewer_data={args.output_dir / 'viewer_data.json'}")


if __name__ == "__main__":
    main()
