from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


FFMPEG_DEFAULT = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/ffmpeg")
FFPROBE_DEFAULT = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/ffprobe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 8-frame H.264 context videos and v2v json files for PhyCo samples."
    )
    parser.add_argument(
        "--sample-list",
        type=Path,
        default=Path("/data/gaoya/dataset/nnsriram97-phyco_kubric/test_500_balanced.txt"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/0623/testdataset"),
    )
    parser.add_argument(
        "--json-root",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons"),
    )
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="0 means all samples in the list.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ffmpeg", type=Path, default=FFMPEG_DEFAULT)
    parser.add_argument("--ffprobe", type=Path, default=FFPROBE_DEFAULT)
    return parser.parse_args()


def load_samples(path: Path, limit: int) -> list[Path]:
    samples = [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit > 0:
        return samples[:limit]
    return samples


def build_caption(sample_dir: Path) -> str:
    for candidate in (
        sample_dir / "caption.txt",
        sample_dir.parent.parent / "common_caption_cosmos.txt",
    ):
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8").strip()
            if text:
                return text

    metadata_path = sample_dir / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in ("input_caption", "caption", "prompt"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        simulation_type = str(metadata.get("simulation_type", sample_dir.parts[-3])).replace("_", " ")
        return simulation_type

    return sample_dir.parts[-3].replace("_", " ")


def probe_output(ffprobe: Path, video_path: Path) -> dict[str, str]:
    probe_cmd = [
        str(ffprobe),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,nb_frames,r_frame_rate,width,height",
        "-of",
        "json",
        str(video_path),
    ]
    result = subprocess.run(probe_cmd, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError(f"ffprobe found no streams in {video_path}")
    return streams[0]


def ensure_frame_clip(
    ffmpeg: Path,
    ffprobe: Path,
    source_video: Path,
    output_video: Path,
    frames: int,
    overwrite: bool,
) -> dict[str, str]:
    if overwrite or not output_video.is_file():
        output_video.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(ffmpeg),
            "-y" if overwrite else "-n",
            "-i",
            str(source_video),
            "-frames:v",
            str(frames),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_video),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    stream = probe_output(ffprobe, output_video)
    if stream.get("codec_name") != "h264":
        raise RuntimeError(f"{output_video} codec is not h264: {stream}")
    if int(stream.get("nb_frames", "0")) != frames:
        raise RuntimeError(f"{output_video} frame count mismatch: expected {frames}, got {stream.get('nb_frames')}")
    return stream


def process_sample(
    sample_dir: Path,
    *,
    output_root: Path,
    json_root: Path,
    frames: int,
    overwrite: bool,
    ffmpeg: Path,
    ffprobe: Path,
) -> dict[str, str]:
    scenario = sample_dir.parts[-3]
    date = sample_dir.parts[-2]
    sample_id = sample_dir.parts[-1]
    stem = f"phyco_kubric_{scenario}_{date}_{sample_id}"
    output_dir = output_root / stem
    output_video = output_dir / f"{stem}_frame{frames}.mp4"
    source_video = sample_dir / "rgba.mp4"
    if not source_video.is_file():
        raise FileNotFoundError(f"missing source video: {source_video}")

    stream = ensure_frame_clip(ffmpeg, ffprobe, source_video, output_video, frames, overwrite)
    payload = {
        "source_video": str(source_video.resolve()),
        "input_caption": build_caption(sample_dir),
        "input_video": str(output_video.resolve()),
    }
    json_root.mkdir(parents=True, exist_ok=True)
    json_path = json_root / f"{stem}.json"
    if overwrite or not json_path.is_file():
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "sample_dir": str(sample_dir),
        "json_path": str(json_path),
        "video_path": str(output_video),
        "codec_name": str(stream.get("codec_name", "")),
        "nb_frames": str(stream.get("nb_frames", "")),
    }


def main() -> None:
    args = parse_args()
    if not args.ffmpeg.is_file():
        raise FileNotFoundError(f"ffmpeg not found: {args.ffmpeg}")
    if not args.ffprobe.is_file():
        raise FileNotFoundError(f"ffprobe not found: {args.ffprobe}")

    samples = load_samples(args.sample_list, args.limit)
    processed = 0
    errors: list[dict[str, str]] = []
    for sample_dir in samples:
        try:
            process_sample(
                sample_dir,
                output_root=args.output_root,
                json_root=args.json_root,
                frames=int(args.frames),
                overwrite=bool(args.overwrite),
                ffmpeg=args.ffmpeg,
                ffprobe=args.ffprobe,
            )
            processed += 1
        except Exception as exc:  # noqa: BLE001
            errors.append({"sample_dir": str(sample_dir), "error": str(exc)})

    summary = {
        "sample_list": str(args.sample_list),
        "requested": len(samples),
        "processed": processed,
        "errors": errors,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
