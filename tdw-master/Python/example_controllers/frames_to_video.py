from pathlib import Path
import sys

import imageio.v2 as imageio


def frames_to_video(frames_dir: Path, output_path: Path, fps: int = 30) -> None:
    frame_paths = sorted(frames_dir.glob("img_*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"No frames found in {frames_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8) as writer:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("Usage: python frames_to_video.py <frames_dir> <output_mp4> [fps]")
    frames_dir_arg = Path(sys.argv[1]).expanduser().resolve()
    output_path_arg = Path(sys.argv[2]).expanduser().resolve()
    fps_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    frames_to_video(frames_dir=frames_dir_arg, output_path=output_path_arg, fps=fps_arg)
    print(output_path_arg)
