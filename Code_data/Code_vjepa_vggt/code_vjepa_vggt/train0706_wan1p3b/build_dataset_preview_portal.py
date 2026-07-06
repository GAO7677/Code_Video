from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code_vjepa_vggt.train0706_wan1p3b.dataset import WanTI2VDataset


DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/agent-data/outputs/train0706_dataset_preview")
DEFAULT_PORTAL_PORT = 8798


@dataclass(frozen=True)
class PreviewCase:
    dataset_name: str
    sample_index: int
    prompt: str
    video_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small preview portal for the 1.3B training datasets.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-openvid", type=int, default=3)
    parser.add_argument("--num-raw-phys", type=int, default=3)
    parser.add_argument("--num-genesis", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--openvid-root", type=str, default="/data/gaoya/dataset/mvp-lab-OpenVidHD-0.4M-720p-48fps/train_subset_0530/train")
    parser.add_argument("--raw-phys-root", type=str, default="/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/raw_v1/industrial_s1_scale2_merged_h264_batch1500")
    parser.add_argument("--genesis-root", type=str, default="/data/gaoya/AAA_test_video/Dataset_physV/0417data/version0515zoom_genesis_rigid")
    parser.add_argument("--port", type=int, default=DEFAULT_PORTAL_PORT)
    return parser.parse_args()


def _ensure_empty_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _frames_to_uint8(frames) -> np.ndarray:
    arr = []
    for frame in frames:
        if isinstance(frame, np.ndarray):
            item = frame
        else:
            item = np.asarray(frame)
        if item.dtype != np.uint8:
            item = np.clip(item, 0.0, 1.0) if item.dtype.kind == "f" and item.max() <= 1.5 else item
            if item.dtype.kind == "f":
                item = (item * 255.0).round()
            item = item.astype(np.uint8)
        arr.append(item)
    return np.stack(arr, axis=0)


def _save_video(path: Path, frames, fps: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames_u8 = _frames_to_uint8(frames)
    imageio.mimsave(path, frames_u8, fps=fps, macro_block_size=None)


def _render_dataset(name: str, dataset, count: int, out_dir: Path, seed: int) -> list[PreviewCase]:
    rng = random.Random(seed)
    total = len(dataset)
    if total <= 0:
        raise RuntimeError(f"{name} dataset is empty")
    indices = list(range(total))
    rng.shuffle(indices)
    picked = indices[: min(count, total)]

    cases: list[PreviewCase] = []
    for rank, sample_index in enumerate(picked):
        sample = dataset[sample_index]
        prompt = str(sample.get("prompt", "")).strip()
        video = sample["video"]
        case_dir = out_dir / f"{rank:02d}__idx{sample_index:06d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        video_path = case_dir / "preview.mp4"
        meta_path = case_dir / "meta.json"
        _save_video(video_path, video, fps=8)
        meta = {
            "dataset_name": name,
            "sample_index": int(sample_index),
            "prompt": prompt,
            "video_path": str(video_path.resolve()),
            "num_frames": int(len(video)),
            "shape": list(np.asarray(video[0]).shape) if len(video) else None,
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        cases.append(
            PreviewCase(
                dataset_name=name,
                sample_index=int(sample_index),
                prompt=prompt,
                video_path=str(video_path.resolve()),
            )
        )
    return cases


def _build_index_html(cases: list[PreviewCase], output_root: Path) -> Path:
    rows = []
    for case in cases:
        rel_video = Path(case.video_path).resolve().relative_to(output_root.resolve()).as_posix()
        rows.append(
            f"""
            <section class="card">
              <div class="title">{case.dataset_name} · idx {case.sample_index}</div>
              <div class="prompt">{case.prompt}</div>
              <video controls playsinline muted preload="metadata" src="{rel_video}"></video>
            </section>
            """
        )
    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>train0706 dataset preview</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: #f5f1e8;
      color: #1d1a16;
    }}
    .wrap {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1 {{
      margin: 0 0 12px;
      font-size: 28px;
    }}
    .sub {{
      margin-bottom: 20px;
      color: #665d53;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
      gap: 16px;
    }}
    .card {{
      background: rgba(255,255,255,0.9);
      border: 1px solid rgba(0,0,0,0.08);
      border-radius: 16px;
      padding: 14px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    }}
    .title {{
      font-weight: 700;
      margin-bottom: 8px;
    }}
    .prompt {{
      font-size: 13px;
      color: #5e564d;
      min-height: 54px;
      margin-bottom: 10px;
      line-height: 1.5;
    }}
    video {{
      width: 100%;
      border-radius: 12px;
      background: #000;
    }}
    code {{
      background: rgba(0,0,0,0.06);
      padding: 2px 6px;
      border-radius: 6px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>train0706 dataset preview</h1>
    <div class="sub">OpenVid, raw phys-state, and Genesis rigid samples rendered from local dataset roots.</div>
    <div class="sub">Serve with: <code>python3 -m http.server {DEFAULT_PORTAL_PORT} --bind localhost --directory {output_root}</code></div>
    <div class="grid">
      {''.join(rows)}
    </div>
  </div>
</body>
</html>"""
    index_path = output_root / "index.html"
    index_path.write_text(page, encoding="utf-8")
    return index_path


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    _ensure_empty_dir(output_root)

    configs = [
        (
            "openvid",
            args.openvid_root,
            args.num_openvid,
            output_root / "openvid",
        ),
        (
            "raw_phys_state_video",
            args.raw_phys_root,
            args.num_raw_phys,
            output_root / "raw_phys_state_video",
        ),
        (
            "genesis_rigid",
            args.genesis_root,
            args.num_genesis,
            output_root / "genesis_rigid",
        ),
    ]

    all_cases: list[PreviewCase] = []
    for name, root, count, out_dir in configs:
        dataset = WanTI2VDataset(
            dataset_base_path=root,
            dataset_metadata_path="",
            dataset_repeat=1,
            height=384,
            width=672,
            num_frames=24,
        ).dataset
        all_cases.extend(_render_dataset(name, dataset, count, out_dir, seed=args.seed + len(all_cases)))

    index_path = _build_index_html(all_cases, output_root)
    summary = {
        "output_root": str(output_root),
        "index_path": str(index_path),
        "num_cases": len(all_cases),
        "port": int(args.port),
        "serve_command": f"python3 -m http.server {int(args.port)} --bind localhost --directory {output_root}",
        "cases": [case.__dict__ for case in all_cases],
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
