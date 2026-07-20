#!/usr/bin/env python3
from argparse import ArgumentParser
import html
import json
from pathlib import Path
import pickle
import random
import sys

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DistributedSampler


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "third_party/dinov3"))
sys.path.insert(0, str(ROOT / "upstream"))

IMAGENET_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 1, 3, 1, 1)
IMAGENET_STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 1, 3, 1, 1)
PALETTE = np.array(
    [
        [0, 0, 0],
        [244, 67, 54],
        [33, 150, 243],
        [76, 175, 80],
        [255, 193, 7],
        [156, 39, 176],
        [0, 188, 212],
        [255, 112, 67],
        [121, 85, 72],
        [63, 81, 181],
    ],
    dtype=np.uint8,
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--cfg-file",
        type=Path,
        default=Path(
            "upstream/config-randsfq/"
            "rsfq2_r-ytvis_hq-dinov3_vitl16_256-slot512.py"
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("/data/gaoya/dataset"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--page-size", type=int, default=24)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def key_to_text(key):
    if isinstance(key, bytes):
        return key.decode("utf-8", errors="replace")
    return str(key)


def source_lengths(dataset, keys):
    from object_centric_bench.datum.dataset import lmdb_open_read

    env = lmdb_open_read(dataset.data_file)
    lengths = []
    with env.begin(write=False) as txn:
        for key in keys:
            sample = pickle.loads(txn.get(key))
            lengths.append(len(sample["video"]))
    env.close()
    return lengths


def make_overlay(rgb, segment):
    labels = segment.to(torch.uint8).argmax(dim=-1).cpu().numpy()
    color = PALETTE[labels % len(PALETTE)]
    foreground = labels > 0
    overlay = rgb.copy()
    overlay[foreground] = (
        rgb[foreground].astype(np.float32) * 0.55
        + color[foreground].astype(np.float32) * 0.45
    ).astype(np.uint8)
    return overlay


def save_clip(path, rgb, segment):
    frames = []
    for frame_id in range(rgb.shape[0]):
        frame = rgb[frame_id]
        overlay = make_overlay(frame, segment[frame_id])
        canvas = np.concatenate([frame, overlay], axis=1)
        frames.append(Image.fromarray(canvas))
    frames[0].save(
        path,
        format="WEBP",
        save_all=True,
        append_images=frames[1:],
        duration=400,
        loop=0,
        quality=82,
        method=4,
    )


def build_html(metadata, clips, page_size):
    cards = []
    for clip in clips:
        cards.append(
            f"""
            <article class="clip" data-card>
              <img src="clips/{clip['file']}" width="512" height="256"
                   alt="Training clip {clip['position']}" loading="lazy">
              <div class="meta">
                <strong>#{clip['position']:02d}</strong>
                <span>index {clip['dataset_index']}</span>
                <span>{clip['source_frames']} source frames</span>
                <span>{clip['foreground_instances']} fg instances</span>
              </div>
              <code>{html.escape(clip['source_key'])}</code>
            </article>
            """
        )
    cards_html = "\n".join(cards)
    title = (
        f"epoch {metadata['epoch']} | rank {metadata['rank']} | "
        f"batch {metadata['batch_index']} | {metadata['batch_shape'][0]} clips"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>xSSC training batch</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: #111315;
      color: #e8eaed;
      font: 14px system-ui, sans-serif;
      letter-spacing: 0;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 2;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 12px 20px;
      border-bottom: 1px solid #34383d;
      background: rgba(17, 19, 21, 0.96);
    }}
    h1 {{ margin: 0; font-size: 18px; font-weight: 650; }}
    .summary {{ color: #aeb4bc; white-space: nowrap; }}
    main {{ padding: 18px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      max-width: 1800px;
      margin: 0 auto;
    }}
    .clip {{
      min-width: 0;
      overflow: hidden;
      border: 1px solid #34383d;
      border-radius: 6px;
      background: #1b1e21;
    }}
    .clip[hidden] {{ display: none; }}
    .clip img {{
      display: block;
      width: 100%;
      height: auto;
      aspect-ratio: 2 / 1;
      object-fit: contain;
      background: #000;
    }}
    .meta {{
      display: grid;
      grid-template-columns: auto 1fr 1fr 1fr;
      gap: 8px;
      align-items: center;
      padding: 9px 10px 5px;
      color: #b9c0c8;
      font-size: 12px;
    }}
    .meta strong {{ color: #fff; }}
    code {{
      display: block;
      overflow: hidden;
      padding: 0 10px 10px;
      color: #7dd3a8;
      font-size: 11px;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    nav {{
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 12px;
      padding: 18px;
    }}
    button {{
      width: 36px;
      height: 36px;
      border: 1px solid #4a5057;
      border-radius: 6px;
      background: #24282c;
      color: #fff;
      font-size: 18px;
      cursor: pointer;
    }}
    button:disabled {{ cursor: default; color: #60666d; }}
    #page {{ min-width: 90px; text-align: center; color: #c8cdd3; }}
    @media (max-width: 1100px) {{ .grid {{ grid-template-columns: repeat(2, 1fr); }} }}
    @media (max-width: 650px) {{
      header {{ align-items: flex-start; flex-direction: column; }}
      .summary {{ white-space: normal; }}
      main {{ padding: 10px; }}
      .grid {{ grid-template-columns: 1fr; }}
      .meta {{ grid-template-columns: auto 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>xSSC training batch</h1>
    <div class="summary">{html.escape(title)} | RGB left, segmentation right</div>
  </header>
  <main><section class="grid">{cards_html}</section></main>
  <nav>
    <button id="prev" title="Previous page" aria-label="Previous page">&larr;</button>
    <span id="page"></span>
    <button id="next" title="Next page" aria-label="Next page">&rarr;</button>
  </nav>
  <script>
    const cards = [...document.querySelectorAll('[data-card]')];
    const pageSize = {page_size};
    const pageCount = Math.ceil(cards.length / pageSize);
    let page = 0;
    function render() {{
      cards.forEach((card, i) => card.hidden = Math.floor(i / pageSize) !== page);
      document.getElementById('page').textContent = `${{page + 1}} / ${{pageCount}}`;
      document.getElementById('prev').disabled = page === 0;
      document.getElementById('next').disabled = page + 1 === pageCount;
    }}
    document.getElementById('prev').onclick = () => {{ page--; render(); }};
    document.getElementById('next').onclick = () => {{ page++; render(); }};
    render();
  </script>
</body>
</html>
"""


def main():
    args = parse_args()
    from object_centric_bench.datum import DataLoader
    from object_centric_bench.util import Config, build_from_config

    config_file = args.cfg_file
    if not config_file.is_absolute():
        config_file = (ROOT / config_file).resolve()
    cfg = Config.fromfile(config_file)
    if args.world_size != cfg.expected_world_size:
        raise ValueError(
            f"world_size {args.world_size} != config world_size {cfg.expected_world_size}"
        )
    if not 0 <= args.rank < args.world_size:
        raise ValueError(f"invalid rank {args.rank} for world_size {args.world_size}")

    set_seed(args.seed + args.rank)
    cfg.dataset_t.base_dir = args.data_dir
    dataset = build_from_config(cfg.dataset_t)
    sampler = DistributedSampler(
        dataset,
        num_replicas=args.world_size,
        rank=args.rank,
        shuffle=True,
        seed=args.seed,
        drop_last=cfg.train_sampler_drop_last,
    )
    sampler.set_epoch(args.epoch)
    worker_seed = args.seed + args.rank

    def worker_init_fn(_):
        set_seed(worker_seed)

    generator = torch.Generator().manual_seed(worker_seed)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size_t,
        sampler=sampler,
        num_workers=cfg.num_work,
        collate_fn=build_from_config(cfg.collate_fn_t),
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
        drop_last=cfg.train_loader_drop_last,
    )
    try:
        batch = next(value for i, value in enumerate(loader) if i == args.batch_index)
    except StopIteration as error:
        raise IndexError(f"batch index {args.batch_index} is out of range") from error

    sampled_indices = list(iter(sampler))
    start = args.batch_index * cfg.batch_size_t
    batch_indices = sampled_indices[start : start + batch["video"].shape[0]]
    batch_keys = [dataset.keys[index] for index in batch_indices]
    lengths = source_lengths(dataset, batch_keys)

    video = (batch["video"] * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 255)
    video = video.round().to(torch.uint8).permute(0, 1, 3, 4, 2).cpu().numpy()
    segment = batch["segment"].cpu()
    expected_shape = (len(batch_indices), cfg.train_clip_frames, 3, *cfg.resolut0)
    if tuple(batch["video"].shape) != expected_shape:
        raise RuntimeError(
            f"unexpected training batch shape {tuple(batch['video'].shape)} != {expected_shape}"
        )

    output_dir = args.output_dir.resolve()
    clip_dir = output_dir / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    clips = []
    for position, (dataset_index, key, length) in enumerate(
        zip(batch_indices, batch_keys, lengths)
    ):
        file_name = f"clip_{position:03d}.webp"
        save_clip(clip_dir / file_name, video[position], segment[position])
        present = segment[position].any(dim=(0, 1, 2))
        foreground_instances = int(present.sum().item()) - int(present[0].item())
        clips.append(
            {
                "position": position,
                "file": file_name,
                "dataset_index": int(dataset_index),
                "source_key": key_to_text(key),
                "source_frames": int(length),
                "foreground_instances": max(0, foreground_instances),
            }
        )

    metadata = {
        "config": str(config_file),
        "data_dir": str(args.data_dir.resolve()),
        "seed": args.seed,
        "epoch": args.epoch,
        "rank": args.rank,
        "world_size": args.world_size,
        "batch_index": args.batch_index,
        "batch_shape": list(batch["video"].shape),
        "segment_shape": list(batch["segment"].shape),
        "normalized_video_min": float(batch["video"].min().item()),
        "normalized_video_max": float(batch["video"].max().item()),
        "normalized_video_mean": float(batch["video"].float().mean().item()),
        "clips": clips,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output_dir / "index.html").write_text(
        build_html(metadata, clips, args.page_size)
    )
    print(json.dumps({key: value for key, value in metadata.items() if key != "clips"}, indent=2))
    print(f"viewer: {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
