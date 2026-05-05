#!/usr/bin/env python3
"""Build a compact local GIF portal for sum0504 train no_collision/env_only samples."""

from __future__ import annotations

import argparse
import html
import json
import shutil
import socket
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
from PIL import Image, ImageOps


DEFAULT_SUMMARY_ROOT = Path("/home/gaoya/Code_Video/Code_data/data0417/data_summary/sum0504")
DEFAULT_OUTPUT_ROOT = Path("/home/gaoya/portal_hub_sim/sum0504_train_gif_portal")
DEFAULT_PORT = 8049
KEEP_COLLISIONS = {"no_collision", "env_only"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--dataset_filter", type=str, default="genesis", choices=["all", "genesis", "movi-d"])
    parser.add_argument("--max_per_leaf", type=int, default=60)
    parser.add_argument("--gif_max_frames", type=int, default=16)
    parser.add_argument("--gif_width", type=int, default=256)
    parser.add_argument("--gif_duration_ms", type=int, default=110)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def sanitize_token(text: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text))
    return token.strip("_") or "item"


def collect_leaf_samples(summary_root: Path, max_per_leaf: int) -> list[dict[str, Any]]:
    train_root = summary_root / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(f"Missing train split under {summary_root}")

    records: list[dict[str, Any]] = []
    for samples_txt in sorted(train_root.rglob("samples.txt")):
        collision = samples_txt.parent.name
        if collision not in KEEP_COLLISIONS:
            continue
        count_bucket = samples_txt.parent.parent.name
        simulator = samples_txt.parent.parent.parent.name
        lines = [
            line.strip()
            for line in samples_txt.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        for sample_dir_text in lines:
            sample_dir = Path(sample_dir_text)
            if not sample_dir.is_dir():
                continue
            sample_dir_lower = str(sample_dir).lower()
            if "movi" in sample_dir_lower:
                dataset = "movi-d"
            else:
                dataset = "genesis"
            pair_meta_path = sample_dir / "pair_meta.json"
            metadata_path = sample_dir / "metadata.json"
            meta_path = sample_dir / "meta.json"
            if not pair_meta_path.is_file() and not metadata_path.is_file() and not meta_path.is_file():
                continue
            if pair_meta_path.is_file():
                meta = load_json(pair_meta_path)
            elif metadata_path.is_file():
                meta = load_json(metadata_path)
            else:
                meta = load_json(meta_path)
            motion = str((meta.get("motion_complexity") or {}).get("label", "unknown"))
            future_len = int(meta.get("future_len", 0)) if str(meta.get("future_len", "")).strip() else 0
            context_len = int(meta.get("context_len", 0)) if str(meta.get("context_len", "")).strip() else 0
            prompt = str(meta.get("prompt", "")).strip()
            if not prompt and (sample_dir / "caption_simple.txt").is_file():
                prompt = (sample_dir / "caption_simple.txt").read_text(encoding="utf-8").strip()
            if not prompt and (sample_dir / "caption.txt").is_file():
                prompt = (sample_dir / "caption.txt").read_text(encoding="utf-8").strip()
            records.append(
                {
                    "sample_dir": sample_dir,
                    "sample_name": sample_dir.name,
                    "pair_meta_path": pair_meta_path if pair_meta_path.is_file() else None,
                    "metadata_path": metadata_path if metadata_path.is_file() else None,
                    "meta_path": meta_path if meta_path.is_file() else None,
                    "dataset": dataset,
                    "simulator": simulator,
                    "count_bucket": count_bucket,
                    "collision": collision,
                    "motion": motion,
                    "future_len": future_len,
                    "context_len": context_len,
                    "prompt": prompt,
                    "leaf_key": f"{simulator}/{count_bucket}/{collision}",
                }
            )
    return records


def balance_records_by_dataset(records: list[dict[str, Any]], max_per_leaf: int) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        grouped[str(record["leaf_key"])][str(record["dataset"])].append(record)

    selected: list[dict[str, Any]] = []
    per_leaf_limit = max(1, int(max_per_leaf))
    for leaf_key in sorted(grouped):
        by_dataset = grouped[leaf_key]
        movi_items = list(by_dataset.get("movi-d", []))
        genesis_items = list(by_dataset.get("genesis", []))
        combined: list[dict[str, Any]] = []

        if movi_items:
            movi_quota = max(1, min(len(movi_items), per_leaf_limit // 2 or 1))
            combined.extend(movi_items[:movi_quota])
        remaining = per_leaf_limit - len(combined)
        if remaining > 0:
            combined.extend(genesis_items[:remaining])
        remaining = per_leaf_limit - len(combined)
        if remaining > 0 and movi_items:
            combined.extend(movi_items[len([item for item in combined if item["dataset"] == "movi-d"]): len([item for item in combined if item["dataset"] == "movi-d"]) + remaining])

        selected.extend(combined[:per_leaf_limit])
    return selected


def filter_records_by_dataset(records: list[dict[str, Any]], dataset_filter: str) -> list[dict[str, Any]]:
    dataset_filter = str(dataset_filter).strip().lower()
    if dataset_filter in {"", "all"}:
        return records
    return [record for record in records if str(record.get("dataset", "")).lower() == dataset_filter]


def sample_video_path(sample_dir: Path) -> Path | None:
    for name in ("full_video.mp4", "videos/rgb.mp4", "context_video.mp4", "future_gt_video.mp4"):
        path = sample_dir / name
        if path.is_file():
            return path
    return None


def sample_rgb_dir(sample_dir: Path) -> Path | None:
    rgb_dir = sample_dir / "rgb"
    if rgb_dir.is_dir():
        return rgb_dir
    return None


def resize_for_gif(image: Image.Image, target_width: int) -> Image.Image:
    if target_width <= 0:
        return image.convert("RGB")
    rgb = image.convert("RGB")
    if rgb.width <= target_width:
        return rgb
    target_height = max(1, int(round(rgb.height * float(target_width) / float(rgb.width))))
    return ImageOps.contain(rgb, (int(target_width), int(target_height)), method=Image.Resampling.LANCZOS)


def build_gif_from_video(
    video_path: Path,
    gif_path: Path,
    *,
    max_frames: int,
    target_width: int,
    duration_ms: int,
) -> bool:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, frame_count // max_frames) if frame_count > max_frames and max_frames > 0 else 1
    frames: list[Image.Image] = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = resize_for_gif(Image.fromarray(rgb), target_width=target_width)
            frames.append(pil)
            if max_frames > 0 and len(frames) >= max_frames:
                break
        frame_idx += 1
    cap.release()
    if not frames:
        return False
    ensure_dir(gif_path.parent)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(duration_ms),
        loop=0,
        optimize=False,
    )
    return True


def build_gif_from_rgb_dir(
    rgb_dir: Path,
    gif_path: Path,
    *,
    max_frames: int,
    target_width: int,
    duration_ms: int,
) -> bool:
    frame_paths = sorted(rgb_dir.glob("frame_*.png"))
    if not frame_paths:
        return False
    step = max(1, len(frame_paths) // max_frames) if len(frame_paths) > max_frames and max_frames > 0 else 1
    frames: list[Image.Image] = []
    for idx, frame_path in enumerate(frame_paths):
        if idx % step != 0:
            continue
        with Image.open(frame_path) as image:
            frames.append(resize_for_gif(image.copy(), target_width=target_width))
        if max_frames > 0 and len(frames) >= max_frames:
            break
    if not frames:
        return False
    ensure_dir(gif_path.parent)
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(duration_ms),
        loop=0,
        optimize=False,
    )
    return True


def export_assets(
    records: list[dict[str, Any]],
    output_root: Path,
    *,
    gif_max_frames: int,
    gif_width: int,
    gif_duration_ms: int,
) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        sample_dir = Path(record["sample_dir"])
        video_path = sample_video_path(sample_dir)
        rgb_dir = sample_rgb_dir(sample_dir)
        if video_path is None and rgb_dir is None:
            continue
        sample_slug = f"{index:04d}_{sanitize_token(record['sample_name'])}"
        asset_dir = output_root / "assets" / sanitize_token(record["leaf_key"]) / sample_slug
        ensure_dir(asset_dir)
        gif_path = asset_dir / "full.gif"
        if not gif_path.exists():
            if video_path is not None:
                ok = build_gif_from_video(
                    video_path,
                    gif_path,
                    max_frames=int(gif_max_frames),
                    target_width=int(gif_width),
                    duration_ms=int(gif_duration_ms),
                )
            else:
                ok = build_gif_from_rgb_dir(
                    rgb_dir,
                    gif_path,
                    max_frames=int(gif_max_frames),
                    target_width=int(gif_width),
                    duration_ms=int(gif_duration_ms),
                )
            if not ok:
                continue
        meta_src = record.get("pair_meta_path") or record.get("metadata_path") or record.get("meta_path")
        meta_name = Path(str(meta_src)).name if meta_src is not None else "meta.json"
        meta_link = asset_dir / meta_name
        if meta_src is not None and not meta_link.exists():
            shutil.copy2(meta_src, meta_link)
        cards.append(
            {
                **record,
                "gif_rel": gif_path.relative_to(output_root).as_posix(),
                "meta_rel": meta_link.relative_to(output_root).as_posix(),
                "sample_dir_text": str(sample_dir),
                "search_text": " ".join(
                    [
                        str(record["sample_name"]),
                        str(record["leaf_key"]),
                        str(record["motion"]),
                        str(record["prompt"]),
                    ]
                ).lower(),
            }
        )
    return cards


def build_index(cards: list[dict[str, Any]]) -> str:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for card in cards:
        grouped[f"{card['dataset']} | {card['leaf_key']}"].append(card)

    sections: list[str] = []
    section_options: list[str] = ['<option value="">全部分组</option>']
    for leaf_key, items in sorted(grouped.items()):
        section_slug = sanitize_token(leaf_key)
        section_options.append(f'<option value="{html.escape(section_slug)}">{html.escape(leaf_key)} ({len(items)})</option>')
        cards_html = []
        for item in items:
            prompt_text = html.escape(str(item.get("prompt") or ""))
            motion = html.escape(str(item["motion"]))
            sample_name = html.escape(str(item["sample_name"]))
            sample_dir = html.escape(str(item["sample_dir_text"]))
            count_bucket = html.escape(str(item["count_bucket"]))
            collision = html.escape(str(item["collision"]))
            dataset = html.escape(str(item["dataset"]))
            meta_rel = html.escape(str(item["meta_rel"]))
            gif_rel = html.escape(str(item["gif_rel"]))
            search_text = html.escape(str(item["search_text"]))
            cards_html.append(
                f"""
                <article class="card" data-section="{html.escape(section_slug)}" data-search="{search_text}">
                  <img loading="lazy" src="{gif_rel}" alt="{sample_name}">
                  <div class="body">
                    <div class="title">{sample_name}</div>
                    <div class="chips">
                      <span>{dataset}</span>
                      <span>{count_bucket}</span>
                      <span>{collision}</span>
                      <span>{motion}</span>
                      <span>ctx {int(item['context_len'])}</span>
                      <span>fut {int(item['future_len'])}</span>
                    </div>
                    <div class="prompt">{prompt_text}</div>
                    <div class="path">{sample_dir}</div>
                    <div class="links"><a href="{meta_rel}">pair_meta.json</a></div>
                  </div>
                </article>
                """
            )
        sections.append(
            f"""
            <section class="section" data-section="{html.escape(section_slug)}">
              <div class="section-head">
                <h2>{html.escape(leaf_key)}</h2>
                <span>{len(items)} samples</span>
              </div>
              <div class="grid">
                {''.join(cards_html)}
              </div>
            </section>
            """
        )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>sum0504 train gif portal</title>
  <style>
    :root {{
      --bg:#f4efe7;
      --panel:#fffdf9;
      --line:#d7cec1;
      --ink:#1f1a17;
      --muted:#6e655d;
      --accent:#8b5e34;
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0;
      color:var(--ink);
      font-family:"Iowan Old Style","Palatino Linotype","Noto Serif SC",serif;
      background:
        radial-gradient(circle at top left, rgba(139,94,52,.12), transparent 28rem),
        linear-gradient(180deg, #f8f4ee 0%, var(--bg) 100%);
    }}
    main {{
      width:min(1860px, calc(100vw - 12px));
      margin:0 auto;
      padding:8px 0 20px;
    }}
    .hero {{
      position:sticky;
      top:6px;
      z-index:5;
      background:rgba(255,253,249,.92);
      backdrop-filter:blur(8px);
      border:1px solid var(--line);
      border-radius:16px;
      padding:10px 12px;
    }}
    h1 {{
      margin:0;
      font-size:1.45rem;
      line-height:1.05;
    }}
    .sub {{
      margin-top:4px;
      color:var(--muted);
      font-size:.88rem;
    }}
    .toolbar {{
      display:flex;
      flex-wrap:wrap;
      gap:8px;
      margin-top:8px;
    }}
    .toolbar input, .toolbar select {{
      border:1px solid var(--line);
      border-radius:10px;
      padding:7px 10px;
      background:#fff;
      color:var(--ink);
      font:inherit;
    }}
    .toolbar input {{ min-width:320px; flex:1; }}
    .sections {{
      display:grid;
      gap:10px;
      margin-top:10px;
    }}
    .section {{
      background:var(--panel);
      border:1px solid var(--line);
      border-radius:16px;
      padding:10px;
    }}
    .section-head {{
      display:flex;
      justify-content:space-between;
      align-items:baseline;
      gap:10px;
      margin-bottom:8px;
    }}
    .section-head h2 {{
      margin:0;
      font-size:1rem;
      line-height:1.1;
    }}
    .section-head span {{
      color:var(--muted);
      font-size:.8rem;
    }}
    .grid {{
      display:grid;
      grid-template-columns:repeat(auto-fill, minmax(240px, 1fr));
      gap:8px;
    }}
    .card {{
      border:1px solid var(--line);
      border-radius:12px;
      overflow:hidden;
      background:#fff;
      min-width:0;
    }}
    .card img {{
      width:100%;
      display:block;
      aspect-ratio:16/9;
      object-fit:cover;
      background:#eae2d7;
    }}
    .body {{
      padding:7px 8px 8px;
    }}
    .title {{
      font-size:.83rem;
      line-height:1.15;
      font-weight:700;
      overflow-wrap:anywhere;
    }}
    .chips {{
      display:flex;
      flex-wrap:wrap;
      gap:4px;
      margin-top:5px;
    }}
    .chips span {{
      border:1px solid var(--line);
      border-radius:999px;
      padding:2px 6px;
      font-size:.69rem;
      color:var(--muted);
      background:#fcf7f0;
      white-space:nowrap;
    }}
    .prompt {{
      margin-top:6px;
      font-size:.72rem;
      line-height:1.28;
      color:var(--ink);
      display:-webkit-box;
      -webkit-line-clamp:2;
      -webkit-box-orient:vertical;
      overflow:hidden;
    }}
    .path {{
      margin-top:6px;
      font-size:.66rem;
      line-height:1.25;
      color:#7a6d60;
      overflow-wrap:anywhere;
    }}
    .links {{
      margin-top:6px;
      font-size:.7rem;
    }}
    a {{
      color:var(--accent);
      text-decoration:none;
    }}
    a:hover {{
      text-decoration:underline;
    }}
    .hidden {{
      display:none !important;
    }}
    @media (max-width: 900px) {{
      main {{ width:min(100vw - 8px, 1860px); }}
      .hero {{ position:static; }}
      .toolbar input {{ min-width:0; width:100%; }}
      .grid {{ grid-template-columns:repeat(auto-fill, minmax(180px, 1fr)); }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>sum0504 train GIF portal</h1>
      <div class="sub">只展示 train split 下的 <code>no_collision</code> 和 <code>env_only</code>。卡片压缩排版，每个样本生成一个本地 full GIF。</div>
      <div class="toolbar">
        <input id="searchBox" type="search" placeholder="搜索 sample / bucket / prompt">
        <select id="sectionFilter">
          {''.join(section_options)}
        </select>
      </div>
    </section>
    <div class="sections">
      {''.join(sections)}
    </div>
  </main>
  <script>
    const searchBox = document.getElementById('searchBox');
    const sectionFilter = document.getElementById('sectionFilter');
    const cards = Array.from(document.querySelectorAll('.card'));
    const sections = Array.from(document.querySelectorAll('.section'));
    function applyFilter() {{
      const query = (searchBox.value || '').trim().toLowerCase();
      const section = sectionFilter.value || '';
      cards.forEach((card) => {{
        const okSection = !section || card.dataset.section === section;
        const okQuery = !query || (card.dataset.search || '').includes(query);
        card.classList.toggle('hidden', !(okSection && okQuery));
      }});
      sections.forEach((sec) => {{
        const visible = sec.querySelector('.card:not(.hidden)');
        sec.classList.toggle('hidden', !visible);
      }});
    }}
    searchBox.addEventListener('input', applyFilter);
    sectionFilter.addEventListener('change', applyFilter);
  </script>
</body>
</html>
"""


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def start_server(output_root: Path, host: str, port: int) -> tuple[int, str]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, port))
    process = subprocess.Popen(
        [
            "python3",
            "-m",
            "http.server",
            str(port),
            "--bind",
            str(host),
        ],
        cwd=str(output_root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return process.pid, f"http://{host}:{port}/index.html"


def main() -> None:
    args = parse_args()
    if args.rebuild and args.output_root.exists():
        shutil.rmtree(args.output_root)
    ensure_dir(args.output_root)

    records = collect_leaf_samples(args.summary_root, max_per_leaf=int(args.max_per_leaf))
    records = filter_records_by_dataset(records, dataset_filter=str(args.dataset_filter))
    if str(args.dataset_filter) == "all":
        records = balance_records_by_dataset(records, max_per_leaf=int(args.max_per_leaf))
    cards = export_assets(
        records,
        args.output_root,
        gif_max_frames=int(args.gif_max_frames),
        gif_width=int(args.gif_width),
        gif_duration_ms=int(args.gif_duration_ms),
    )
    html_text = build_index(cards)
    (args.output_root / "index.html").write_text(html_text, encoding="utf-8")
    write_json(
        args.output_root / "build_summary.json",
        {
            "summary_root": str(args.summary_root),
            "output_root": str(args.output_root),
            "dataset_filter": str(args.dataset_filter),
            "num_records": len(records),
            "num_cards": len(cards),
            "kept_collisions": sorted(KEEP_COLLISIONS),
            "port": int(args.port),
        },
    )
    pid, url = start_server(args.output_root, str(args.host), int(args.port))
    print(json.dumps({"pid": pid, "url": url, "output_root": str(args.output_root), "num_cards": len(cards)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
