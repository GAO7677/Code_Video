from __future__ import annotations

import argparse
import html
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a local HTML gallery comparing original training clips and processed episode clips."
    )
    parser.add_argument("--data-root", required=True, help="Episode root with train/val directories.")
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-cases", type=int, default=16)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--display-long-side", type=int, default=480)
    return parser.parse_args()


def clean_text(text: object) -> str:
    return " ".join(str(text or "").strip().split())


def load_records(split_dir: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for json_path in sorted(split_dir.glob("*.json")):
        npz_path = json_path.with_suffix(".npz")
        if not npz_path.exists():
            continue
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        source = payload.get("source")
        if not isinstance(source, dict):
            continue
        full_video_path = clean_text(source.get("full_video_path"))
        if not full_video_path:
            continue
        if not Path(full_video_path).exists():
            continue
        records.append(
            {
                "case_id": json_path.stem,
                "json_path": json_path,
                "npz_path": npz_path,
                "payload": payload,
                "source": source,
                "full_video_path": Path(full_video_path),
                "source_label": clean_text(source.get("dataset")) or "Unknown",
                "prompt": clean_text(payload.get("prompt")),
                "categories": source.get("categories") if isinstance(source.get("categories"), list) else [],
            }
        )
    return records


def select_records(records: list[dict[str, object]], max_cases: int) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["source_label"])].append(record)
    ordered: list[dict[str, object]] = []
    labels = sorted(grouped.keys())
    while len(ordered) < max_cases:
        added = False
        for label in labels:
            bucket = grouped[label]
            if bucket:
                ordered.append(bucket.pop(0))
                added = True
                if len(ordered) >= max_cases:
                    break
        if not added:
            break
    return ordered


def read_original_segment(
    video_path: Path,
    start_frame: int,
    num_frames: int,
) -> np.ndarray:
    reader = imageio.get_reader(str(video_path))
    try:
        frames = []
        for frame_idx in range(start_frame, start_frame + num_frames):
            frames.append(np.asarray(reader.get_data(frame_idx), dtype=np.uint8))
        return np.stack(frames, axis=0)
    finally:
        reader.close()


def to_uint8_rgb(frame_chw: np.ndarray) -> np.ndarray:
    frame = np.transpose(np.clip(frame_chw, 0.0, 1.0), (1, 2, 0))
    return np.ascontiguousarray((frame * 255.0).round().astype(np.uint8))


def infer_display_hw(original_h: int, original_w: int, display_long_side: int) -> tuple[int, int]:
    scale = float(display_long_side) / float(max(original_h, original_w))
    return max(1, int(round(original_h * scale))), max(1, int(round(original_w * scale)))


def resize_video(frames_thwc: np.ndarray, display_hw: tuple[int, int]) -> np.ndarray:
    display_h, display_w = display_hw
    resized = [
        cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_CUBIC)
        for frame in frames_thwc
    ]
    return np.stack(resized, axis=0)


def build_strip(frames_thwc: np.ndarray, samples: int = 6) -> np.ndarray:
    if frames_thwc.shape[0] <= samples:
        indices = list(range(frames_thwc.shape[0]))
    else:
        indices = np.linspace(0, frames_thwc.shape[0] - 1, samples).round().astype(int).tolist()
    tiles = [frames_thwc[idx] for idx in indices]
    return np.concatenate(tiles, axis=1)


def save_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def write_mp4(path: Path, frames_thwc: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames_thwc.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer for {path}")
    for frame in frames_thwc:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def render_html(report: dict[str, object]) -> str:
    cards: list[str] = []
    for case in report["cases"]:
        cards.append(
            f"""
            <section class="case-card">
              <div class="head">
                <div>
                  <h2>{html.escape(case['case_id'])}</h2>
                  <div class="meta">
                    <span>{html.escape(case['source_label'])}</span>
                    <span>{html.escape(case['categories_text'])}</span>
                    <span>original={case['original_hw']}</span>
                    <span>train={case['train_hw']}</span>
                    <span>frames={case['num_frames']}</span>
                    <span>start={case['clip_start_frame']}</span>
                  </div>
                </div>
              </div>
              <div class="prompt">{html.escape(case['prompt'])}</div>
              <div class="video-grid">
                <div class="video-card">
                  <div class="title">原始训练视频片段</div>
                  <video controls preload="metadata" src="{case['original_video_rel']}"></video>
                  <div class="note">来自本地原视频，按训练时使用的 frame window 截取，但不做训练分辨率压缩。</div>
                </div>
                <div class="video-card">
                  <div class="title">处理后训练视频 `96x96`</div>
                  <video controls preload="metadata" src="{case['processed_video_rel']}"></video>
                  <div class="note">这是模型真正读入的 episode 张量，已经被压到训练分辨率。</div>
                </div>
                <div class="video-card">
                  <div class="title">处理后视频按原宽高比恢复显示</div>
                  <video controls preload="metadata" src="{case['processed_display_video_rel']}"></video>
                  <div class="note">像素仍然来自处理后张量，只是为了便于肉眼判断形变，把显示比例恢复到接近原视频。</div>
                </div>
              </div>
              <div class="strip-grid">
                <div class="strip-card">
                  <div class="title">原始关键帧条带</div>
                  <img src="{case['original_strip_rel']}" alt="original strip" />
                </div>
                <div class="strip-card">
                  <div class="title">处理后关键帧条带</div>
                  <img src="{case['processed_strip_rel']}" alt="processed strip" />
                </div>
              </div>
              <details>
                <summary>展开元信息</summary>
                <pre>{html.escape(case['metadata_json'])}</pre>
              </details>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>原视频 vs 处理后训练视频</title>
  <style>
    :root {{
      --bg: #f3ede4;
      --panel: rgba(255,255,255,0.88);
      --ink: #171613;
      --muted: #6d685f;
      --line: #ddd1c0;
      --accent: #1f5f54;
      --accent2: #b86a33;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(255,255,255,0.7), transparent 28%),
        linear-gradient(180deg, #f5efe3 0%, #eadfcd 100%);
      font-family: "Source Han Sans SC", "Noto Sans SC", sans-serif;
    }}
    .wrap {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 28px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 34px;
    }}
    .lead {{
      color: var(--muted);
      max-width: 1100px;
      line-height: 1.7;
      margin-bottom: 22px;
    }}
    .summary {{
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
      margin-bottom: 24px;
    }}
    .summary-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px 16px;
      min-width: 180px;
    }}
    .summary-card strong {{
      display: block;
      color: var(--accent);
      font-size: 28px;
    }}
    .summary-card span {{
      color: var(--muted);
      font-size: 13px;
    }}
    .case-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 18px;
      margin-bottom: 22px;
      box-shadow: 0 18px 50px rgba(77, 56, 24, 0.08);
    }}
    .head h2 {{
      margin: 0;
      font-size: 20px;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 8px;
    }}
    .meta span {{
      background: #efe4d2;
      color: var(--accent2);
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
    }}
    .prompt {{
      margin: 12px 0 16px;
      padding: 12px 14px;
      border-left: 4px solid var(--accent);
      background: rgba(255,250,242,0.82);
      line-height: 1.7;
      color: #2a2722;
    }}
    .video-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 16px;
    }}
    .video-card, .strip-card {{
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(255,255,255,0.72);
      padding: 12px;
    }}
    .title {{
      font-weight: 700;
      margin-bottom: 8px;
    }}
    .note {{
      color: var(--muted);
      font-size: 13px;
      margin-top: 8px;
      line-height: 1.6;
    }}
    video, img {{
      width: 100%;
      display: block;
      border-radius: 12px;
      background: #111;
    }}
    .strip-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 14px;
    }}
    details {{
      margin-top: 8px;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      line-height: 1.5;
      background: #faf4ea;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      overflow: auto;
    }}
    @media (max-width: 1100px) {{
      .video-grid {{
        grid-template-columns: 1fr;
      }}
      .strip-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>原始训练视频 vs 处理后训练视频</h1>
    <div class="lead">
      这里展示的是同一个训练样本的三种视图：
      原始本地视频中被截出的训练片段；
      模型真正读入的处理后训练视频；
      以及仅恢复显示宽高比后的处理后视频。
      这样可以直接看出压缩、拉伸、裁剪和细节损失来自哪里。
    </div>
    <div class="summary">
      <div class="summary-card"><strong>{report['case_count']}</strong><span>可对比样本数</span></div>
      <div class="summary-card"><strong>{report['split']}</strong><span>数据 split</span></div>
      <div class="summary-card"><strong>{report['source_label']}</strong><span>当前来源</span></div>
    </div>
    {''.join(cards)}
  </div>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    split_dir = Path(args.data_root) / args.split
    output_dir = Path(args.output_dir)
    assets_dir = output_dir / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(split_dir)
    selected = select_records(records, args.max_cases)
    cases: list[dict[str, object]] = []

    for record in selected:
        payload = record["payload"]
        source = record["source"]
        npz = np.load(record["npz_path"], allow_pickle=False)
        context_frames = np.asarray(npz["context_frames"], dtype=np.float32)
        future_frames = np.asarray(npz["future_frames"], dtype=np.float32)
        processed_frames = np.concatenate([context_frames, future_frames], axis=0)
        processed_rgb = np.stack([to_uint8_rgb(frame) for frame in processed_frames], axis=0)

        start_frame = int(source.get("clip_start_frame") or 0)
        num_frames = int(processed_rgb.shape[0])
        original_rgb = read_original_segment(record["full_video_path"], start_frame, num_frames)
        original_h, original_w = original_rgb.shape[1:3]
        display_h, display_w = infer_display_hw(original_h, original_w, args.display_long_side)

        original_display = resize_video(original_rgb, (display_h, display_w))
        processed_display = resize_video(processed_rgb, (display_h, display_w))

        case_id = str(record["case_id"])
        original_video = assets_dir / f"{case_id}__original.mp4"
        processed_video = assets_dir / f"{case_id}__processed.mp4"
        processed_display_video = assets_dir / f"{case_id}__processed_display.mp4"
        original_strip = assets_dir / f"{case_id}__original_strip.png"
        processed_strip = assets_dir / f"{case_id}__processed_strip.png"

        write_mp4(original_video, original_display, args.fps)
        write_mp4(processed_video, processed_rgb, args.fps)
        write_mp4(processed_display_video, processed_display, args.fps)
        save_png(original_strip, build_strip(original_display))
        save_png(processed_strip, build_strip(processed_display))

        cases.append(
            {
                "case_id": case_id,
                "source_label": record["source_label"],
                "categories_text": ", ".join(record["categories"]) if record["categories"] else "uncategorized",
                "prompt": record["prompt"],
                "original_hw": f"{original_h}x{original_w}",
                "train_hw": f"{processed_rgb.shape[1]}x{processed_rgb.shape[2]}",
                "num_frames": num_frames,
                "clip_start_frame": start_frame,
                "original_video_rel": str(original_video.relative_to(output_dir)),
                "processed_video_rel": str(processed_video.relative_to(output_dir)),
                "processed_display_video_rel": str(processed_display_video.relative_to(output_dir)),
                "original_strip_rel": str(original_strip.relative_to(output_dir)),
                "processed_strip_rel": str(processed_strip.relative_to(output_dir)),
                "metadata_json": json.dumps(payload, ensure_ascii=False, indent=2),
            }
        )

    report = {
        "split": args.split,
        "case_count": len(cases),
        "source_label": "OpenVidHD local raw clips",
        "cases": cases,
    }
    (output_dir / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "index.html").write_text(render_html(report), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "case_count": len(cases)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
