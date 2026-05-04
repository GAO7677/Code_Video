#!/usr/bin/env python3
"""Build a local comparison portal for current Genesis train window splitting."""

from __future__ import annotations

import html
import json
import subprocess
from pathlib import Path
from typing import Any

import imageio_ffmpeg


ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases")
STAGE1_ROOT = ROOT / "stage1adapter" / "train" / "genesis" / "rigid"
OUTPUT_DIR = Path("/home/gaoya/portal_hub_sim/stage1adapter_split_compare")

EXAMPLE_PATTERNS = [
    ("single_object_preview/count_01", "case900_random_parabola"),
    ("single_object_preview/count_01", "case003_static_highdrop"),
    ("interaction_pair_plus_dynamic/count_02", "case002_static_right"),
    ("interaction_pair_plus_dynamic/count_02", "case005_entry_left"),
    ("multi_object_free_motion/count_02", "case210_multi2_projectile_nocollision"),
    ("interaction_pair_plus_dynamic/count_03_04", "case000_static_center"),
]


def load_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def transcode_video(src: Path, dst: Path) -> str:
    if not src.exists():
        return ""
    ensure_dir(dst.parent)
    if dst.exists():
        return str(dst)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(src),
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0 or not dst.exists():
        return ""
    return str(dst)


def parse_case_name(sample_name: str) -> str:
    for part in sample_name.split("__")[1:]:
        if part.startswith("case"):
            return part
    return ""


def pick_examples() -> list[Path]:
    selected: list[Path] = []
    for subdir, needle in EXAMPLE_PATTERNS:
        root = STAGE1_ROOT / subdir
        found = None
        for meta_path in sorted(root.rglob("meta.json")):
            if needle in meta_path.parent.name:
                found = meta_path.parent
                break
        if found is not None:
            selected.append(found)
    return selected


def build_record(sample_dir: Path, index: int) -> dict[str, Any]:
    meta = load_json(sample_dir / "meta.json")
    pair_meta = load_json(sample_dir / "pair_meta.json")
    source_dir = Path(str((meta.get("source_paths") or {}).get("source_sample_dir", "")))
    source_meta_path = Path(str((meta.get("source_paths") or {}).get("source_meta_json_path", "")))
    source_meta = load_json(source_meta_path) if source_meta_path.exists() else {}
    source_video = source_dir / "videos" / "rgb.mp4"
    asset_prefix = f"sample_{index:02d}"

    media = []
    media_specs = [
        ("Source Raw", source_video),
        ("Window Full", Path(str((meta.get("paths") or {}).get("full_video_path", "")))),
        ("Context", Path(str((meta.get("paths") or {}).get("context_video_path", "")))),
        ("Future GT", Path(str((meta.get("paths") or {}).get("future_gt_video_path", "")))),
    ]
    for label, path in media_specs:
        if path and path.exists():
            dst = OUTPUT_DIR / "assets" / asset_prefix / f"{label.lower().replace(' ', '_')}.mp4"
            built = transcode_video(path, dst)
            media.append({"label": label, "path": built or str(path)})

    sel = pair_meta.get("selection_info") or {}
    source_total_frames = int(source_meta.get("frames", 0) or len(list((source_dir / "rgb").glob("frame_*.png"))))
    return {
        "sample_name": sample_dir.name,
        "sample_dir": str(sample_dir),
        "source_dir": str(source_dir),
        "dataset": str(meta.get("dataset", "")),
        "case_name": parse_case_name(sample_dir.name),
        "caption": str(source_meta.get("caption") or meta.get("caption") or ""),
        "detail_caption": str(source_meta.get("detail_caption") or ""),
        "object_count_bucket": str(source_meta.get("object_count_bucket") or ""),
        "collision_type_bucket": str(source_meta.get("collision_type_bucket") or ""),
        "collision_count_bucket": str(source_meta.get("collision_count_bucket") or ""),
        "source_total_frames": source_total_frames,
        "context_frames": int(meta.get("context_frames", 0) or 0),
        "future_frames": int(meta.get("future_frames", 0) or 0),
        "full_frames": int(meta.get("raw_frames", 0) or 0),
        "motion_complexity": str((meta.get("adapter_window") or {}).get("motion_complexity", "")),
        "segment_kind": str((meta.get("adapter_window") or {}).get("segment_kind", "")),
        "ratio_key": str((meta.get("adapter_window") or {}).get("ratio_key", "")),
        "selection_info": sel,
        "media": media,
    }


def build_html(records: list[dict[str, Any]]) -> str:
    cards = []
    for item in records:
        media_html = []
        for media in item["media"]:
            media_html.append(
                f"""
                <div class="media-block">
                  <div class="media-title">{html.escape(media['label'])}</div>
                  <video controls preload="metadata" playsinline src="{html.escape(media['path'])}"></video>
                </div>
                """
            )
        selection_text = json.dumps(item["selection_info"], ensure_ascii=False, indent=2)
        cards.append(
            f"""
            <article class="card">
              <div class="head">
                <div>
                  <h2>{html.escape(item['sample_name'])}</h2>
                  <div class="pills">
                    <span class="pill">{html.escape(item['dataset'])}</span>
                    <span class="pill">window</span>
                    <span class="pill">{html.escape(item['case_name'])}</span>
                  </div>
                </div>
              </div>
              <div class="caption-box">
                <div class="caption-title">Raw Caption</div>
                <p>{html.escape(item['caption'])}</p>
              </div>
              <div class="info-grid">
                <div class="info-box">
                  <div class="caption-title">Comparison Info</div>
                  <pre>source_dir: {html.escape(item['source_dir'])}
sample_dir: {html.escape(item['sample_dir'])}
object_count_bucket: {html.escape(item['object_count_bucket'])}
collision_type_bucket: {html.escape(item['collision_type_bucket'])}
collision_count_bucket: {html.escape(item['collision_count_bucket'])}
source_total_frames: {item['source_total_frames']}
window_full_frames: {item['full_frames']}
context_frames: {item['context_frames']}
future_frames: {item['future_frames']}
motion_complexity: {html.escape(item['motion_complexity'])}
segment_kind: {html.escape(item['segment_kind'])}
ratio_key: {html.escape(item['ratio_key'])}</pre>
                </div>
                <div class="info-box">
                  <div class="caption-title">Current Split</div>
                  <pre>{html.escape(selection_text)}</pre>
                </div>
              </div>
              <div class="note">
                对比含义：
                <code>Source Raw</code> 是原始整段视频；
                <code>Window Full</code> 是当前导出的无碰撞窗口；
                <code>Context/Future GT</code> 是在这段窗口上按固定 ratio 切开的结果。
              </div>
              <div class="media-grid">{''.join(media_html)}</div>
            </article>
            """
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stage1adapter Split Compare</title>
  <style>
    body {{
      margin: 0;
      font-family: Arial, sans-serif;
      background: #f2efe8;
      color: #181512;
    }}
    .wrap {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 18px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
    }}
    .sub {{
      margin: 0 0 16px;
      color: #5f584e;
      line-height: 1.5;
    }}
    .card {{
      background: #fffdf8;
      border: 1px solid #ddd4c7;
      border-radius: 12px;
      padding: 14px;
      margin-bottom: 16px;
    }}
    .head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }}
    h2 {{
      margin: 0 0 6px;
      font-size: 18px;
      word-break: break-all;
    }}
    .pills {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .pill {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      background: #ece6da;
      font-size: 12px;
      color: #544b40;
    }}
    .caption-box, .info-box {{
      background: #faf7f0;
      border: 1px solid #e6ded0;
      border-radius: 8px;
      padding: 10px;
    }}
    .caption-title, .media-title {{
      font-weight: 700;
      font-size: 12px;
      color: #645a4c;
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .caption-box p {{
      margin: 0;
      line-height: 1.55;
    }}
    .info-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 10px;
    }}
    .info-box pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      line-height: 1.45;
      font-size: 12px;
      color: #302a23;
    }}
    .note {{
      margin-top: 10px;
      color: #5d5448;
      line-height: 1.5;
      font-size: 13px;
    }}
    .media-grid {{
      margin-top: 12px;
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
    }}
    .media-block video {{
      width: 100%;
      display: block;
      border-radius: 8px;
      background: #000;
    }}
    code {{
      background: #eee7da;
      padding: 1px 4px;
      border-radius: 4px;
    }}
    @media (max-width: 1100px) {{
      .media-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      .info-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Stage1adapter Split Compare</h1>
    <p class="sub">
      这个页面对比当前 Genesis train window 的切分方式。每个样本同时展示原始 source video、当前导出的 full window，以及固定切分得到的 context 和 future GT。
    </p>
    {''.join(cards)}
  </div>
</body>
</html>"""


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    records = [build_record(sample_dir, idx) for idx, sample_dir in enumerate(pick_examples())]
    (OUTPUT_DIR / "index.html").write_text(build_html(records), encoding="utf-8")
    print(OUTPUT_DIR / "index.html")


if __name__ == "__main__":
    main()
