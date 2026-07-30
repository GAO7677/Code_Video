#!/usr/bin/env python3
"""Build a compact case-selectable gallery for the latest four LoRA variants."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import shutil
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--input-list", type=Path, required=True)
    parser.add_argument(
        "--complete-only",
        action="store_true",
        help="Publish only cases whose generated video exists for every method.",
    )
    parser.add_argument(
        "--method",
        action="append",
        required=True,
        help="Display label and generation directory as LABEL=DIRECTORY",
    )
    return parser.parse_args()


def parse_methods(values: list[str]) -> list[tuple[str, str]]:
    methods: list[tuple[str, str]] = []
    for value in values:
        label, separator, directory = value.partition("=")
        if not separator or not label.strip() or not directory.strip():
            raise ValueError(f"Invalid --method value: {value!r}")
        methods.append((label.strip(), directory.strip()))
    return methods


def render_gt_clip(source: Path, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        environment_ffmpeg = Path(sys.executable).with_name("ffmpeg")
        ffmpeg = str(environment_ffmpeg) if environment_ffmpeg.is_file() else None
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to render 49-frame GT clips")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            "fps=30",
            "-frames:v",
            "49",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(destination),
        ],
        check=True,
    )


def link_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        destination.unlink()
    destination.symlink_to(source.resolve())


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def build_page(
    cases: list[dict[str, object]],
    methods: list[tuple[str, str]],
    num_requested: int,
) -> str:
    options = "".join(
        f'<option value="case-{index}">{index + 1:02d} · {escape(case["stem"])}</option>'
        for index, case in enumerate(cases)
    )
    sections: list[str] = []
    for index, case in enumerate(cases):
        generated = []
        for method_index, (label, _) in enumerate(methods):
            video_path = case["methods"][method_index]
            generated.append(
                f"""
                <div class="video-cell">
                  <div class="video-label">{escape(label)}</div>
                  <video preload="metadata" playsinline muted src="{escape(video_path)}"></video>
                </div>
                """
            )
        sections.append(
            f"""
            <section class="case-view" id="case-{index}" data-case-index="{index}">
              <div class="case-heading">
                <div class="case-number">CASE {index + 1:02d}</div>
                <h2>{escape(case["stem"])}</h2>
                <p>{escape(case["prompt"])}</p>
              </div>
              <div class="source-row">
                <div class="video-cell source-cell">
                  <div class="video-label">GT · 49 frames @ 30 FPS</div>
                  <video preload="metadata" playsinline muted src="{escape(case["gt"])}"></video>
                </div>
                <div class="video-cell source-cell">
                  <div class="video-label">Input context · 8 frames</div>
                  <video preload="metadata" playsinline muted src="{escape(case["context"])}"></video>
                </div>
              </div>
              <div class="method-row">
                {''.join(generated)}
              </div>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>xSSC LoRA · test_5 权重对比</title>
  <style>
    :root {{
      --bg: #f4f6f7;
      --surface: #ffffff;
      --ink: #172126;
      --muted: #647178;
      --line: #d6dde0;
      --accent: #006d77;
      --accent-2: #9b3a31;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, "Noto Sans SC", Arial, sans-serif;
    }}
    .toolbar {{
      position: sticky;
      top: 0;
      z-index: 10;
      display: flex;
      align-items: center;
      gap: 10px;
      min-height: 58px;
      padding: 9px 18px;
      background: rgba(255, 255, 255, 0.96);
      border-bottom: 1px solid var(--line);
    }}
    .title {{
      margin-right: auto;
      font-size: 16px;
      font-weight: 750;
    }}
    select, button {{
      height: 38px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--surface);
      color: var(--ink);
      font: inherit;
    }}
    select {{
      width: min(560px, 48vw);
      padding: 0 10px;
    }}
    button {{
      width: 38px;
      padding: 0;
      cursor: pointer;
      font-size: 17px;
    }}
    button:hover {{ border-color: var(--accent); color: var(--accent); }}
    main {{
      max-width: 1860px;
      margin: 0 auto;
      padding: 18px;
    }}
    .case-view {{ display: none; }}
    .case-view.active {{ display: block; }}
    .case-heading {{
      padding: 4px 2px 16px;
      border-bottom: 1px solid var(--line);
    }}
    .case-number {{
      color: var(--accent);
      font-size: 12px;
      font-weight: 800;
    }}
    h2 {{
      margin: 6px 0;
      font-size: 20px;
      line-height: 1.3;
      overflow-wrap: anywhere;
    }}
    .case-heading p {{
      max-width: 1200px;
      margin: 0;
      color: var(--muted);
      line-height: 1.55;
    }}
    .source-row, .method-row {{
      display: grid;
      gap: 12px;
      margin-top: 14px;
    }}
    .source-row {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    .method-row {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
    .video-cell {{
      min-width: 0;
      padding: 9px;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 6px;
    }}
    .video-label {{
      min-height: 24px;
      padding: 1px 2px 7px;
      font-size: 13px;
      font-weight: 750;
      color: var(--accent-2);
    }}
    video {{
      display: block;
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: contain;
      background: #101416;
    }}
    @media (max-width: 1100px) {{
      .method-row {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .toolbar {{ flex-wrap: wrap; }}
      .title {{ width: 100%; }}
      select {{ width: calc(100% - 150px); }}
    }}
    @media (max-width: 650px) {{
      main {{ padding: 10px; }}
      .source-row, .method-row {{ grid-template-columns: 1fr; }}
      select {{ width: calc(100% - 144px); }}
    }}
  </style>
</head>
<body>
  <div class="toolbar">
    <div class="title">xSSC LoRA · test_5 对比 · {len(cases)}/{num_requested} cases</div>
    <select id="case-select" aria-label="选择案例">{options}</select>
    <button id="play" title="播放当前案例全部视频" aria-label="播放">▶</button>
    <button id="pause" title="暂停当前案例全部视频" aria-label="暂停">Ⅱ</button>
    <button id="replay" title="从头播放当前案例全部视频" aria-label="重新播放">↺</button>
  </div>
  <main>{''.join(sections)}</main>
  <script>
    const select = document.getElementById("case-select");
    const cases = [...document.querySelectorAll(".case-view")];
    function activeCase() {{
      return document.getElementById(select.value);
    }}
    function videos() {{
      return [...activeCase().querySelectorAll("video")];
    }}
    function showCase() {{
      cases.forEach(node => node.classList.toggle("active", node.id === select.value));
    }}
    select.addEventListener("change", () => {{
      cases.flatMap(node => [...node.querySelectorAll("video")]).forEach(video => video.pause());
      showCase();
    }});
    document.getElementById("play").addEventListener("click", () => {{
      videos().forEach(video => video.play().catch(() => {{}}));
    }});
    document.getElementById("pause").addEventListener("click", () => {{
      videos().forEach(video => video.pause());
    }});
    document.getElementById("replay").addEventListener("click", () => {{
      videos().forEach(video => {{ video.currentTime = 0; video.play().catch(() => {{}}); }});
    }});
    showCase();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    generation_root = run_root / "generations"
    gallery_root = run_root / "gallery"
    media_root = gallery_root / "media"
    methods = parse_methods(args.method)
    input_paths = [
        Path(line.strip())
        for line in args.input_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cases: list[dict[str, object]] = []
    for input_path in input_paths:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        stem = input_path.stem
        generated_sources = [
            generation_root / method_directory / f"{stem}.mp4"
            for _, method_directory in methods
        ]
        if args.complete_only and not all(path.is_file() for path in generated_sources):
            continue
        case_media = media_root / stem
        gt_path = case_media / "gt_49f_30fps.mp4"
        render_gt_clip(Path(payload["source_video"]), gt_path)
        context_path = case_media / "context_8f.mp4"
        link_file(Path(payload["input_video"]), context_path)
        method_paths: list[str] = []
        for method_index, source in enumerate(generated_sources):
            destination = case_media / f"method_{method_index:02d}.mp4"
            link_file(source, destination)
            method_paths.append(destination.relative_to(gallery_root).as_posix())
        cases.append(
            {
                "stem": stem,
                "prompt": payload["input_caption"],
                "gt": gt_path.relative_to(gallery_root).as_posix(),
                "context": context_path.relative_to(gallery_root).as_posix(),
                "methods": method_paths,
            }
        )
    gallery_root.mkdir(parents=True, exist_ok=True)
    (gallery_root / "index.html").write_text(
        build_page(cases, methods, len(input_paths)),
        encoding="utf-8",
    )
    manifest = {
        "num_requested": len(input_paths),
        "num_cases": len(cases),
        "num_pending": len(input_paths) - len(cases),
        "methods": [{"label": label, "directory": directory} for label, directory in methods],
        "cases": cases,
    }
    (gallery_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(gallery_root / "index.html")


if __name__ == "__main__":
    main()
