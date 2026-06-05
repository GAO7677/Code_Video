#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a local HTML gallery for dataset context/future/full videos."
    )
    parser.add_argument("--data-root", required=True, help="Episode root containing train/val/test folders.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", required=True, help="Gallery output directory.")
    parser.add_argument("--max-cases", type=int, default=8)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--port", type=int, default=18856)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--no-serve", action="store_true")
    return parser.parse_args()


def to_uint8_rgb(frame_chw: np.ndarray) -> np.ndarray:
    image = np.transpose(np.clip(frame_chw, 0.0, 1.0), (1, 2, 0))
    return np.ascontiguousarray((image * 255.0).round().astype(np.uint8))


def write_mp4(path: Path, frames_tchw: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    t, _, height, width = frames_tchw.shape
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer for {path}")
    for idx in range(t):
        rgb = to_uint8_rgb(frames_tchw[idx])
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    writer.release()


def build_strip(frames_tchw: np.ndarray, samples: int = 6) -> np.ndarray:
    frame_count = int(frames_tchw.shape[0])
    if frame_count <= samples:
        indices = list(range(frame_count))
    else:
        indices = np.linspace(0, frame_count - 1, samples).round().astype(int).tolist()
    tiles = [to_uint8_rgb(frames_tchw[idx]) for idx in indices]
    return np.concatenate(tiles, axis=1)


def save_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def choose_case_files(split_dir: Path, max_cases: int) -> list[Path]:
    files = sorted(split_dir.glob("*.npz"))
    if len(files) <= max_cases:
        return files
    selected: list[Path] = []
    seen_templates: set[str] = set()
    for path in files:
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        template_key = str(meta.get("template_key", "")).strip()
        if template_key and template_key not in seen_templates:
            selected.append(path)
            seen_templates.add(template_key)
            if len(selected) >= max_cases:
                return selected
    for path in files:
        if path in selected:
            continue
        selected.append(path)
        if len(selected) >= max_cases:
            break
    return selected


def is_port_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def start_server(output_dir: Path, port: int) -> int:
    log_path = output_dir / f"http_{port}.log"
    pid_path = output_dir / f"http_{port}.pid"
    if pid_path.exists():
        try:
            pid = int(pid_path.read_text().strip())
            os.kill(pid, 0)
            if is_port_open(port):
                return pid
        except Exception:
            pid_path.unlink(missing_ok=True)

    with open(log_path, "wb") as handle:
        proc = subprocess.Popen(
            ["python3", "-m", "http.server", str(port), "--bind", "127.0.0.1"],
            cwd=str(output_dir),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_path.write_text(str(proc.pid), encoding="utf-8")
    time.sleep(1.0)
    return proc.pid


def render_html(report: dict[str, object]) -> str:
    cards: list[str] = []
    for case in report["cases"]:
        cards.append(
            f"""
            <section class="case-card">
              <div class="card-head">
                <div>
                  <h2>{html.escape(case['case_id'])}</h2>
                  <div class="meta-line">
                    <span>split={html.escape(case['split'])}</span>
                    <span>template={html.escape(case['template_key'])}</span>
                    <span>frames={case['full_steps']} ({case['context_steps']}+{case['future_steps']})</span>
                    <span>objects={case['num_objects']}</span>
                    <span>hw={case['frame_hw']}</span>
                  </div>
                </div>
              </div>
              <div class="prompt">{html.escape(case['prompt'])}</div>
              <div class="video-grid">
                <div class="video-card">
                  <div class="title">Context Video</div>
                  <video controls preload="metadata" src="{case['context_video_rel']}"></video>
                </div>
                <div class="video-card">
                  <div class="title">Future Video</div>
                  <video controls preload="metadata" src="{case['future_video_rel']}"></video>
                </div>
                <div class="video-card">
                  <div class="title">Full Video</div>
                  <video controls preload="metadata" src="{case['full_video_rel']}"></video>
                </div>
              </div>
              <div class="strip-grid">
                <div class="strip-card">
                  <div class="title">Context Strip</div>
                  <img src="{case['context_strip_rel']}" alt="context strip" />
                </div>
                <div class="strip-card">
                  <div class="title">Future Strip</div>
                  <img src="{case['future_strip_rel']}" alt="future strip" />
                </div>
                <div class="strip-card">
                  <div class="title">Full Strip</div>
                  <img src="{case['full_strip_rel']}" alt="full strip" />
                </div>
              </div>
              <details class="raw-block">
                <summary>展开元信息</summary>
                <pre>{html.escape(case['metadata_json'])}</pre>
              </details>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>训练数据集视频 Gallery</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --panel: rgba(255, 252, 245, 0.92);
      --ink: #171613;
      --muted: #6d685f;
      --line: #d9cfbd;
      --accent: #1f5f54;
      --accent2: #b86a33;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(255,255,255,0.65), transparent 28%),
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
      letter-spacing: 0.02em;
    }}
    .lead {{
      max-width: 1080px;
      color: var(--muted);
      line-height: 1.7;
      margin-bottom: 18px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 20px;
    }}
    .summary-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px 16px;
    }}
    .summary-card strong {{
      display: block;
      font-size: 28px;
      color: var(--accent);
      margin-bottom: 4px;
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
    }}
    .card-head h2 {{
      margin: 0;
      font-size: 20px;
    }}
    .meta-line {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 8px;
    }}
    .meta-line span {{
      background: #efe4d2;
      color: var(--accent2);
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 12px;
    }}
    .prompt {{
      color: var(--muted);
      line-height: 1.7;
      margin: 10px 0 16px;
    }}
    .video-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 14px;
    }}
    .video-card, .strip-card {{
      background: #fcf8f2;
      border: 1px solid #eadfce;
      border-radius: 14px;
      padding: 12px;
    }}
    .title {{
      font-weight: 700;
      color: var(--accent);
      margin-bottom: 10px;
    }}
    .strip-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 12px;
    }}
    video, img {{
      width: 100%;
      display: block;
      border-radius: 12px;
      background: #000;
    }}
    .raw-block summary {{
      cursor: pointer;
      color: var(--accent);
      font-weight: 700;
    }}
    .raw-block pre {{
      margin: 12px 0 0;
      padding: 14px;
      border-radius: 12px;
      background: #f2ebdf;
      overflow-x: auto;
      white-space: pre-wrap;
      line-height: 1.55;
    }}
    @media (max-width: 1180px) {{
      .summary, .video-grid, .strip-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>训练数据集视频 Gallery</h1>
    <p class="lead">这页只看训练数据集本身，不看 predictor 或 Wan 输出。每个 episode 单独展示 `context video`、`future video` 和拼接后的 `full video`，方便检查数据切分是否合理、动作是否连续、prompt/template 是否和视频内容一致。</p>
    <section class="summary">
      <div class="summary-card"><strong>{report['split']}</strong><span>当前 split</span></div>
      <div class="summary-card"><strong>{report['case_count']}</strong><span>展示 case 数</span></div>
      <div class="summary-card"><strong>{report['context_steps']}</strong><span>context 帧数 K</span></div>
      <div class="summary-card"><strong>{report['future_steps']}</strong><span>future 帧数 T</span></div>
    </section>
    {''.join(cards)}
  </div>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    split_dir = data_root / args.split
    output_dir = Path(args.output_dir)
    assets_dir = output_dir / "assets"

    if args.clean and output_dir.exists():
        for child in output_dir.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
            else:
                for sub in sorted(child.rglob("*"), reverse=True):
                    if sub.is_file() or sub.is_symlink():
                        sub.unlink()
                    elif sub.is_dir():
                        sub.rmdir()
                child.rmdir()

    assets_dir.mkdir(parents=True, exist_ok=True)
    selected = choose_case_files(split_dir, args.max_cases)
    if not selected:
        raise FileNotFoundError(f"no .npz episodes found under {split_dir}")

    report_cases: list[dict[str, object]] = []
    context_steps = None
    future_steps = None
    for path in selected:
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        with np.load(path, allow_pickle=False) as payload:
            context_frames = payload["context_frames"].astype(np.float32)
            future_frames = payload["future_frames"].astype(np.float32)
        full_frames = np.concatenate([context_frames, future_frames], axis=0)

        context_steps = int(context_frames.shape[0]) if context_steps is None else context_steps
        future_steps = int(future_frames.shape[0]) if future_steps is None else future_steps
        case_id = path.stem

        context_video = assets_dir / f"{case_id}__context.mp4"
        future_video = assets_dir / f"{case_id}__future.mp4"
        full_video = assets_dir / f"{case_id}__full.mp4"
        context_strip = assets_dir / f"{case_id}__context_strip.png"
        future_strip = assets_dir / f"{case_id}__future_strip.png"
        full_strip = assets_dir / f"{case_id}__full_strip.png"

        write_mp4(context_video, context_frames, args.fps)
        write_mp4(future_video, future_frames, args.fps)
        write_mp4(full_video, full_frames, args.fps)
        save_png(context_strip, build_strip(context_frames))
        save_png(future_strip, build_strip(future_frames))
        save_png(full_strip, build_strip(full_frames))

        report_cases.append(
            {
                "case_id": case_id,
                "split": args.split,
                "template_key": str(meta.get("template_key", "unknown")),
                "prompt": str(meta.get("prompt", "")),
                "full_steps": int(full_frames.shape[0]),
                "context_steps": int(context_frames.shape[0]),
                "future_steps": int(future_frames.shape[0]),
                "num_objects": int(np.load(path, allow_pickle=False)["context_states"].shape[1]),
                "frame_hw": f"{int(full_frames.shape[-1])}x{int(full_frames.shape[-2])}",
                "context_video_rel": str(context_video.relative_to(output_dir)),
                "future_video_rel": str(future_video.relative_to(output_dir)),
                "full_video_rel": str(full_video.relative_to(output_dir)),
                "context_strip_rel": str(context_strip.relative_to(output_dir)),
                "future_strip_rel": str(future_strip.relative_to(output_dir)),
                "full_strip_rel": str(full_strip.relative_to(output_dir)),
                "metadata_json": json.dumps(meta, ensure_ascii=False, indent=2),
            }
        )

    report = {
        "split": args.split,
        "case_count": len(report_cases),
        "context_steps": context_steps,
        "future_steps": future_steps,
        "cases": report_cases,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "index.html").write_text(render_html(report), encoding="utf-8")
    if args.no_serve:
        print(json.dumps({"gallery": str(output_dir / "index.html"), **report}, ensure_ascii=False, indent=2))
        return
    pid = start_server(output_dir, args.port)
    print(json.dumps({"gallery": str(output_dir / "index.html"), "server": f"http://127.0.0.1:{args.port}", "pid": pid, **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
