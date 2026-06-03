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


def parse_args():
    parser = argparse.ArgumentParser(description="Watch adapter checkpoints and export case visualizations.")
    parser.add_argument("--env-py", required=True, help="Python executable used to run export_trained_cases.py.")
    parser.add_argument("--project-root", required=True, help="Project root containing scripts/export_trained_cases.py.")
    parser.add_argument("--episode-root", required=True)
    parser.add_argument("--predictor-checkpoint", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--port", type=int, default=18833)
    parser.add_argument("--poll-seconds", type=int, default=180)
    parser.add_argument("--max-cases", type=int, default=12)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--splits", nargs="+", default=["val", "test"])
    return parser.parse_args()


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


def checkpoint_signature(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_size}"


def render_index(items: list[dict], port: int) -> str:
    cards = []
    for item in items:
        cards.append(
            f"""
            <article class="card">
              <div class="eyebrow">{html.escape(item['kind'])}</div>
              <h3>{html.escape(item['name'])}</h3>
              <p class="meta">mtime: {html.escape(item['mtime_text'])}</p>
              <p class="meta">checkpoint: {html.escape(item['checkpoint'])}</p>
              <a class="link" href="{html.escape(item['rel'])}/index.html">打开这个 ckpt 的 case 页面</a>
            </article>
            """
        )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>训练中 ckpt 可视化</title>
  <style>
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Source Han Sans SC", "Noto Sans SC", sans-serif;
      color: #1f1f1c;
      background: linear-gradient(180deg, #f8f4ec 0%, #eee3d5 100%);
    }}
    .page {{
      max-width: 1400px;
      margin: 0 auto;
      padding: 28px;
    }}
    .hero, .card {{
      background: rgba(255, 252, 247, 0.94);
      border: 1px solid #dbcdbd;
      border-radius: 18px;
    }}
    .hero {{
      padding: 22px;
      margin-bottom: 18px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 14px;
    }}
    .card {{
      padding: 16px;
    }}
    h1, h3 {{
      margin-top: 0;
    }}
    .eyebrow {{
      color: #b8642a;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin-bottom: 8px;
    }}
    .meta {{
      color: #6e675d;
      line-height: 1.6;
      word-break: break-all;
    }}
    .link {{
      color: #0f5a52;
      font-weight: 700;
      text-decoration: none;
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>训练中 ckpt 可视化总览</h1>
      <p>这个页面会持续收集训练过程中导出的不同 adapter checkpoint case 可视化。根目录地址：<a href="http://127.0.0.1:{port}">http://127.0.0.1:{port}</a></p>
    </section>
    <section class="grid">
      {''.join(cards)}
    </section>
  </div>
</body>
</html>"""


def main():
    args = parse_args()
    project_root = Path(args.project_root)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_root = Path(args.output_root)
    cases_root = output_root / "cases"
    output_root.mkdir(parents=True, exist_ok=True)
    cases_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "watch_state.json"
    state = {}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            state = {}

    export_script = project_root / "scripts" / "export_trained_cases.py"
    start_server(output_root, args.port)

    while True:
        predictor_ckpt = Path(args.predictor_checkpoint)
        if not predictor_ckpt.exists():
            print(f"[watch] predictor checkpoint not ready: {predictor_ckpt}", flush=True)
            time.sleep(args.poll_seconds)
            continue

        candidates = sorted(
            [path for path in checkpoint_dir.glob("adapter*.pt") if path.is_file()],
            key=lambda path: path.stat().st_mtime,
        )
        items = []
        for ckpt in candidates:
            signature = checkpoint_signature(ckpt)
            prev_sig = state.get(str(ckpt))
            export_dir = cases_root / ckpt.stem
            if prev_sig != signature:
                cmd = [
                    args.env_py,
                    str(export_script),
                    "--episode-root",
                    args.episode_root,
                    "--predictor",
                    str(predictor_ckpt),
                    "--adapter",
                    str(ckpt),
                    "--output-dir",
                    str(export_dir),
                    "--max-cases",
                    str(args.max_cases),
                    "--fps",
                    str(args.fps),
                    "--device",
                    args.device,
                    "--no-serve",
                ]
                if export_dir.exists():
                    cmd.append("--clean")
                cmd.extend(["--splits", *args.splits])
                print(f"[watch] exporting {ckpt.name}", flush=True)
                result = subprocess.run(cmd, cwd=str(project_root))
                if result.returncode == 0:
                    state[str(ckpt)] = signature
                    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
                else:
                    print(f"[watch] export failed for {ckpt}", flush=True)

            items.append(
                {
                    "name": ckpt.stem,
                    "kind": "best" if "best" in ckpt.stem else "snapshot",
                    "checkpoint": str(ckpt),
                    "mtime_text": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ckpt.stat().st_mtime)),
                    "rel": f"cases/{ckpt.stem}",
                }
            )

        items.sort(key=lambda item: item["mtime_text"], reverse=True)
        (output_root / "index.html").write_text(render_index(items, args.port), encoding="utf-8")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
