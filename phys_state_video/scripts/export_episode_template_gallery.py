#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phys_state_video.schemas import StateIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export episode gallery sampled by template/family.")
    parser.add_argument("--data-root", required=True, help="Episode root containing train/val/test.")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-template", type=int, default=3)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--serve-port", type=int, default=18886)
    return parser.parse_args()


def clean_text(text: str | object) -> str:
    return " ".join(str(text or "").strip().split())


def template_family(template_key: str) -> str:
    parts = clean_text(template_key).split("_")
    if len(parts) >= 2:
        return parts[1].upper()
    return "UNKNOWN"


def find_ffmpeg() -> str:
    candidates = [
        shutil.which("ffmpeg"),
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/home/gaoya/miniconda3/envs/wan/bin/ffmpeg",
        "/home/gaoya/miniconda3/envs/wan-cu128/bin/ffmpeg",
        "/data/gaoya/miniconda3/envs/wan/bin/ffmpeg",
        "/data/gaoya/miniconda3/envs/vjepa2/bin/ffmpeg",
        "/data/gaoya/home_miniconda3/pkgs/ffmpeg-8.0.0-gpl_hc3e963e_905/bin/ffmpeg",
        "/home/gaoya/.marscode/ai-chat/binary/1.6.38/modules/ai-agent/ffmpeg",
        "/home/gaoya/.marscode/ai-chat/binary/1.6.36/modules/ai-agent/ffmpeg",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise RuntimeError("ffmpeg not found")


def load_records(split_dir: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for npz_path in sorted(split_dir.glob("*.npz")):
        json_path = npz_path.with_suffix(".json")
        meta = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}
        tmpl = clean_text(meta.get("template_key"))
        records.append(
            {
                "npz_path": npz_path,
                "json_path": json_path,
                "metadata": meta,
                "template_key": tmpl or "unknown",
                "family": template_family(tmpl),
                "prompt": clean_text(meta.get("prompt")) or "(empty prompt)",
                "sample_id": clean_text(meta.get("sample_id")) or npz_path.stem,
                "window_index": int(meta.get("window_index", 0)),
                "window_start": int(meta.get("window_start", 0)),
            }
        )
    return records


def select_records(records: list[dict[str, object]], per_template: int) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["template_key"])].append(record)
    selected: list[dict[str, object]] = []
    for template_key in sorted(grouped.keys()):
        bucket = sorted(
            grouped[template_key],
            key=lambda row: (str(row["sample_id"]), int(row["window_index"]), int(row["window_start"])),
        )
        selected.extend(bucket[:per_template])
    return selected


def to_uint8_rgb(frame_chw: np.ndarray) -> np.ndarray:
    image = np.transpose(np.clip(frame_chw, 0.0, 1.0), (1, 2, 0))
    return np.ascontiguousarray((image * 255.0).round().astype(np.uint8))


def denorm_box(box: np.ndarray, height: int, width: int) -> np.ndarray:
    return box.astype(np.float32) * np.asarray([width, height, width, height], dtype=np.float32)


def draw_text(frame: np.ndarray, text: str, y: int) -> None:
    cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1, cv2.LINE_AA)


def render_video_panel(frames_tchw: np.ndarray, context_len: int) -> np.ndarray:
    out = []
    total = frames_tchw.shape[0]
    for idx in range(total):
        frame = to_uint8_rgb(frames_tchw[idx])
        phase = "context" if idx < context_len else "future"
        draw_text(frame, f"{phase} frame {idx + 1}/{total}", 22)
        out.append(frame)
    return np.stack(out, axis=0)


def render_overlay_panel(
    frames_tchw: np.ndarray,
    states_tnd: np.ndarray,
    boxes_tn4: np.ndarray,
    context_len: int,
) -> np.ndarray:
    out = []
    total = frames_tchw.shape[0]
    track_colors = [
        (255, 80, 80),
        (80, 170, 255),
        (80, 220, 140),
        (255, 190, 60),
        (210, 120, 255),
        (255, 120, 200),
    ]
    for idx in range(total):
        frame = to_uint8_rgb(frames_tchw[idx])
        h, w = frame.shape[:2]
        phase = "context" if idx < context_len else "future"
        for obj_idx in range(states_tnd.shape[1]):
            state = states_tnd[idx, obj_idx]
            if float(state[StateIndex.EXISTENCE]) <= 0.5 or float(state[StateIndex.VISIBILITY]) <= 0.05:
                continue
            box = denorm_box(boxes_tn4[idx, obj_idx], h, w)
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            color = track_colors[obj_idx % len(track_colors)]
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            cx = int(round(float(state[StateIndex.CENTER_X]) * w))
            cy = int(round(float(state[StateIndex.CENTER_Y]) * h))
            cv2.circle(frame, (cx, cy), 3, color, -1)
        draw_text(frame, f"{phase} overlay {idx + 1}/{total}", 22)
        out.append(frame)
    return np.stack(out, axis=0)


def save_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def build_strip(frames_thwc: np.ndarray, count: int = 6) -> np.ndarray:
    if frames_thwc.shape[0] <= count:
        indices = list(range(frames_thwc.shape[0]))
    else:
        indices = np.linspace(0, frames_thwc.shape[0] - 1, count).round().astype(int).tolist()
    tiles = [frames_thwc[idx] for idx in indices]
    return np.concatenate(tiles, axis=1)


def write_browser_mp4(path: Path, frames_thwc: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp.mp4")
    h, w = frames_thwc.shape[1:3]
    writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer: {tmp_path}")
    for frame in frames_thwc:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    ffmpeg = find_ffmpeg()
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(tmp_path),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    tmp_path.unlink(missing_ok=True)


def summarize_selected(selected: list[dict[str, object]]) -> tuple[dict[str, int], dict[str, int]]:
    family_counts = Counter(str(item["family"]) for item in selected)
    template_counts = Counter(str(item["template_key"]) for item in selected)
    return dict(sorted(family_counts.items())), dict(sorted(template_counts.items()))


def render_html(report: dict[str, object]) -> str:
    family_sections = []
    for family_name, family in report["families"].items():
        cards = []
        for case in family["cases"]:
            cards.append(
                f"""
                <article class="case-card">
                  <div class="case-head">
                    <div>
                      <div class="eyebrow">{html.escape(family_name)} / {html.escape(case['template_key'])}</div>
                      <h3>{html.escape(case['case_id'])}</h3>
                    </div>
                    <div class="chip">window={case['window_index']} start={case['window_start']}</div>
                  </div>
                  <div class="prompt">{html.escape(case['prompt'])}</div>
                  <div class="media-grid">
                    <section class="media-card">
                      <div class="media-title">Context 帧条带</div>
                      <img src="{html.escape(case['context_strip_rel'])}" alt="context strip" />
                    </section>
                    <section class="media-card">
                      <div class="media-title">Full Episode</div>
                      <video controls preload="metadata" src="{html.escape(case['full_video_rel'])}"></video>
                    </section>
                    <section class="media-card">
                      <div class="media-title">State Overlay</div>
                      <video controls preload="metadata" src="{html.escape(case['overlay_video_rel'])}"></video>
                    </section>
                  </div>
                </article>
                """
            )
        family_sections.append(
            f"""
            <section class="family-block">
              <div class="family-head">
                <h2>{html.escape(family_name)}</h2>
                <div class="family-meta">模板数 {family['template_count']} / case 数 {family['case_count']}</div>
              </div>
              {''.join(cards)}
            </section>
            """
        )

    summary_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{count}</td></tr>"
        for name, count in report["template_counts"].items()
    )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Episode 分类抽样可视化</title>
  <style>
    :root {{
      --bg0: #f7f2ea;
      --bg1: #eadfce;
      --panel: rgba(255, 251, 245, 0.96);
      --line: #ddcfbc;
      --ink: #201b17;
      --muted: #6d665d;
      --accent: #0d5b54;
      --accent2: #b96b34;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "IBM Plex Sans", "Source Han Sans SC", "Noto Sans SC", sans-serif;
      background:
        radial-gradient(circle at left top, rgba(185, 107, 52, 0.12), transparent 24%),
        radial-gradient(circle at right top, rgba(13, 91, 84, 0.12), transparent 28%),
        linear-gradient(180deg, var(--bg0) 0%, var(--bg1) 100%);
    }}
    .page {{ max-width: 1680px; margin: 0 auto; padding: 26px; }}
    .hero, .panel, .family-block, .case-card, .media-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
    }}
    .hero, .panel, .family-block, .case-card {{ padding: 20px; margin-bottom: 18px; }}
    .eyebrow {{ color: var(--accent2); letter-spacing: 0.08em; text-transform: uppercase; font-size: 12px; margin-bottom: 6px; }}
    .intro, .prompt, .muted {{ color: var(--muted); line-height: 1.7; }}
    .summary {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; margin-top:14px; }}
    .summary-card {{ background: rgba(245,239,229,0.88); border:1px solid var(--line); border-radius:16px; padding:14px; }}
    .summary-card strong {{ display:block; font-size:28px; color:var(--accent); }}
    .summary-card span {{ color:var(--muted); font-size:13px; }}
    .family-head, .case-head {{ display:flex; justify-content:space-between; gap:16px; align-items:center; margin-bottom:12px; }}
    .family-meta, .chip {{ padding:8px 12px; border-radius:999px; border:1px solid var(--line); color:var(--muted); background:rgba(245,239,229,0.88); }}
    .media-grid {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }}
    .media-card {{ padding:12px; }}
    .media-title {{ color:var(--accent); font-weight:700; margin-bottom:8px; }}
    video, img {{ width:100%; display:block; border-radius:12px; background:#000; }}
    table {{ width:100%; border-collapse: collapse; margin-top:12px; font-size:14px; }}
    th, td {{ padding:10px 12px; border-bottom:1px solid var(--line); text-align:left; }}
    th {{ color:var(--accent); background:rgba(245,239,229,0.88); }}
    @media (max-width: 1200px) {{
      .summary {{ grid-template-columns:repeat(2,minmax(0,1fr)); }}
      .media-grid {{ grid-template-columns:1fr; }}
    }}
    @media (max-width: 760px) {{
      .summary {{ grid-template-columns:1fr; }}
      .family-head, .case-head {{ flex-direction:column; align-items:flex-start; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="eyebrow">{html.escape(str(report['split']).upper())}</div>
      <h1>4020 个 Episode 按类别抽样可视化</h1>
      <p class="intro">这页按 `template_key` 和 `F1~F5 family` 从训练 episode 中抽样展示。每个 case 展示 context 条带、完整 episode 视频、以及多物体 state overlay，方便检查训练样本的运动模式、状态连续性和目标框质量。</p>
      <div class="summary">
        <div class="summary-card"><strong>{report['total_episodes']}</strong><span>split 内 episode 总数</span></div>
        <div class="summary-card"><strong>{report['selected_cases']}</strong><span>本页抽样 case 数</span></div>
        <div class="summary-card"><strong>{report['family_count']}</strong><span>family 数</span></div>
        <div class="summary-card"><strong>{report['template_count']}</strong><span>template 数</span></div>
      </div>
    </section>
    <section class="panel">
      <h2>Template 抽样统计</h2>
      <p class="muted">当前策略是每个 `template_key` 抽 `{report['per_template']}` 个 case，所以这里可以快速看出页面覆盖了哪些模板。</p>
      <div style="overflow-x:auto;">
        <table>
          <thead><tr><th>Template</th><th>抽样数</th></tr></thead>
          <tbody>{summary_rows}</tbody>
        </table>
      </div>
    </section>
    {''.join(family_sections)}
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
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(split_dir)
    selected = select_records(records, args.per_template)
    family_counts, template_counts = summarize_selected(selected)

    family_cases: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in selected:
        payload = np.load(record["npz_path"], allow_pickle=False)
        context_frames = payload["context_frames"].astype(np.float32)
        future_frames = payload["future_frames"].astype(np.float32)
        context_states = payload["context_states"].astype(np.float32)
        future_states = payload["future_states"].astype(np.float32)
        context_boxes = payload["context_boxes"].astype(np.float32)
        future_boxes = payload["future_boxes"].astype(np.float32)

        frames = np.concatenate([context_frames, future_frames], axis=0)
        states = np.concatenate([context_states, future_states], axis=0)
        boxes = np.concatenate([context_boxes, future_boxes], axis=0)
        context_len = context_frames.shape[0]

        case_id = str(record["npz_path"].stem)
        case_dir = assets_dir / case_id
        context_strip = build_strip(np.stack([to_uint8_rgb(frame) for frame in context_frames], axis=0), count=min(6, context_len))
        full_video = render_video_panel(frames, context_len)
        overlay_video = render_overlay_panel(frames, states, boxes, context_len)

        context_strip_path = case_dir / "context_strip.png"
        full_video_path = case_dir / "full.browser.mp4"
        overlay_video_path = case_dir / "overlay.browser.mp4"
        save_png(context_strip_path, context_strip)
        write_browser_mp4(full_video_path, full_video, args.fps)
        write_browser_mp4(overlay_video_path, overlay_video, args.fps)

        family_cases[str(record["family"])].append(
            {
                "case_id": case_id,
                "template_key": str(record["template_key"]),
                "prompt": str(record["prompt"]),
                "window_index": int(record["window_index"]),
                "window_start": int(record["window_start"]),
                "context_strip_rel": str(context_strip_path.relative_to(output_dir)),
                "full_video_rel": str(full_video_path.relative_to(output_dir)),
                "overlay_video_rel": str(overlay_video_path.relative_to(output_dir)),
            }
        )

    families = {
        family_name: {
            "template_count": len({case["template_key"] for case in cases}),
            "case_count": len(cases),
            "cases": sorted(cases, key=lambda item: (item["template_key"], item["case_id"])),
        }
        for family_name, cases in sorted(family_cases.items())
    }

    report = {
        "split": args.split,
        "data_root": str(data_root),
        "per_template": args.per_template,
        "total_episodes": len(records),
        "selected_cases": len(selected),
        "family_count": len(families),
        "template_count": len(template_counts),
        "family_counts": family_counts,
        "template_counts": template_counts,
        "families": families,
    }
    (output_dir / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "index.html").write_text(render_html(report), encoding="utf-8")
    print(json.dumps({"index": str(output_dir / 'index.html'), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
