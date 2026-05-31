from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phys_state_video.schemas import StateIndex


SOURCE_ORDER = ["PhysicsIQ", "OpenVidHD", "WebVid10M", "ExistingCurated"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a local HTML gallery for curated training episodes.")
    parser.add_argument("--data-root", required=True, help="Episode root with train/val directories.")
    parser.add_argument("--split", default="train", choices=["train", "val"], help="Dataset split to visualize.")
    parser.add_argument("--output-dir", required=True, help="Gallery output directory.")
    parser.add_argument("--fps", type=int, default=6, help="Playback fps for exported mp4 previews.")
    parser.add_argument("--max-cases", type=int, default=16, help="Maximum number of cases to export.")
    parser.add_argument("--per-source", type=int, default=4, help="Preferred number of cases per source domain.")
    parser.add_argument(
        "--display-long-side",
        type=int,
        default=480,
        help="Long side for display-only aspect-restored previews.",
    )
    return parser.parse_args()


def clean_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def infer_source_label(file_stem: str, metadata: dict[str, object]) -> str:
    source = metadata.get("source")
    if isinstance(source, dict):
        dataset_name = clean_text(source.get("dataset", ""))
        if dataset_name:
            return dataset_name
    if "perspective-center" in file_stem or file_stem[:4].isdigit():
        return "PhysicsIQ"
    return "ExistingCurated"


def extract_categories(metadata: dict[str, object]) -> list[str]:
    source = metadata.get("source")
    if isinstance(source, dict):
        categories = source.get("categories")
        if isinstance(categories, list):
            return [clean_text(item) for item in categories if clean_text(item)]
    return []


def load_records(split_dir: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for npz_path in sorted(split_dir.glob("*.npz")):
        json_path = npz_path.with_suffix(".json")
        metadata: dict[str, object] = {}
        if json_path.exists():
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
        file_stem = npz_path.stem
        records.append(
            {
                "file_stem": file_stem,
                "npz_path": npz_path,
                "json_path": json_path,
                "metadata": metadata,
                "prompt": clean_text(metadata.get("prompt", "")),
                "source_label": infer_source_label(file_stem, metadata),
                "categories": extract_categories(metadata),
            }
        )
    return records


def select_records(records: list[dict[str, object]], max_cases: int, per_source: int) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        grouped[record["source_label"]].append(record)

    selected: list[dict[str, object]] = []
    seen_paths: set[Path] = set()

    for source_label in SOURCE_ORDER:
        source_records = grouped.get(source_label, [])
        if not source_records:
            continue
        picked_categories: set[str] = set()
        source_selected = 0
        for record in source_records:
            if len(selected) >= max_cases:
                break
            if record["npz_path"] in seen_paths:
                continue
            categories = record["categories"] or ["uncategorized"]
            primary = categories[0]
            if source_selected >= per_source:
                break
            if primary in picked_categories and len(source_records) > per_source:
                continue
            picked_categories.add(primary)
            selected.append(record)
            seen_paths.add(record["npz_path"])
            source_selected += 1
        if source_selected >= per_source:
            continue
        for record in source_records:
            if len(selected) >= max_cases or source_selected >= per_source:
                break
            if record["npz_path"] in seen_paths:
                continue
            selected.append(record)
            seen_paths.add(record["npz_path"])
            source_selected += 1

    if len(selected) < max_cases:
        for record in records:
            if len(selected) >= max_cases:
                break
            if record["npz_path"] in seen_paths:
                continue
            selected.append(record)
            seen_paths.add(record["npz_path"])

    return selected[:max_cases]


def to_uint8_rgb(frame_chw: np.ndarray) -> np.ndarray:
    image = np.transpose(np.clip(frame_chw, 0.0, 1.0), (1, 2, 0))
    return np.ascontiguousarray((image * 255.0).round().astype(np.uint8))


def denorm_box(box: np.ndarray, height: int, width: int) -> np.ndarray:
    scale = np.asarray([width, height, width, height], dtype=np.float32)
    return box.astype(np.float32) * scale


def draw_label(frame_rgb: np.ndarray, text: str, y: int) -> None:
    cv2.putText(
        frame_rgb,
        text,
        (8, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_rgb,
        text,
        (8, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )


def render_overlay_frames(
    frames_tchw: np.ndarray,
    states_tn: np.ndarray,
    boxes_t4: np.ndarray,
    context_len: int,
    display_hw: tuple[int, int] | None = None,
    preview_label: str | None = None,
) -> np.ndarray:
    overlays: list[np.ndarray] = []
    trail: list[tuple[int, int]] = []
    total = frames_tchw.shape[0]
    for idx in range(total):
        frame = to_uint8_rgb(frames_tchw[idx])
        if display_hw is not None and tuple(frame.shape[:2]) != tuple(display_hw):
            display_h, display_w = display_hw
            frame = cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_CUBIC)
        height, width = frame.shape[:2]
        box = denorm_box(boxes_t4[idx], height, width)
        x0, y0, x1, y1 = [int(round(value)) for value in box]
        center_x = float(states_tn[idx, StateIndex.CENTER_X] * width)
        center_y = float(states_tn[idx, StateIndex.CENTER_Y] * height)
        center = (int(round(center_x)), int(round(center_y)))
        visibility = float(states_tn[idx, StateIndex.VISIBILITY])
        depth = float(states_tn[idx, StateIndex.DEPTH])
        log_scale = float(states_tn[idx, StateIndex.LOG_SCALE])
        confidence = float(states_tn[idx, StateIndex.CONFIDENCE])
        phase = "context" if idx < context_len else "future"
        color = (47, 156, 255) if phase == "context" else (255, 142, 43)
        if visibility > 0.5:
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            cv2.circle(frame, center, 3, color, -1)
            trail.append(center)
        if len(trail) >= 2:
            for left, right in zip(trail[:-1], trail[1:]):
                cv2.line(frame, left, right, (52, 214, 183), 2, cv2.LINE_AA)
        draw_label(frame, f"{phase} frame {idx + 1}/{total}", 18)
        draw_label(frame, f"vis={visibility:.2f}  conf={confidence:.2f}", 38)
        draw_label(frame, f"depth={depth:.3f}  log_scale={log_scale:.3f}", 58)
        if preview_label:
            draw_label(frame, preview_label, 78)
        overlays.append(frame)
    return np.stack(overlays, axis=0)


def write_mp4(path: Path, frames_thwc: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames_thwc.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer for {path}")
    for frame in frames_thwc:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


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


def infer_original_hw(file_stem: str, metadata: dict[str, object]) -> tuple[int, int] | None:
    resize = metadata.get("resize")
    if isinstance(resize, dict):
        original_height = resize.get("original_height")
        original_width = resize.get("original_width")
        try:
            height = int(original_height)
            width = int(original_width)
        except (TypeError, ValueError):
            height = 0
            width = 0
        if height >= 64 and width >= 64:
            return height, width
    for text in (file_stem, json.dumps(metadata, ensure_ascii=False)):
        match = re.search(r"(?<!\d)(\d{2,5})x(\d{2,5})(?!\d)", text)
        if match is None:
            continue
        height = int(match.group(1))
        width = int(match.group(2))
        if height >= 64 and width >= 64:
            return height, width
    return None


def infer_display_hw(
    original_hw: tuple[int, int] | None,
    display_long_side: int,
    fallback_hw: tuple[int, int],
) -> tuple[int, int]:
    if original_hw is None:
        return fallback_hw
    original_h, original_w = original_hw
    if original_h <= 0 or original_w <= 0:
        return fallback_hw
    scale = float(display_long_side) / float(max(original_h, original_w))
    display_h = max(1, int(round(original_h * scale)))
    display_w = max(1, int(round(original_w * scale)))
    return display_h, display_w


def summarize_motion(states_tn: np.ndarray, boxes_t4: np.ndarray) -> dict[str, float]:
    visibility = states_tn[:, StateIndex.VISIBILITY] > 0.5
    visible_states = states_tn[visibility]
    visible_boxes = boxes_t4[visibility]
    if visible_states.shape[0] < 2:
        return {"path_length": 0.0, "net_displacement": 0.0, "scale_span": 0.0}
    centers = visible_states[:, [StateIndex.CENTER_X, StateIndex.CENTER_Y]]
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    widths = np.clip(visible_boxes[:, 2] - visible_boxes[:, 0], 1e-6, 1.0)
    heights = np.clip(visible_boxes[:, 3] - visible_boxes[:, 1], 1e-6, 1.0)
    log_areas = np.log(widths * heights)
    return {
        "path_length": float(np.sum(steps)),
        "net_displacement": float(np.linalg.norm(centers[-1] - centers[0])),
        "scale_span": float(np.max(log_areas) - np.min(log_areas)),
    }


def save_state_plot(path: Path, states_tn: np.ndarray, context_len: int) -> None:
    frames = np.arange(states_tn.shape[0], dtype=np.int32)
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.0), dpi=140)

    axes[0, 0].plot(frames, states_tn[:, StateIndex.CENTER_X], label="center_x", color="#1f77b4")
    axes[0, 0].plot(frames, states_tn[:, StateIndex.CENTER_Y], label="center_y", color="#ff7f0e")
    axes[0, 0].set_title("Normalized Center")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(frames, states_tn[:, StateIndex.DEPTH], color="#2ca02c")
    axes[0, 1].set_title("Relative Depth")

    axes[1, 0].plot(frames, states_tn[:, StateIndex.LOG_SCALE], color="#d62728")
    axes[1, 0].set_title("Log Scale")

    axes[1, 1].plot(frames, states_tn[:, StateIndex.VISIBILITY], label="visibility", color="#9467bd")
    axes[1, 1].plot(frames, states_tn[:, StateIndex.CONFIDENCE], label="confidence", color="#8c564b")
    axes[1, 1].set_title("Visibility / Confidence")
    axes[1, 1].legend(fontsize=8)

    for ax in axes.reshape(-1):
        ax.axvline(context_len - 0.5, color="#444444", linestyle="--", linewidth=1.0)
        ax.grid(alpha=0.28)
        ax.set_xlabel("Frame")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def render_html(report: dict[str, object]) -> str:
    source_filters = "".join(
        f'<button class="filter-btn" data-key="source" data-value="{item}">{item}</button>'
        for item in report["source_counts"].keys()
    )
    category_filters = "".join(
        f'<button class="filter-btn" data-key="category" data-value="{item}">{item}</button>'
        for item in report["category_counts"].keys()
    )

    cards = []
    for case in report["cases"]:
        categories = ", ".join(case["categories"]) if case["categories"] else "uncategorized"
        cards.append(
            f"""
            <section class="case-card" data-source="{case['source_label']}" data-category="{categories}">
              <div class="card-head">
                <div>
                  <h2>{case['case_id']}</h2>
                  <div class="meta-line">
                    <span>{case['source_label']}</span>
                    <span>{categories}</span>
                    <span>train={case['train_hw']}</span>
                    <span>display={case['display_hw']}</span>
                    <span>path={case['motion']['path_length']:.3f}</span>
                    <span>disp={case['motion']['net_displacement']:.3f}</span>
                    <span>scale-span={case['motion']['scale_span']:.3f}</span>
                  </div>
                </div>
              </div>
              <div class="prompt">{case['prompt']}</div>
              <div class="asset-grid">
                <div class="asset-card">
                  <div class="asset-title">训练张量预览 `{case['train_hw']}`</div>
                  <video controls preload="metadata" src="{case['video_rel']}"></video>
                  <div class="asset-note">这是模型真实读入的训练 episode，已经在构建阶段被直接压成正方形。</div>
                </div>
                <div class="asset-card">
                  <div class="asset-title">按原始宽高比恢复的预览</div>
                  <video controls preload="metadata" src="{case['display_video_rel']}"></video>
                  <div class="asset-note">{case['display_note']}</div>
                </div>
                <div class="asset-card">
                  <div class="asset-title">关键帧条带</div>
                  <img src="{case['display_strip_rel']}" alt="strip" />
                </div>
                <div class="asset-card">
                  <div class="asset-title">状态曲线</div>
                  <img src="{case['plot_rel']}" alt="plot" />
                </div>
              </div>
              <details class="raw-block">
                <summary>展开元信息</summary>
                <pre>{case['metadata_json']}</pre>
              </details>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>训练样本 Gallery</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --panel: rgba(255, 252, 245, 0.9);
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
      max-width: 1500px;
      margin: 0 auto;
      padding: 28px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 34px;
      letter-spacing: 0.02em;
    }}
    .lead {{
      max-width: 980px;
      color: var(--muted);
      line-height: 1.7;
      margin-bottom: 18px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 16px;
    }}
    .summary-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px 16px;
      box-shadow: 0 18px 50px rgba(77, 56, 24, 0.08);
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
    .filters {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 8px 0 24px;
    }}
    .filter-group {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      background: rgba(255,255,255,0.55);
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 10px;
    }}
    .filter-group label {{
      color: var(--muted);
      font-size: 13px;
      padding: 0 4px;
    }}
    .filter-btn {{
      border: 1px solid var(--line);
      background: #fffaf2;
      color: var(--ink);
      border-radius: 999px;
      padding: 6px 10px;
      cursor: pointer;
    }}
    .filter-btn.active {{
      background: var(--accent);
      color: white;
      border-color: var(--accent);
    }}
    .case-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 18px;
      margin-bottom: 22px;
      box-shadow: 0 18px 50px rgba(77, 56, 24, 0.08);
    }}
    .card-head {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: start;
      margin-bottom: 8px;
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
      padding: 4px 8px;
      font-size: 12px;
    }}
    .prompt {{
      color: var(--muted);
      line-height: 1.7;
      margin-bottom: 14px;
      white-space: pre-wrap;
    }}
    .asset-grid {{
      display: grid;
      grid-template-columns: 1.35fr 1fr 1fr;
      gap: 14px;
    }}
    .asset-card {{
      background: rgba(255,255,255,0.72);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
    }}
    .asset-title {{
      font-weight: 700;
      margin-bottom: 10px;
      color: var(--accent);
    }}
    .asset-note {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.55;
    }}
    video, img {{
      width: 100%;
      display: block;
      border-radius: 10px;
      background: #000;
    }}
    .raw-block {{
      margin-top: 12px;
    }}
    .raw-block summary {{
      cursor: pointer;
      color: var(--accent);
      font-weight: 700;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      line-height: 1.55;
      color: #2d2b28;
      background: #f9f4eb;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      margin: 10px 0 0;
      max-height: 220px;
      overflow: auto;
    }}
    @media (max-width: 1200px) {{
      .summary {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .asset-grid {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 760px) {{
      .summary {{ grid-template-columns: 1fr; }}
      .wrap {{ padding: 16px; }}
      h1 {{ font-size: 28px; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>训练样本可视化</h1>
    <div class="lead">
      这页展示当前 object-motion 训练集里一批代表样本。左侧的训练张量预览就是模型真实看到的训练 episode；
      这些 episode 在构建时被直接 resize 成正方形，所以会出现低分辨率和宽高比形变。右侧预览仅用于展示，
      会依据元数据里可恢复的原始宽高比，把同一段训练张量反变形回更接近原视频的比例，便于检查样本内容本身。
    </div>

    <div class="summary">
      <div class="summary-card"><strong>{report['case_count']}</strong><span>已展示样本数</span></div>
      <div class="summary-card"><strong>{report['split']}</strong><span>数据划分</span></div>
      <div class="summary-card"><strong>{report['source_count']}</strong><span>来源域数</span></div>
      <div class="summary-card"><strong>{report['category_count']}</strong><span>类别数</span></div>
    </div>

    <div class="filters">
      <div class="filter-group">
        <label>来源</label>
        <button class="filter-btn active" data-key="source" data-value="all">全部</button>
        {source_filters}
      </div>
      <div class="filter-group">
        <label>类别</label>
        <button class="filter-btn active" data-key="category" data-value="all">全部</button>
        {category_filters}
      </div>
    </div>

    <div id="cards">
      {''.join(cards)}
    </div>
  </div>
  <script>
    const active = {{ source: 'all', category: 'all' }};
    function applyFilters() {{
      const cards = document.querySelectorAll('.case-card');
      cards.forEach((card) => {{
        const sourceOk = active.source === 'all' || card.dataset.source === active.source;
        const cats = (card.dataset.category || '').split(',').map(x => x.trim()).filter(Boolean);
        const categoryOk = active.category === 'all' || cats.includes(active.category);
        card.style.display = (sourceOk && categoryOk) ? '' : 'none';
      }});
      document.querySelectorAll('.filter-btn').forEach((btn) => {{
        const key = btn.dataset.key;
        const value = btn.dataset.value;
        btn.classList.toggle('active', active[key] === value);
      }});
    }}
    document.querySelectorAll('.filter-btn').forEach((btn) => {{
      btn.addEventListener('click', () => {{
        active[btn.dataset.key] = btn.dataset.value;
        applyFilters();
      }});
    }});
    applyFilters();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    split_dir = data_root / args.split
    output_dir = Path(args.output_dir)
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(split_dir)
    selected = select_records(records, args.max_cases, args.per_source)

    source_counts = Counter(item["source_label"] for item in selected)
    category_counts = Counter()
    report_cases: list[dict[str, object]] = []

    for record in selected:
        payload = np.load(record["npz_path"], allow_pickle=False)
        context_frames = payload["context_frames"].astype(np.float32)
        future_frames = payload["future_frames"].astype(np.float32)
        context_states = payload["context_states"].astype(np.float32)[:, 0, :]
        future_states = payload["future_states"].astype(np.float32)[:, 0, :]
        context_boxes = payload["context_boxes"].astype(np.float32)[:, 0, :]
        future_boxes = payload["future_boxes"].astype(np.float32)[:, 0, :]

        frames = np.concatenate([context_frames, future_frames], axis=0)
        states = np.concatenate([context_states, future_states], axis=0)
        boxes = np.concatenate([context_boxes, future_boxes], axis=0)
        context_len = context_frames.shape[0]
        train_hw = tuple(int(value) for value in frames.shape[-2:])
        original_hw = infer_original_hw(record["file_stem"], record["metadata"])
        display_hw = infer_display_hw(original_hw, args.display_long_side, train_hw)

        overlay = render_overlay_frames(
            frames,
            states,
            boxes,
            context_len,
            display_hw=train_hw,
            preview_label="training tensor",
        )
        display_overlay = render_overlay_frames(
            frames,
            states,
            boxes,
            context_len,
            display_hw=display_hw,
            preview_label="aspect-restored preview" if original_hw is not None else "upsampled preview",
        )
        strip = build_strip(display_overlay, samples=6)
        motion = summarize_motion(states, boxes)
        plot_path = assets_dir / f"{record['file_stem']}__states.png"
        video_path = assets_dir / f"{record['file_stem']}__overlay.mp4"
        display_video_path = assets_dir / f"{record['file_stem']}__display_overlay.mp4"
        strip_path = assets_dir / f"{record['file_stem']}__display_strip.png"
        write_mp4(video_path, overlay, args.fps)
        write_mp4(display_video_path, display_overlay, args.fps)
        save_png(strip_path, strip)
        save_state_plot(plot_path, states, context_len)

        if original_hw is not None:
            display_note = (
                f"根据元数据推断原始分辨率约为 {original_hw[1]}x{original_hw[0]}，"
                f"当前以 {display_hw[1]}x{display_hw[0]} 做展示级宽高比恢复。"
            )
        else:
            display_note = "该样本缺少原始宽高比元数据，当前只能做放大展示，无法恢复真实比例。"

        categories = record["categories"] or ["uncategorized"]
        for category in categories:
            category_counts[category] += 1

        report_cases.append(
            {
                "case_id": record["file_stem"],
                "source_label": record["source_label"],
                "categories": categories,
                "prompt": record["prompt"] or "(empty prompt)",
                "motion": motion,
                "video_rel": str(video_path.relative_to(output_dir)),
                "display_video_rel": str(display_video_path.relative_to(output_dir)),
                "display_strip_rel": str(strip_path.relative_to(output_dir)),
                "plot_rel": str(plot_path.relative_to(output_dir)),
                "train_hw": f"{train_hw[1]}x{train_hw[0]}",
                "display_hw": f"{display_hw[1]}x{display_hw[0]}",
                "display_note": display_note,
                "metadata_json": json.dumps(record["metadata"], ensure_ascii=False, indent=2),
            }
        )

    report = {
        "split": args.split,
        "case_count": len(report_cases),
        "source_count": len(source_counts),
        "category_count": len(category_counts),
        "source_counts": dict(source_counts),
        "category_counts": dict(category_counts),
        "cases": report_cases,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "index.html").write_text(render_html(report), encoding="utf-8")
    print(json.dumps({"gallery": str(output_dir / "index.html"), **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
