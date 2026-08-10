#!/usr/bin/env python3
"""Build a static viewer for the exact PyBullet window used by training."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from collections import Counter
from pathlib import Path


DEFAULT_DATASET_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0717pybullet_5000_vbenchtop5"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "full_sa_no_object_pybullet100_training_data"
)
TRAIN_FRAMES = 49
CONTEXT_FRAMES = 8
FPS = 30.0


def stable_split(family: str, case_id: str) -> str:
    digest = hashlib.sha1(f"{family}/{case_id}".encode("utf-8")).hexdigest()
    value = int(digest[:12], 16) / float(16**12 - 1)
    if value < 0.90:
        return "train"
    if value < 0.95:
        return "val"
    return "test"


def relative_video_path(row: dict) -> str:
    return (
        f"dataset/cases/{row['family_key']}/{row['case_id']}/videos/"
        f"{row['case_id']}.mp4"
    )


def video_panel(title: str, subtitle: str, src: str, end: float | None) -> str:
    end_attr = "" if end is None else f' data-end="{end:.6f}"'
    return f"""
      <figure>
        <figcaption><strong>{html.escape(title)}</strong><span>{html.escape(subtitle)}</span></figcaption>
        <video controls muted playsinline preload="metadata"{end_attr}>
          <source src="{html.escape(src)}" type="video/mp4">
        </video>
      </figure>"""


def build_page(rows: list[dict], split_counts: Counter, source_counts: Counter) -> str:
    cards = []
    for row in rows:
        src = relative_video_path(row)
        source_frames = int(row.get("motion_metrics", {}).get("frame_count", 90))
        cards.append(
            f"""
    <article class="case" id="{html.escape(row['family_key'])}">
      <header>
        <div><span class="family">{html.escape(row['family_key'])}</span>
        <code>{html.escape(row['case_id'])}</code></div>
        <p>{html.escape(str(row.get('caption', '')))}</p>
      </header>
      <div class="videos">
        {video_panel('原始源视频', f'{source_frames} 帧 · 30 FPS · 3.000 秒', src, None)}
        {video_panel('训练窗口', 'frame 0–48 · 49 帧 · 1.633 秒', src, TRAIN_FRAMES / FPS)}
        {video_panel('条件视频', 'frame 0–7 · 8 帧 · 0.267 秒', src, CONTEXT_FRAMES / FPS)}
      </div>
      <div class="timeline" aria-label="49-frame training timeline">
        <div class="context" style="width:{CONTEXT_FRAMES / TRAIN_FRAMES * 100:.6f}%">8f context</div>
        <div class="future">41f supervised continuation</div>
      </div>
    </article>"""
        )

    raw_frame_summary = ", ".join(
        f"{frames}f × {count}" for frames, count in sorted(source_counts.items())
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Full-SA + No-Object · PyBullet 100% 训练数据</title>
  <style>
    :root {{ color-scheme: dark; --bg:#0b1016; --panel:#121a24; --line:#293647;
      --text:#eef5ff; --muted:#9aacc2; --cyan:#4dd4c6; --amber:#ffbf69; }}
    * {{ box-sizing:border-box }}
    body {{ margin:0; background:radial-gradient(circle at 15% 0,#17303a 0,transparent 28%),var(--bg);
      color:var(--text); font:15px/1.55 Inter,ui-sans-serif,system-ui,sans-serif; }}
    main {{ width:min(1560px,calc(100% - 32px)); margin:auto; padding:28px 0 72px }}
    .hero,.case {{ background:rgba(18,26,36,.94); border:1px solid var(--line); border-radius:20px; }}
    .hero {{ padding:24px; margin-bottom:20px }}
    h1 {{ margin:0 0 8px; font-size:clamp(26px,4vw,48px); letter-spacing:-.035em }}
    .lead {{ color:var(--muted); max-width:1100px; margin:0 }}
    .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; margin-top:20px }}
    .stat {{ padding:14px; border-radius:14px; background:#0d141d; border:1px solid var(--line) }}
    .stat b {{ display:block; color:var(--cyan); font-size:24px }}
    .stat span {{ color:var(--muted); font-size:12px }}
    .note {{ margin-top:16px; padding:13px 15px; border-left:3px solid var(--amber); background:#171716; color:#e6dbc9 }}
    nav {{ display:flex; flex-wrap:wrap; gap:8px; margin:16px 0 20px }}
    nav a {{ color:var(--text); text-decoration:none; border:1px solid var(--line); border-radius:999px; padding:7px 12px }}
    .case {{ padding:18px; margin:14px 0; scroll-margin-top:16px }}
    .case header {{ display:flex; gap:16px; align-items:baseline; justify-content:space-between; margin-bottom:12px }}
    .case header p {{ margin:0; color:var(--muted); text-align:right; max-width:900px }}
    .family {{ color:#071315; background:var(--cyan); font-weight:800; border-radius:7px; padding:3px 8px; margin-right:8px }}
    code {{ color:#bcd0e8 }}
    .videos {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px }}
    figure {{ margin:0; background:#080d13; border:1px solid var(--line); border-radius:14px; overflow:hidden }}
    figcaption {{ display:flex; justify-content:space-between; gap:8px; padding:10px 12px }}
    figcaption span {{ color:var(--muted); font-size:12px; text-align:right }}
    video {{ display:block; width:100%; aspect-ratio:16/9; background:#000 }}
    .timeline {{ display:flex; height:28px; margin-top:12px; border-radius:8px; overflow:hidden; font-size:11px; font-weight:700 }}
    .timeline div {{ display:grid; place-items:center; white-space:nowrap }}
    .context {{ background:var(--amber); color:#231503; min-width:100px }}
    .future {{ background:#235f66; flex:1 }}
    #replay {{ position:fixed; right:22px; bottom:20px; z-index:5; border:0; border-radius:999px;
      padding:13px 18px; background:var(--cyan); color:#061313; font-weight:800; cursor:pointer; box-shadow:0 8px 30px #0008 }}
    @media(max-width:900px) {{ .videos {{ grid-template-columns:1fr }} .case header {{ display:block }}
      .case header p {{ text-align:left; margin-top:10px }} }}
  </style>
</head>
<body>
<main>
  <section class="hero">
    <h1>Full-SA + No-Object (PyBullet 100%) · 训练数据</h1>
    <p class="lead">页面严格按训练 Dataset 的读取方式展示：原始视频均为 90 帧，训练使用 prefix 采样，读取 frame 0–48；其中 frame 0–7 作为干净条件帧，loss 排除对应的 context latent，仅监督非条件 latent 时间位置。</p>
    <div class="stats">
      <div class="stat"><b>49 帧</b><span>每个训练样本的模型输入窗口</span></div>
      <div class="stat"><b>8 帧</b><span>固定条件帧，frame 0–7</span></div>
      <div class="stat"><b>41 帧</b><span>条件之后的像素时间区间</span></div>
      <div class="stat"><b>90 帧</b><span>每条原始 PyBullet 视频</span></div>
      <div class="stat"><b>{split_counts['train']}</b><span>确定性 train split 样本数</span></div>
      <div class="stat"><b>512×896</b><span>训练预处理分辨率</span></div>
    </div>
    <div class="note">数据全集：{sum(split_counts.values())} 条；train/val/test = {split_counts['train']}/{split_counts['val']}/{split_counts['test']}。源视频帧数分布：{html.escape(raw_frame_summary)}。下方每个 F1–F10 family 展示一个真实 train-split 样本。</div>
  </section>
  <nav>{''.join(f'<a href="#{family}">{family}</a>' for family in sorted((r['family_key'] for r in rows), key=lambda x:int(x[1:])))}</nav>
  {''.join(cards)}
</main>
<button id="replay">↻ 全部重新播放</button>
<script>
  const clipped=[...document.querySelectorAll('video[data-end]')];
  for(const video of clipped){{
    video.addEventListener('timeupdate',()=>{{
      const end=Number(video.dataset.end);
      if(video.currentTime>=end){{ video.pause(); video.currentTime=0; }}
    }});
  }}
  document.getElementById('replay').addEventListener('click',()=>{{
    for(const video of document.querySelectorAll('video')){{ video.currentTime=0; video.play().catch(()=>{{}}); }}
  }});
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    split_counts = Counter(
        stable_split(row["family_key"], row["case_id"]) for row in manifest
    )
    source_counts = Counter(
        int(row.get("motion_metrics", {}).get("frame_count", 0)) for row in manifest
    )
    selected: dict[str, dict] = {}
    for row in manifest:
        family = str(row["family_key"])
        if family not in selected and stable_split(family, row["case_id"]) == "train":
            selected[family] = row
    rows = sorted(selected.values(), key=lambda row: int(str(row["family_key"])[1:]))
    if len(rows) != 10:
        raise RuntimeError(f"Expected one train sample for F1-F10, found {len(rows)}")

    output_root.mkdir(parents=True, exist_ok=True)
    dataset_link = output_root / "dataset"
    if dataset_link.is_symlink():
        if dataset_link.resolve() != dataset_root:
            raise RuntimeError(f"Unexpected dataset symlink target: {dataset_link.resolve()}")
    elif dataset_link.exists():
        raise RuntimeError(f"Refusing to replace existing path: {dataset_link}")
    else:
        dataset_link.symlink_to(dataset_root, target_is_directory=True)

    (output_root / "index.html").write_text(
        build_page(rows, split_counts, source_counts), encoding="utf-8"
    )
    (output_root / "viewer_manifest.json").write_text(
        json.dumps(
            {
                "dataset_root": str(dataset_root),
                "training_frames": TRAIN_FRAMES,
                "context_frames": CONTEXT_FRAMES,
                "fps": FPS,
                "split_counts": dict(split_counts),
                "selected_cases": [
                    {"family_key": row["family_key"], "case_id": row["case_id"]}
                    for row in rows
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output_root / "index.html")


if __name__ == "__main__":
    main()
