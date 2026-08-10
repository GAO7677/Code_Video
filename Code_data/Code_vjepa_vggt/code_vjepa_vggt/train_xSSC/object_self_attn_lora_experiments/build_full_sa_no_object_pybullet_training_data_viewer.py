#!/usr/bin/env python3
"""Build one viewer for the PyBullet-100% and Kubric-100% training data."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from collections import Counter
from pathlib import Path

from decord import VideoReader, cpu


DEFAULT_PYBULLET_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0717pybullet_5000_vbenchtop5"
)
DEFAULT_KUBRIC_ROOT = Path("/data/gaoya/dataset/nnsriram97-phyco_kubric")
DEFAULT_KUBRIC_INDEX = Path(
    "/data/gaoya/agent-data/cache/kubric_no_gt_box_dataset/indices/"
    "kubric_index_192dff90cbc0e357.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "full_sa_no_object_pybullet100_training_data"
)
TRAIN_FRAMES = 49
CONTEXT_FRAMES = 8


def stable_split(key: str) -> str:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    value = int(digest[:12], 16) / float(16**12 - 1)
    if value < 0.90:
        return "train"
    if value < 0.95:
        return "val"
    return "test"


def probe_video(path: Path) -> tuple[int, float]:
    reader = VideoReader(str(path), ctx=cpu(0))
    return len(reader), float(reader.get_avg_fps())


def ensure_dataset_link(link: Path, target: Path) -> None:
    if link.is_symlink():
        if link.resolve() != target:
            raise RuntimeError(f"Unexpected dataset symlink target: {link.resolve()}")
    elif link.exists():
        raise RuntimeError(f"Refusing to replace existing path: {link}")
    else:
        link.symlink_to(target, target_is_directory=True)


def video_panel(title: str, subtitle: str, src: str, end: float | None) -> str:
    end_attr = "" if end is None else f' data-end="{end:.6f}"'
    return f"""
      <figure>
        <figcaption><strong>{html.escape(title)}</strong><span>{html.escape(subtitle)}</span></figcaption>
        <video controls muted playsinline preload="metadata"{end_attr}>
          <source src="{html.escape(src)}" type="video/mp4">
        </video>
      </figure>"""


def case_card(row: dict) -> str:
    fps = float(row["fps"])
    source_frames = int(row["source_frames"])
    source_duration = source_frames / fps
    return f"""
    <article class="case" id="{html.escape(row['anchor'])}">
      <header>
        <div><span class="family">{html.escape(row['group'])}</span>
        <code>{html.escape(row['case_id'])}</code></div>
        <p>{html.escape(row['caption'])}</p>
      </header>
      <div class="videos">
        {video_panel('原始源视频', f'{source_frames} 帧 · {fps:g} FPS · {source_duration:.3f} 秒', row['src'], None)}
        {video_panel('训练窗口', f'frame 0–48 · 49 帧 · {TRAIN_FRAMES / fps:.3f} 秒', row['src'], TRAIN_FRAMES / fps)}
        {video_panel('条件视频', f'frame 0–7 · 8 帧 · {CONTEXT_FRAMES / fps:.3f} 秒', row['src'], CONTEXT_FRAMES / fps)}
      </div>
      <div class="timeline" aria-label="49-frame training timeline">
        <div class="context" style="width:{CONTEXT_FRAMES / TRAIN_FRAMES * 100:.6f}%">8f context</div>
        <div class="future">41f supervised continuation</div>
      </div>
    </article>"""


def dataset_nav(rows: list[dict]) -> str:
    return "".join(
        f'<a href="#{html.escape(row["anchor"])}">{html.escape(row["group"])}</a>'
        for row in rows
    )


def build_page(
    pybullet_rows: list[dict],
    kubric_rows: list[dict],
    pybullet_split_counts: Counter,
    pybullet_source_counts: Counter,
    kubric_scenario_counts: Counter,
    kubric_hint_counts: Counter,
) -> str:
    pybullet_source_summary = ", ".join(
        f"{frames}f × {count}" for frames, count in sorted(pybullet_source_counts.items())
    )
    kubric_hint_summary = ", ".join(
        f"{frames}f × {count}" for frames, count in sorted(kubric_hint_counts.items())
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Full-SA + No-Object · 训练数据</title>
  <style>
    :root {{ color-scheme:dark; --bg:#0b1016; --panel:#121a24; --line:#293647;
      --text:#eef5ff; --muted:#9aacc2; --cyan:#4dd4c6; --amber:#ffbf69; }}
    * {{ box-sizing:border-box }}
    body {{ margin:0; background:radial-gradient(circle at 15% 0,#17303a 0,transparent 28%),var(--bg);
      color:var(--text); font:15px/1.55 Inter,ui-sans-serif,system-ui,sans-serif; }}
    main {{ width:min(1560px,calc(100% - 32px)); margin:auto; padding:28px 0 72px }}
    .hero,.dataset-summary,.case {{ background:rgba(18,26,36,.94); border:1px solid var(--line); border-radius:20px }}
    .hero,.dataset-summary {{ padding:24px; margin-bottom:18px }}
    h1 {{ margin:0 0 8px; font-size:clamp(26px,4vw,48px); letter-spacing:-.035em }}
    h2 {{ margin:0 0 7px; font-size:28px }}
    .lead,.dataset-summary>p {{ color:var(--muted); max-width:1160px; margin:0 }}
    .tabs {{ position:sticky; top:12px; z-index:6; display:flex; gap:10px; padding:10px; margin:18px 0;
      width:max-content; max-width:100%; border:1px solid var(--line); border-radius:999px; background:#0b1016e8; backdrop-filter:blur(12px) }}
    .tab {{ border:0; border-radius:999px; padding:11px 17px; background:#192433; color:var(--text); cursor:pointer; font-weight:750 }}
    .tab.active {{ background:var(--cyan); color:#061313 }}
    .stats {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; margin-top:20px }}
    .stat {{ padding:14px; border-radius:14px; background:#0d141d; border:1px solid var(--line) }}
    .stat b {{ display:block; color:var(--cyan); font-size:24px }}
    .stat span {{ color:var(--muted); font-size:12px }}
    .note {{ margin-top:16px; padding:13px 15px; border-left:3px solid var(--amber); background:#171716; color:#e6dbc9 }}
    nav {{ display:flex; flex-wrap:wrap; gap:8px; margin:16px 0 20px }}
    nav a {{ color:var(--text); text-decoration:none; border:1px solid var(--line); border-radius:999px; padding:7px 12px }}
    .dataset-view[hidden] {{ display:none }}
    .case {{ padding:18px; margin:14px 0; scroll-margin-top:90px }}
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
    #replay {{ position:fixed; right:22px; bottom:20px; z-index:7; border:0; border-radius:999px;
      padding:13px 18px; background:var(--cyan); color:#061313; font-weight:800; cursor:pointer; box-shadow:0 8px 30px #0008 }}
    @media(max-width:900px) {{ .videos {{ grid-template-columns:1fr }} .case header {{ display:block }}
      .case header p {{ text-align:left; margin-top:10px }} .tabs {{ width:100%; border-radius:18px }} .tab {{ flex:1 }} }}
  </style>
</head>
<body>
<main>
  <section class="hero">
    <h1>Full-SA + No-Object · 训练数据窗口</h1>
    <p class="lead">同一模型结构的两个 100% 数据集消融。两者都用 prefix 采样读取 frame 0–48，共 49 帧；frame 0–7 为 8 帧干净条件，loss 排除 context latent，只监督非条件 latent 时间位置。</p>
    <div class="stats">
      <div class="stat"><b>49 帧</b><span>统一训练窗口，frame 0–48</span></div>
      <div class="stat"><b>8 帧</b><span>统一条件窗口，frame 0–7</span></div>
      <div class="stat"><b>41 帧</b><span>条件后的像素时间区间</span></div>
      <div class="stat"><b>512×896</b><span>统一训练预处理分辨率</span></div>
    </div>
  </section>

  <div class="tabs" role="tablist">
    <button class="tab active" data-target="pybullet" role="tab">PyBullet 100%</button>
    <button class="tab" data-target="kubric" role="tab">Kubric 100%</button>
  </div>

  <section class="dataset-view" id="pybullet" data-view="pybullet">
    <div class="dataset-summary">
      <h2>Full-SA + No-Object (PyBullet 100%)</h2>
      <p>原始源视频统一为 90 帧、30 FPS。训练采用确定性 90/5/5 train/val/test 切分，并从每条 train 视频取前 49 帧。</p>
      <div class="stats">
        <div class="stat"><b>{pybullet_split_counts['train']}</b><span>train split 样本</span></div>
        <div class="stat"><b>{sum(pybullet_split_counts.values())}</b><span>完整 manifest 样本</span></div>
        <div class="stat"><b>90 帧</b><span>原始视频长度</span></div>
        <div class="stat"><b>30 FPS</b><span>训练窗口时长 1.633 秒</span></div>
        <div class="stat"><b>10</b><span>F1–F10 family</span></div>
      </div>
      <div class="note">train/val/test = {pybullet_split_counts['train']}/{pybullet_split_counts['val']}/{pybullet_split_counts['test']}；源帧分布：{html.escape(pybullet_source_summary)}。下方每个 family 展示一个真实 train 样本。</div>
    </div>
    <nav>{dataset_nav(pybullet_rows)}</nav>
    {''.join(case_card(row) for row in pybullet_rows)}
  </section>

  <section class="dataset-view" id="kubric" data-view="kubric" hidden>
    <div class="dataset-summary">
      <h2>Full-SA + No-Object (Kubric 100%)</h2>
      <p>PhyCo Kubric replay 索引使用 69 帧作为最低可用长度；进入训练后 Dataset 将输出长度重设为 49 帧、条件长度重设为 8 帧，并按 prefix 读取前缀。代表视频为 24 FPS。</p>
      <div class="stats">
        <div class="stat"><b>{sum(kubric_scenario_counts.values())}</b><span>精确 train replay 索引样本</span></div>
        <div class="stat"><b>{len(kubric_scenario_counts)}</b><span>训练 scenario</span></div>
        <div class="stat"><b>≥69 帧</b><span>构建 replay 索引的门槛</span></div>
        <div class="stat"><b>97/98 帧</b><span>源视频主要长度</span></div>
        <div class="stat"><b>24 FPS</b><span>训练窗口时长 2.042 秒</span></div>
      </div>
      <div class="note">索引中的 frame-count hint 分布：{html.escape(kubric_hint_summary)}。下方按全部 {len(kubric_scenario_counts)} 个 scenario 各展示一个实际存在的 train 样本。</div>
    </div>
    <nav>{dataset_nav(kubric_rows)}</nav>
    {''.join(case_card(row) for row in kubric_rows)}
  </section>
</main>
<button id="replay">↻ 当前数据集全部重播</button>
<script>
  for(const video of document.querySelectorAll('video[data-end]')){{
    video.addEventListener('timeupdate',()=>{{
      const end=Number(video.dataset.end);
      if(video.currentTime>=end){{ video.pause(); video.currentTime=0; }}
    }});
  }}
  const tabs=[...document.querySelectorAll('.tab')];
  const views=[...document.querySelectorAll('.dataset-view')];
  for(const tab of tabs){{
    tab.addEventListener('click',()=>{{
      for(const video of document.querySelectorAll('video')) video.pause();
      for(const item of tabs) item.classList.toggle('active',item===tab);
      for(const view of views) view.hidden=view.dataset.view!==tab.dataset.target;
      history.replaceState(null,'',`#${{tab.dataset.target}}`);
      window.scrollTo({{top:document.querySelector('.tabs').offsetTop-12,behavior:'smooth'}});
    }});
  }}
  document.getElementById('replay').addEventListener('click',()=>{{
    const active=document.querySelector('.dataset-view:not([hidden])');
    for(const video of active.querySelectorAll('video')){{ video.currentTime=0; video.play().catch(()=>{{}}); }}
  }});
  const initial=location.hash.slice(1);
  if(initial==='kubric') document.querySelector('[data-target="kubric"]').click();
</script>
</body>
</html>
"""


def load_pybullet_rows(root: Path) -> tuple[list[dict], Counter, Counter]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    split_counts = Counter(
        stable_split(f"{row['family_key']}/{row['case_id']}") for row in manifest
    )
    source_counts = Counter(
        int(row.get("motion_metrics", {}).get("frame_count", 0)) for row in manifest
    )
    selected: dict[str, dict] = {}
    for row in manifest:
        family = str(row["family_key"])
        key = f"{family}/{row['case_id']}"
        if family not in selected and stable_split(key) == "train":
            selected[family] = row
    chosen = sorted(selected.values(), key=lambda row: int(str(row["family_key"])[1:]))
    if len(chosen) != 10:
        raise RuntimeError(f"Expected one train sample for F1-F10, found {len(chosen)}")
    rows = []
    for row in chosen:
        frames = int(row.get("motion_metrics", {}).get("frame_count", 90))
        fps = float(row.get("motion_metrics", {}).get("fps", 30.0))
        rows.append(
            {
                "anchor": f"pybullet-{row['family_key']}",
                "group": str(row["family_key"]),
                "case_id": str(row["case_id"]),
                "caption": str(row.get("caption", "")),
                "src": (
                    f"pybullet_dataset/cases/{row['family_key']}/{row['case_id']}/"
                    f"videos/{row['case_id']}.mp4"
                ),
                "source_frames": frames,
                "fps": fps,
            }
        )
    return rows, split_counts, source_counts


def load_kubric_rows(root: Path, index_path: Path) -> tuple[list[dict], Counter, Counter]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    samples = payload["samples"]
    scenario_counts = Counter(str(row["scenario"]) for row in samples)
    hint_counts = Counter(
        str(row["frame_count_hint"])
        if row.get("frame_count_hint") is not None
        else "unknown"
        for row in samples
    )
    selected: dict[str, dict] = {}
    for row in samples:
        scenario = str(row["scenario"])
        if scenario not in selected and Path(row["video_path"]).is_file():
            selected[scenario] = row
    if set(selected) != set(scenario_counts):
        missing = sorted(set(scenario_counts) - set(selected))
        raise RuntimeError(f"No existing representative video for scenarios: {missing}")

    rows = []
    for scenario, row in sorted(selected.items()):
        video_path = Path(row["video_path"]).resolve()
        source_frames, fps = probe_video(video_path)
        relative_path = video_path.relative_to(root).as_posix()
        rows.append(
            {
                "anchor": f"kubric-{scenario}",
                "group": scenario,
                "case_id": f"{row['date']}/{row['sample_id']}",
                "caption": str(row["prompt"]),
                "src": f"kubric_dataset/{relative_path}",
                "source_frames": source_frames,
                "fps": fps,
            }
        )
    return rows, scenario_counts, hint_counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pybullet-root", type=Path, default=DEFAULT_PYBULLET_ROOT)
    parser.add_argument("--kubric-root", type=Path, default=DEFAULT_KUBRIC_ROOT)
    parser.add_argument("--kubric-index", type=Path, default=DEFAULT_KUBRIC_INDEX)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    pybullet_root = args.pybullet_root.expanduser().resolve()
    kubric_root = args.kubric_root.expanduser().resolve()
    kubric_index = args.kubric_index.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    pybullet_rows, pybullet_splits, pybullet_source_counts = load_pybullet_rows(
        pybullet_root
    )
    kubric_rows, kubric_scenarios, kubric_hint_counts = load_kubric_rows(
        kubric_root, kubric_index
    )

    output_root.mkdir(parents=True, exist_ok=True)
    ensure_dataset_link(output_root / "pybullet_dataset", pybullet_root)
    ensure_dataset_link(output_root / "kubric_dataset", kubric_root)
    legacy_link = output_root / "dataset"
    if legacy_link.is_symlink() and legacy_link.resolve() == pybullet_root:
        pass

    (output_root / "index.html").write_text(
        build_page(
            pybullet_rows,
            kubric_rows,
            pybullet_splits,
            pybullet_source_counts,
            kubric_scenarios,
            kubric_hint_counts,
        ),
        encoding="utf-8",
    )
    (output_root / "viewer_manifest.json").write_text(
        json.dumps(
            {
                "training_frames": TRAIN_FRAMES,
                "context_frames": CONTEXT_FRAMES,
                "pybullet": {
                    "dataset_root": str(pybullet_root),
                    "split_counts": dict(pybullet_splits),
                    "selected_cases": pybullet_rows,
                },
                "kubric": {
                    "dataset_root": str(kubric_root),
                    "index": str(kubric_index),
                    "train_samples": sum(kubric_scenarios.values()),
                    "scenario_counts": dict(kubric_scenarios),
                    "selected_cases": kubric_rows,
                },
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
