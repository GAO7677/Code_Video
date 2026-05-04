#!/usr/bin/env python3
"""Build a compact local visualization portal for data_summary/sum0504."""

from __future__ import annotations

import argparse
import html
import json
import subprocess
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import imageio_ffmpeg


DEFAULT_SUMMARY_ROOT = Path("/home/gaoya/Code_Video/Code_data/data0417/data_summary/sum0504")
DEFAULT_OUTPUT_DIR = Path("/home/gaoya/portal_hub_sim/sum0504_portal")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def find_meta_path(sample_dir: Path) -> Path | None:
    for name in ("meta.json", "metadata.json"):
        candidate = sample_dir / name
        if candidate.exists():
            return candidate
    return None


def transcode_video(src_video: Path, dst_video: Path) -> str:
    if not src_video.exists():
        return ""
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    if dst_video.exists():
        return str(dst_video)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(src_video),
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dst_video),
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0 or not dst_video.exists():
        return ""
    return str(dst_video)


def build_video_from_frames(frame_dir: Path, dst_video: Path) -> str:
    frames = sorted(frame_dir.glob("frame_*.png"))
    if not frames:
        return ""
    dst_video.parent.mkdir(parents=True, exist_ok=True)
    if dst_video.exists():
        return str(dst_video)
    writer = imageio.get_writer(
        str(dst_video),
        fps=8,
        codec="libx264",
        format="FFMPEG",
        pixelformat="yuv420p",
        macro_block_size=1,
    )
    try:
        for frame in frames:
            writer.append_data(imageio.imread(frame))
    finally:
        writer.close()
    return str(dst_video) if dst_video.exists() else ""


def pick_media(sample_dir: Path, meta: dict[str, Any], output_dir: Path, asset_prefix: str) -> list[dict[str, str]]:
    cards: list[dict[str, str]] = []
    paths = meta.get("paths", {}) if isinstance(meta.get("paths"), dict) else {}
    raw_outputs = meta.get("outputs", {}) if isinstance(meta.get("outputs"), dict) else {}

    window_keys = [
        ("Context", paths.get("context_video_path")),
        ("Future", paths.get("future_gt_video_path")),
        ("Full", paths.get("full_video_path")),
    ]
    for label, video_path in window_keys:
        if video_path and Path(video_path).exists():
            mp4_path = output_dir / "assets" / asset_prefix / f"{label.lower()}.mp4"
            built = transcode_video(Path(video_path), mp4_path)
            cards.append({"label": label, "kind": "video", "path": built or str(Path(video_path))})
    if cards:
        return cards

    raw_video = None
    if "rgb_video" in raw_outputs:
        raw_video = sample_dir / str(raw_outputs["rgb_video"])
    elif (sample_dir / "videos" / "rgb.mp4").exists():
        raw_video = sample_dir / "videos" / "rgb.mp4"
    elif (sample_dir / "rgb.mp4").exists():
        raw_video = sample_dir / "rgb.mp4"
    if raw_video is not None and raw_video.exists():
        mp4_path = output_dir / "assets" / asset_prefix / "raw.mp4"
        built = transcode_video(raw_video, mp4_path)
        cards.append({"label": "Raw", "kind": "video", "path": built or str(raw_video)})
        return cards

    built = build_video_from_frames(sample_dir / "rgb", output_dir / "assets" / asset_prefix / "raw.mp4")
    if built:
        cards.append({"label": "Raw", "kind": "video", "path": built})
    return cards


def parse_case_name(sample_name: str) -> str:
    for part in sample_name.split("__")[1:]:
        if part.startswith("case"):
            return part
    return ""


def infer_view_type(sample_dir: Path, meta: dict[str, Any]) -> str:
    value = str(meta.get("view_type") or "").strip()
    if value:
        return value
    sample_text = str(sample_dir)
    if "/stage1adapter/" in sample_text:
        return "window"
    return "raw"


def build_record(sample_dir: Path, output_dir: Path, asset_prefix: str) -> dict[str, Any]:
    meta_path = find_meta_path(sample_dir)
    meta = load_json(meta_path) if meta_path is not None else {}
    media = pick_media(sample_dir, meta, output_dir, asset_prefix)
    return {
        "sample_dir": str(sample_dir),
        "sample_name": sample_dir.name,
        "case_name": parse_case_name(sample_dir.name),
        "caption": str(meta.get("caption") or meta.get("prompt") or ""),
        "detail_caption": str(meta.get("detail_caption") or meta.get("description") or ""),
        "dataset": str(meta.get("dataset") or meta.get("dataset_name") or meta.get("dataset_source") or ""),
        "view_type": infer_view_type(sample_dir, meta),
        "media": media,
    }


def build_html(groups: list[dict[str, Any]]) -> str:
    tree: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    for group in groups:
        tree.setdefault(group["split"], {}).setdefault(group["simulator_type"], {}).setdefault(group["count_bucket"], []).append(group)

    tree_parts = []
    for split in sorted(tree):
        sim_blocks = []
        for simulator_type in sorted(tree[split]):
            count_blocks = []
            for count_bucket in sorted(tree[split][simulator_type]):
                leaves = "\n".join(
                    f"""
                    <li>
                      <button class="tree-leaf" type="button" data-target="{html.escape(group['slug'])}" title="{html.escape(group['title'])}">
                        <span class="leaf-name">{html.escape(group['collision_bucket'])}</span>
                        <span class="leaf-count">{group['total']}</span>
                      </button>
                    </li>
                    """
                    for group in tree[split][simulator_type][count_bucket]
                )
                count_blocks.append(
                    f"""
                    <section class="tree-branch" data-branch-level="count">
                      <button class="tree-toggle" type="button" aria-expanded="true">
                        <span>{html.escape(count_bucket)}</span>
                        <span class="branch-count">{len(tree[split][simulator_type][count_bucket])}</span>
                      </button>
                      <ul class="tree-list">{leaves}</ul>
                    </section>
                    """
                )
            sim_blocks.append(
                f"""
                <section class="tree-branch" data-branch-level="simulator">
                  <button class="tree-toggle" type="button" aria-expanded="true">
                    <span>{html.escape(simulator_type)}</span>
                    <span class="branch-count">{sum(len(v) for v in tree[split][simulator_type].values())}</span>
                  </button>
                  <div class="tree-children">{''.join(count_blocks)}</div>
                </section>
                """
            )
        tree_parts.append(
            f"""
            <section class="tree-branch" data-branch-level="split">
              <button class="tree-toggle" type="button" aria-expanded="true">
                <span>{html.escape(split)}</span>
                <span class="branch-count">{sum(sum(len(v) for v in sim.values()) for sim in tree[split].values())}</span>
              </button>
              <div class="tree-children">{''.join(sim_blocks)}</div>
            </section>
            """
        )

    sections = []
    for group in groups:
        cards = []
        for item in group["items"]:
            media_html = []
            for media in item["media"]:
                media_html.append(
                    f"<div class='media-block'><div class='media-title'>{html.escape(media['label'])}</div>"
                    f"<video controls preload='metadata' playsinline src='{html.escape(media['path'])}'></video></div>"
                )
            cards.append(
                f"""
                <article class="card">
                  <div class="card-head">
                    <h3>{html.escape(item['sample_name'])}</h3>
                    <div class="pill-row">
                      <span class="pill">{html.escape(item['dataset'] or 'Unknown')}</span>
                      <span class="pill">{html.escape(item['view_type'] or 'unknown')}</span>
                      <span class="pill">{html.escape(item['case_name'] or 'no_case')}</span>
                    </div>
                  </div>
                  <p class="path">{html.escape(item['sample_dir'])}</p>
                  <div class="caption-box">
                    <div class="caption-title">Raw Caption</div>
                    <p>{html.escape(item['caption'] or '<missing>')}</p>
                  </div>
                  <div class="media-grid">{''.join(media_html) if media_html else "<div class='missing'>media missing</div>"}</div>
                </article>
                """
            )
        sections.append(
            f"""
            <section
              id="{html.escape(group['slug'])}"
              class="group"
              data-slug="{html.escape(group['slug'])}"
              data-title="{html.escape(group['title'].lower())}"
            >
              <div class="group-head">
                <div>
                  <h2>{html.escape(group['title'])}</h2>
                  <p class="group-sub">{html.escape(group['subtitle'])}</p>
                </div>
                <p>{group['shown']} / {group['total']} samples</p>
              </div>
              <div class="cards">{''.join(cards)}</div>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>sum0504 Portal</title>
  <style>
    :root {{
      --bg: #f3efe8;
      --panel: #fffdf8;
      --ink: #1f1c18;
      --muted: #6d6459;
      --line: #d8cbbb;
      --accent: #8a542d;
      --accent-soft: #f7ebdd;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, rgba(138,84,45,0.10), transparent 24%),
        linear-gradient(180deg, #faf6ef 0%, var(--bg) 100%);
    }}
    header {{
      background: rgba(255,253,248,0.94);
      border-bottom: 1px solid var(--line);
      padding: 14px 18px;
    }}
    h1 {{ margin: 0 0 6px; font-size: 26px; }}
    .sub {{ margin: 0; color: var(--muted); font-size: 13px; }}
    .layout {{
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      min-height: calc(100vh - 76px);
    }}
    aside {{
      position: sticky;
      top: 0;
      align-self: start;
      height: calc(100vh - 1px);
      overflow: auto;
      border-right: 1px solid var(--line);
      background: rgba(255, 251, 244, 0.95);
      backdrop-filter: blur(10px);
      padding: 14px 12px 18px;
    }}
    .sidebar-title {{
      margin: 0 0 4px;
      font-size: 17px;
      color: var(--accent);
    }}
    .sidebar-sub {{
      margin: 0 0 10px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }}
    .sidebar-search {{
      width: 100%;
      margin: 0 0 10px;
      padding: 9px 10px;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: #fffdf8;
      color: var(--ink);
    }}
    .sidebar-status {{
      margin: 0 0 10px;
      color: var(--muted);
      font-size: 12px;
    }}
    .tree-root, .tree-children, .tree-list {{
      display: grid;
      gap: 8px;
    }}
    .tree-root {{ gap: 10px; }}
    .tree-branch {{
      border: 1px solid #e7d8c5;
      border-radius: 12px;
      background: #fffaf3;
      overflow: hidden;
    }}
    .tree-toggle {{
      width: 100%;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      padding: 10px 11px;
      border: 0;
      background: #fff4e7;
      color: var(--ink);
      cursor: pointer;
      font-weight: 600;
      text-transform: none;
    }}
    .tree-toggle:hover {{ background: #ffedd8; }}
    .tree-children {{ padding: 8px; }}
    .tree-list {{
      list-style: none;
      margin: 0;
      padding: 8px;
    }}
    .branch-count, .leaf-count {{
      min-width: 26px;
      text-align: center;
      padding: 2px 8px;
      border-radius: 999px;
      background: rgba(138,84,45,0.12);
      font-size: 12px;
      color: var(--accent);
    }}
    .tree-leaf {{
      width: 100%;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      text-align: left;
      padding: 9px 10px;
      border-radius: 10px;
      border: 1px solid #eadcca;
      background: #fffdf8;
      color: var(--ink);
      cursor: pointer;
    }}
    .tree-leaf:hover {{
      border-color: #c89969;
      background: #fff8ef;
    }}
    .tree-leaf.active {{
      border-color: var(--accent);
      background: #fcefe0;
      box-shadow: inset 0 0 0 1px rgba(138,84,45,0.18);
    }}
    .leaf-name {{ word-break: break-word; line-height: 1.25; }}
    .is-collapsed > .tree-children,
    .is-collapsed > .tree-list {{ display: none; }}
    main {{
      padding: 16px 18px 28px;
      min-width: 0;
    }}
    .content-head {{
      position: sticky;
      top: 0;
      z-index: 5;
      background: rgba(243,239,232,0.94);
      backdrop-filter: blur(8px);
      padding: 2px 0 12px;
      margin-bottom: 10px;
      border-bottom: 1px solid rgba(216,203,187,0.7);
    }}
    .content-head h2 {{
      margin: 0 0 5px;
      font-size: 20px;
    }}
    .content-meta {{
      margin: 0;
      color: var(--muted);
      font-size: 12px;
    }}
    .group {{ display: none; }}
    .group.active {{ display: block; }}
    .group-head {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
      margin-bottom: 10px;
    }}
    .group-head h2 {{ margin: 0 0 4px; font-size: 19px; }}
    .group-sub {{ margin: 0; color: var(--muted); font-size: 12px; }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      box-shadow: 0 8px 20px rgba(45, 30, 12, 0.04);
    }}
    .card-head {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: start;
    }}
    .card h3 {{
      margin: 0;
      font-size: 15px;
      line-height: 1.25;
      flex: 1;
      min-width: 0;
    }}
    .pill-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      justify-content: end;
      max-width: 46%;
    }}
    .pill {{
      display: inline-block;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      border: 1px solid #ead8c7;
      padding: 2px 8px;
      font-size: 11px;
      white-space: nowrap;
    }}
    .path {{
      margin: 8px 0 10px;
      color: var(--muted);
      font-size: 11px;
      word-break: break-all;
    }}
    .caption-box {{
      background: #fff8ef;
      border: 1px solid #e8dccb;
      border-radius: 10px;
      padding: 9px 10px;
    }}
    .caption-title, .media-title {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--accent);
      margin-bottom: 5px;
    }}
    .caption-box p {{
      margin: 0;
      white-space: pre-wrap;
      line-height: 1.35;
      font-size: 12px;
    }}
    .media-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      margin-top: 10px;
    }}
    video {{
      width: 100%;
      display: block;
      border-radius: 10px;
      border: 1px solid #d7cab8;
      background: #000;
    }}
    .missing {{
      border: 1px dashed var(--line);
      border-radius: 10px;
      min-height: 120px;
      display: grid;
      place-items: center;
      color: var(--muted);
      background: #faf5ed;
      font-size: 12px;
    }}
    @media (max-width: 1200px) {{
      .cards {{ grid-template-columns: 1fr; }}
    }}
    @media (max-width: 1024px) {{
      .layout {{ grid-template-columns: 1fr; }}
      aside {{
        position: static;
        height: auto;
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }}
      .content-head {{
        position: static;
        backdrop-filter: none;
      }}
      .media-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>sum0504 Overview</h1>
    <p class="sub">按 `train/val/test -> rigid -> count_bucket -> collision_bucket` 浏览，每个叶子类展示前 10 个样本。</p>
  </header>
  <div class="layout">
    <aside>
      <h2 class="sidebar-title">Category Tree</h2>
      <p class="sidebar-sub">左侧按层级切换，右侧只显示当前类别；媒体统一为浏览器兼容 mp4。</p>
      <input id="nav-search" class="sidebar-search" type="text" placeholder="search leaf or count bucket">
      <p id="nav-status" class="sidebar-status"></p>
      <div class="tree-root">{''.join(tree_parts)}</div>
    </aside>
    <main>
      <div class="content-head">
        <h2 id="content-title">Select a category</h2>
        <p id="content-meta" class="content-meta"></p>
      </div>
      {''.join(sections)}
    </main>
  </div>
  <script>
    const qs = (id) => document.getElementById(id);
    const sections = Array.from(document.querySelectorAll(".group"));
    const leaves = Array.from(document.querySelectorAll(".tree-leaf"));
    const toggles = Array.from(document.querySelectorAll(".tree-toggle"));
    const navSearch = qs("nav-search");
    const navStatus = qs("nav-status");
    const contentTitle = qs("content-title");
    const contentMeta = qs("content-meta");

    function updateNavStatus() {{
      const visibleLeaves = leaves.filter((leaf) => leaf.closest("li").style.display !== "none").length;
      navStatus.textContent = `Showing ${{visibleLeaves}} / ${{leaves.length}} categories`;
    }}

    function setActive(slug) {{
      let matched = false;
      for (const section of sections) {{
        const isActive = section.dataset.slug === slug;
        section.classList.toggle("active", isActive);
        matched = matched || isActive;
        if (isActive) {{
          contentTitle.textContent = section.querySelector("h2").textContent;
          contentMeta.textContent = section.querySelector(".group-sub").textContent;
        }}
      }}
      for (const leaf of leaves) {{
        leaf.classList.toggle("active", leaf.dataset.target === slug);
      }}
      if (matched) {{
        window.location.hash = slug;
      }}
    }}

    function applySearch() {{
      const query = navSearch.value.trim().toLowerCase();
      for (const leaf of leaves) {{
        const text = leaf.textContent.toLowerCase();
        leaf.closest("li").style.display = (!query || text.includes(query)) ? "" : "none";
      }}
      for (const branch of Array.from(document.querySelectorAll(".tree-branch"))) {{
        const hasVisibleLeaf = Array.from(branch.querySelectorAll(":scope li")).some((li) => li.style.display !== "none");
        if (branch.dataset.branchLevel === "count") {{
          branch.style.display = hasVisibleLeaf ? "" : "none";
        }}
      }}
      for (const level of ["simulator", "split"]) {{
        for (const branch of Array.from(document.querySelectorAll(`.tree-branch[data-branch-level="${{level}}"]`))) {{
          const hasVisibleChild = Array.from(branch.querySelectorAll(".tree-branch, li")).some((node) => node.style.display !== "none");
          branch.style.display = hasVisibleChild ? "" : "none";
        }}
      }}
      updateNavStatus();
    }}

    for (const toggle of toggles) {{
      toggle.addEventListener("click", () => {{
        const branch = toggle.parentElement;
        const expanded = toggle.getAttribute("aria-expanded") === "true";
        toggle.setAttribute("aria-expanded", expanded ? "false" : "true");
        branch.classList.toggle("is-collapsed", expanded);
      }});
    }}

    for (const leaf of leaves) {{
      leaf.addEventListener("click", () => setActive(leaf.dataset.target));
    }}

    navSearch.addEventListener("input", applySearch);
    applySearch();

    const validSlugs = new Set(sections.map((section) => section.dataset.slug));
    const initialSlug = decodeURIComponent(window.location.hash.replace(/^#/, ""));
    const defaultLeaf = leaves.find((leaf) => validSlugs.has(leaf.dataset.target));
    setActive(validSlugs.has(initialSlug) ? initialSlug : (defaultLeaf ? defaultLeaf.dataset.target : ""));
  </script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build visualization portal for sum0504 categories")
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit_per_group", type=int, default=10)
    args = parser.parse_args()

    summary_root = args.summary_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    groups = []
    for list_path in sorted(summary_root.rglob("samples.txt")):
        rel_parts = list_path.relative_to(summary_root).parts
        if len(rel_parts) != 5:
            continue
        split, simulator_type, count_bucket, collision_bucket, _ = rel_parts
        data = [line.strip() for line in list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        slug = "__".join([split, simulator_type, count_bucket, collision_bucket])
        title = f"{split}/{simulator_type}/{count_bucket}/{collision_bucket}"
        subtitle = f"split={split} | simulator={simulator_type} | count={count_bucket} | collision={collision_bucket}"
        items = []
        for idx, item in enumerate(data[: max(0, args.limit_per_group)]):
            sample_dir = Path(str(item))
            asset_prefix = f"{slug}/{idx:03d}_{sample_dir.name}"
            items.append(build_record(sample_dir, args.output_dir, asset_prefix))
        groups.append(
            {
                "slug": slug,
                "title": title,
                "subtitle": subtitle,
                "split": split,
                "simulator_type": simulator_type,
                "count_bucket": count_bucket,
                "collision_bucket": collision_bucket,
                "total": len(data),
                "shown": len(items),
                "items": items,
            }
        )

    (args.output_dir / "index.html").write_text(build_html(groups), encoding="utf-8")
    (args.output_dir / "manifest.json").write_text(json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output_dir / "index.html")


if __name__ == "__main__":
    main()
