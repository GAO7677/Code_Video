#!/usr/bin/env python3
"""Build a local visualization portal for all by_raw_window categories."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio


DEFAULT_SUMMARY_ROOT = Path("/home/gaoya/Code_Video/Code_data/data0417/data_summary/by_raw_window")
DEFAULT_OUTPUT_DIR = Path("/home/gaoya/portal_hub_sim/by_raw_window_portal")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def find_meta_path(sample_dir: Path) -> Path | None:
    for name in ("meta.json", "metadata.json"):
        candidate = sample_dir / name
        if candidate.exists():
            return candidate
    return None


def build_gif_from_video(src_video: Path, dst_gif: Path) -> str:
    if not src_video.exists():
        return ""
    dst_gif.parent.mkdir(parents=True, exist_ok=True)
    if dst_gif.exists():
        return str(dst_gif)
    reader = imageio.get_reader(str(src_video))
    frames = []
    try:
        for frame in reader:
            frames.append(frame)
    finally:
        reader.close()
    if not frames:
        return ""
    imageio.mimsave(str(dst_gif), frames, format="GIF", fps=8)
    return str(dst_gif)


def build_gif_from_frames(frame_dir: Path, dst_gif: Path) -> str:
    frames = sorted(frame_dir.glob("frame_*.png"))
    if not frames:
        return ""
    dst_gif.parent.mkdir(parents=True, exist_ok=True)
    if dst_gif.exists():
        return str(dst_gif)
    images = [imageio.imread(frame) for frame in frames]
    imageio.mimsave(str(dst_gif), images, format="GIF", fps=8)
    return str(dst_gif)


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
            gif_path = output_dir / "assets" / asset_prefix / f"{label.lower()}.gif"
            cards.append({"label": label, "kind": "gif", "path": build_gif_from_video(Path(video_path), gif_path)})
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
        cards.append({"label": "Raw Video", "kind": "video", "path": str(raw_video)})
        return cards

    gif = build_gif_from_frames(sample_dir / "rgb", output_dir / "assets" / asset_prefix / "raw.gif")
    if gif:
        cards.append({"label": "Raw GIF", "kind": "gif", "path": gif})
    return cards


def build_record(sample_dir: Path, output_dir: Path, asset_prefix: str) -> dict[str, Any]:
    meta_path = find_meta_path(sample_dir)
    meta = load_json(meta_path) if meta_path is not None else {}
    media = pick_media(sample_dir, meta, output_dir, asset_prefix)
    return {
        "sample_dir": str(sample_dir),
        "sample_name": sample_dir.name,
        "caption": str(meta.get("caption") or meta.get("prompt") or ""),
        "detail_caption": str(meta.get("detail_caption") or meta.get("description") or ""),
        "dataset": str(meta.get("dataset") or meta.get("dataset_name") or meta.get("dataset_source") or ""),
        "view_type": str(meta.get("view_type") or ""),
        "media": media,
    }


def build_html(groups: list[dict[str, Any]]) -> str:
    tree: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for group in groups:
        tree.setdefault(group["view_group"], {}).setdefault(group["split_group"], []).append(group)

    tree_parts = []
    for view_group in sorted(tree):
        split_blocks = []
        for split_group in sorted(tree[view_group]):
            leaves = "\n".join(
                f"""
                <li>
                  <button
                    class="tree-leaf"
                    type="button"
                    data-target="{html.escape(group['slug'])}"
                    title="{html.escape(group['title'])}"
                  >
                    <span class="leaf-name">{html.escape(group['title'].split('/')[-1])}</span>
                    <span class="leaf-count">{group['total']}</span>
                  </button>
                </li>
                """
                for group in tree[view_group][split_group]
            )
            split_blocks.append(
                f"""
                <section class="tree-branch" data-branch-level="split">
                  <button class="tree-toggle" type="button" aria-expanded="true">
                    <span>{html.escape(split_group)}</span>
                    <span class="branch-count">{len(tree[view_group][split_group])}</span>
                  </button>
                  <ul class="tree-list">
                    {leaves}
                  </ul>
                </section>
                """
            )
        tree_parts.append(
            f"""
            <section class="tree-branch" data-branch-level="view">
              <button class="tree-toggle" type="button" aria-expanded="true">
                <span>{html.escape(view_group)}</span>
                <span class="branch-count">{sum(len(v) for v in tree[view_group].values())}</span>
              </button>
              <div class="tree-children">
                {''.join(split_blocks)}
              </div>
            </section>
            """
        )

    sections = []
    for group in groups:
        cards = []
        for item in group["items"]:
            media_html = []
            for media in item["media"]:
                if media["kind"] == "video":
                    media_html.append(
                        f"<div class='media-block'><div class='media-title'>{html.escape(media['label'])}</div>"
                        f"<video controls preload='metadata' playsinline src='{html.escape(media['path'])}'></video></div>"
                    )
                elif media["kind"] == "gif":
                    media_html.append(
                        f"<div class='media-block'><div class='media-title'>{html.escape(media['label'])}</div>"
                        f"<img class='gif-preview' src='{html.escape(media['path'])}' alt='{html.escape(media['label'])}'></div>"
                    )
            media_block = "".join(media_html) if media_html else "<div class='missing'>media missing</div>"
            cards.append(
                f"""
                <article class="card">
                  <h3>{html.escape(item['sample_name'])}</h3>
                  <p class="path">{html.escape(item['sample_dir'])}</p>
                  <p class="meta-line">dataset={html.escape(item['dataset'])} | view_type={html.escape(item['view_type'])}</p>
                  <div class="captions">
                    <div class="caption-box">
                      <div class="caption-title">Caption</div>
                      <p>{html.escape(item['caption'] or '<missing>')}</p>
                    </div>
                    <div class="caption-box">
                      <div class="caption-title">Detail Caption</div>
                      <p>{html.escape(item['detail_caption'] or '<missing>')}</p>
                    </div>
                  </div>
                  <div class="media-grid">{media_block}</div>
                </article>
                """
            )
        sections.append(
            f"""
            <section
              id="{html.escape(group['slug'])}"
              class="group"
              data-slug="{html.escape(group['slug'])}"
              data-view-group="{html.escape(group['view_group'])}"
              data-split-group="{html.escape(group['split_group'])}"
              data-title="{html.escape(group['title'].lower())}"
            >
              <div class="group-head">
                <h2>{html.escape(group['title'])}</h2>
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
  <title>By Raw Window Portal</title>
  <style>
    :root {{
      --bg: #f4f1eb;
      --panel: #fffdf8;
      --ink: #201d19;
      --muted: #6c655c;
      --line: #d8ccbc;
      --accent: #8d4d20;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, rgba(141,77,32,0.08), transparent 26%),
        linear-gradient(180deg, #faf7f0 0%, var(--bg) 100%);
    }}
    header {{
      background: rgba(255,253,248,0.94);
      border-bottom: 1px solid var(--line);
      padding: 16px 22px;
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    .sub {{ margin: 0; color: var(--muted); }}
    .layout {{
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      min-height: calc(100vh - 88px);
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
      padding: 18px 16px 24px;
    }}
    .sidebar-head {{
      margin-bottom: 14px;
    }}
    .sidebar-title {{
      margin: 0 0 4px;
      font-size: 18px;
      color: var(--accent);
    }}
    .sidebar-sub {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }}
    .sidebar-search {{
      width: 100%;
      margin: 14px 0 12px;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid var(--line);
      background: #fffdf8;
      color: var(--ink);
    }}
    .sidebar-status {{
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 12px;
    }}
    .tree-root {{
      display: grid;
      gap: 10px;
    }}
    .tree-branch {{
      border: 1px solid #e7d8c5;
      border-radius: 14px;
      background: #fffaf3;
      overflow: hidden;
    }}
    .tree-toggle {{
      width: 100%;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 11px 12px;
      border: 0;
      background: #fff4e7;
      color: var(--ink);
      cursor: pointer;
      font-weight: 600;
      text-transform: capitalize;
    }}
    .tree-toggle:hover {{
      background: #ffedd8;
    }}
    .branch-count, .leaf-count {{
      min-width: 28px;
      text-align: center;
      padding: 2px 8px;
      border-radius: 999px;
      background: rgba(141,77,32,0.12);
      font-size: 12px;
      color: var(--accent);
    }}
    .tree-children {{
      display: grid;
      gap: 8px;
      padding: 10px;
    }}
    .tree-list {{
      list-style: none;
      margin: 0;
      padding: 10px;
      display: grid;
      gap: 8px;
    }}
    .tree-leaf {{
      width: 100%;
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      text-align: left;
      padding: 10px 12px;
      border-radius: 12px;
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
      box-shadow: inset 0 0 0 1px rgba(141,77,32,0.18);
    }}
    .leaf-name {{
      word-break: break-word;
      line-height: 1.35;
    }}
    .is-collapsed > .tree-children,
    .is-collapsed > .tree-list {{
      display: none;
    }}
    main {{
      padding: 20px 22px 40px;
      min-width: 0;
    }}
    .content-head {{
      position: sticky;
      top: 0;
      z-index: 5;
      background: rgba(244,241,235,0.92);
      backdrop-filter: blur(8px);
      padding: 4px 0 16px;
      margin-bottom: 8px;
      border-bottom: 1px solid rgba(216,204,188,0.7);
    }}
    .content-head h2 {{
      margin: 0 0 6px;
      font-size: 22px;
    }}
    .content-meta {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
    }}
    .group {{
      display: none;
      margin-bottom: 42px;
    }}
    .group.active {{
      display: block;
    }}
    .group-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      margin-bottom: 14px;
    }}
    .cards {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 16px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 16px;
      box-shadow: 0 10px 24px rgba(45, 30, 12, 0.05);
    }}
    .card h3 {{ margin: 0 0 6px; font-size: 18px; }}
    .path {{
      margin: 0 0 6px;
      color: var(--muted);
      font-size: 12px;
      word-break: break-all;
    }}
    .meta-line {{
      margin: 0;
      color: #7b5b43;
      font-size: 12px;
    }}
    .captions {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 12px;
    }}
    .caption-box {{
      background: #fff8ef;
      border: 1px solid #e8dccb;
      border-radius: 12px;
      padding: 10px;
    }}
    .caption-title, .media-title {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--accent);
      margin-bottom: 6px;
    }}
    .caption-box p {{
      margin: 0;
      white-space: pre-wrap;
      line-height: 1.45;
    }}
    .media-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 12px;
    }}
    video, .gif-preview {{
      width: 100%;
      display: block;
      border-radius: 12px;
      border: 1px solid #d7cab8;
      background: #000;
    }}
    .missing {{
      border: 1px dashed var(--line);
      border-radius: 12px;
      min-height: 140px;
      display: grid;
      place-items: center;
      color: var(--muted);
      background: #faf5ed;
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
      .captions, .media-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>By Raw Window Overview</h1>
    <p class="sub">Visualization portal for every classification under data_summary/by_raw_window.</p>
  </header>
  <div class="layout">
    <aside>
      <div class="sidebar-head">
        <h2 class="sidebar-title">Category Tree</h2>
        <p class="sidebar-sub">按 `raw/window -> split -> category` 浏览，每次只显示一个类别。</p>
      </div>
      <input id="nav-search" class="sidebar-search" type="text" placeholder="search category">
      <p id="nav-status" class="sidebar-status"></p>
      <div class="tree-root">
        {''.join(tree_parts)}
      </div>
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
          contentMeta.textContent = `${{section.dataset.viewGroup}} / ${{section.dataset.splitGroup}} · ${{section.querySelector(".group-head p").textContent}}`;
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
        if (branch.dataset.branchLevel === "split") {{
          branch.style.display = hasVisibleLeaf ? "" : "none";
        }}
      }}
      for (const branch of Array.from(document.querySelectorAll('.tree-branch[data-branch-level="view"]'))) {{
        const hasVisibleSplit = Array.from(branch.querySelectorAll('.tree-branch[data-branch-level="split"]')).some((node) => node.style.display !== "none");
        branch.style.display = hasVisibleSplit ? "" : "none";
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
    parser = argparse.ArgumentParser(description="Build visualization portal for by_raw_window categories")
    parser.add_argument("--summary_root", type=Path, default=DEFAULT_SUMMARY_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit_per_group", type=int, default=10)
    args = parser.parse_args()

    summary_root = args.summary_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    groups = []
    for list_path in sorted(summary_root.rglob("*.json")):
        if list_path.name == "summary.json":
            continue
        data = load_json(list_path)
        if not isinstance(data, list):
            continue
        rel = list_path.relative_to(summary_root).with_suffix("")
        slug = "__".join(rel.parts)
        rel_parts = rel.parts
        view_group = rel_parts[0] if rel_parts else "misc"
        split_group = "misc"
        if view_group in {"raw", "window"}:
            if len(rel_parts) >= 2 and rel_parts[1] in {"train", "test"}:
                split_group = rel_parts[1]
            elif len(rel_parts) >= 2 and rel_parts[1] == "benchmark":
                split_group = "benchmark"
        items = []
        for idx, item in enumerate(data[: max(0, args.limit_per_group)]):
            sample_dir = Path(str(item))
            asset_prefix = f"{slug}/{idx:03d}_{sample_dir.name}"
            items.append(build_record(sample_dir, args.output_dir, asset_prefix))
        groups.append(
            {
                "slug": slug,
                "title": rel.as_posix(),
                "total": len(data),
                "shown": len(items),
                "view_group": view_group,
                "split_group": split_group,
                "items": items,
            }
        )

    (args.output_dir / "index.html").write_text(build_html(groups), encoding="utf-8")
    (args.output_dir / "manifest.json").write_text(json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output_dir / "index.html")


if __name__ == "__main__":
    main()
