# 用途：批量检查 physics 样本并生成本地浏览页面。
"""该脚本用于批量生成物理样本的检查图并可选启动网页浏览；输入为 dataset_root 下的 Genesis 样本目录，输出为每个 sample_dir/visualizations 下的 PNG 可视化和数据集级 HTML 索引。"""
import argparse
import html
import json
import subprocess
import sys
from functools import partial
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List
from urllib.parse import urlsplit

META_FILENAMES = ("meta.json", "metadata.json")


def find_meta_path(sample_dir: Path) -> Path | None:
    for filename in META_FILENAMES:
        candidate = sample_dir / filename
        if candidate.exists():
            return candidate
    return None


@dataclass
class SampleCard:
    composition: str
    count_bucket: str
    scene_id: str
    rel_dir: str
    rgb_video: str
    summary_frames: str
    summary_state: str
    contact_timeline: str
    scene_init_html: str
    scene_init_glb: str
    motion_types: List[str]
    motion_groups: List[str]
    roles: List[str]


def is_valid_sample_dir(sample_dir: Path) -> bool:
    meta_path = find_meta_path(sample_dir)
    physics_dir = sample_dir / "physics"
    if meta_path is None or not physics_dir.exists():
        return False
    required_physics = [
        physics_dir / "rigid_kinematics.npz",
        physics_dir / "depth_metric.npy",
        physics_dir / "seg.npy",
        physics_dir / "contact_graph.npy",
    ]
    if not all(path.exists() for path in required_physics):
        return False
    rgb_dir = sample_dir / "rgb"
    video_path = sample_dir / "videos" / "rgb.mp4"
    has_rgb_frames = rgb_dir.exists() and any(rgb_dir.glob("frame_*.png"))
    return has_rgb_frames or video_path.exists()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-generate physics inspection figures and optionally serve an HTML browser for a dataset root."
    )
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--num_preview_frames", type=int, default=4)
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip samples that already have all visualization PNGs.",
    )
    parser.add_argument(
        "--serve_only",
        action="store_true",
        help="Do not regenerate figures; only rebuild the HTML index and optionally serve it.",
    )
    parser.add_argument("--serve", action="store_true", help="Start a local HTTP server after generating the summaries.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8041)
    return parser.parse_args()


def has_visualization_bundle(sample_dir: Path) -> bool:
    vis_dir = sample_dir / "visualizations"
    required = [
        vis_dir / "summary_frames.png",
        vis_dir / "summary_state.png",
        vis_dir / "contact_timeline.png",
        vis_dir / "scene_init_interactive.html",
        vis_dir / "scene_init.glb",
    ]
    return all(path.exists() for path in required)


def build_index(cards: List[SampleCard], dataset_root: Path) -> Path:
    records = [
        {
            "composition": card.composition,
            "count_bucket": card.count_bucket,
            "scene_id": card.scene_id,
            "rel_dir": card.rel_dir,
            "rgb_video": f"/{card.rgb_video}",
            "summary_frames": f"/{card.summary_frames}",
            "summary_state": f"/{card.summary_state}",
            "contact_timeline": f"/{card.contact_timeline}",
            "scene_init_html": f"/{card.scene_init_html}",
            "scene_init_glb": f"/{card.scene_init_glb}",
            "motion_types": card.motion_types,
            "motion_groups": card.motion_groups,
            "roles": card.roles,
        }
        for card in cards
    ]
    records_json = (
        json.dumps(records, ensure_ascii=False)
        .replace("</", "<\\/")
        .replace("<!--", "<\\!--")
    )

    page = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Physics Sample Browser</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --panel: rgba(255, 252, 247, 0.9);
      --ink: #1f1c17;
      --muted: #6b6256;
      --accent: #c05621;
      --accent-soft: #f2c29c;
      --border: rgba(31, 28, 23, 0.12);
      --shadow: 0 18px 48px rgba(53, 39, 24, 0.14);
      --shadow-soft: 0 10px 24px rgba(53, 39, 24, 0.1);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at top left, rgba(192, 86, 33, 0.18), transparent 24rem),
        radial-gradient(circle at top right, rgba(123, 74, 36, 0.14), transparent 28rem),
        linear-gradient(180deg, #f8f4ee 0%, var(--bg) 45%, #ede3d4 100%);
    }}
    .shell {{
      width: min(1480px, calc(100vw - 28px));
      margin: 0 auto;
      padding: 24px 0 40px;
    }}
    .hero {{
      padding: 28px;
      border: 1px solid var(--border);
      border-radius: 28px;
      background: linear-gradient(135deg, rgba(255, 250, 243, 0.95), rgba(250, 240, 227, 0.82));
      box-shadow: var(--shadow);
    }}
    h1 {{
      margin: 0;
      font-size: clamp(2rem, 3vw, 3.4rem);
      letter-spacing: 0.02em;
    }}
    .sub {{
      margin-top: 12px;
      max-width: 72rem;
      color: var(--muted);
      line-height: 1.65;
      font-size: 1rem;
    }}
    .toolbar {{
      display: grid;
      grid-template-columns: minmax(260px, 1.6fr) 220px 180px 220px 160px;
      gap: 12px;
      margin-top: 22px;
    }}
    input,
    select {{
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 14px 16px;
      font: inherit;
      color: var(--ink);
      background: rgba(255, 255, 255, 0.72);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.5);
    }}
    .stats {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 18px;
    }}
    .pill {{
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 8px 14px;
      background: rgba(255, 255, 255, 0.72);
      color: var(--muted);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 18px;
      margin-top: 24px;
      align-items: start;
    }}
    .card {{
      display: flex;
      flex-direction: column;
      gap: 12px;
      padding: 14px;
      border: 1px solid var(--border);
      border-radius: 22px;
      background: var(--panel);
      box-shadow: var(--shadow-soft);
    }}
    .card-head {{
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
    }}
    .card h3 {{
      margin: 0;
      font-size: 1.1rem;
      line-height: 1.35;
    }}
    .chip-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .chip {{
      border: 1px solid rgba(192, 86, 33, 0.18);
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 0.84rem;
      color: #8f3f1e;
      background: rgba(242, 194, 156, 0.24);
    }}
    video {{
      width: 100%;
      aspect-ratio: 4 / 3;
      max-height: 230px;
      object-fit: contain;
      border-radius: 16px;
      background: #0f0f0f;
      box-shadow: inset 0 0 0 1px rgba(255,255,255,0.08);
    }}
    .fig-grid {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 10px;
    }}
    .fig-grid a {{
      display: block;
      text-decoration: none;
    }}
    .fig-grid img {{
      width: 100%;
      aspect-ratio: 1.45 / 1;
      object-fit: cover;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.7);
    }}
    .scene-link {{
      display: flex !important;
      align-items: center;
      justify-content: center;
      min-height: 100%;
      aspect-ratio: 1.45 / 1;
      border-radius: 12px;
      border: 1px solid rgba(192, 86, 33, 0.2);
      background:
        radial-gradient(circle at top left, rgba(192, 86, 33, 0.18), transparent 55%),
        linear-gradient(135deg, rgba(255, 248, 240, 0.95), rgba(247, 231, 214, 0.88));
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.7);
      color: #7a3518;
      font-size: 0.96rem;
      font-weight: 700;
      letter-spacing: 0.02em;
      text-align: center;
      padding: 14px;
      line-height: 1.5;
    }}
    .scene-link span {{
      display: block;
    }}
    .meta {{
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .meta-line {{
      display: grid;
      gap: 4px;
    }}
    .meta-label {{
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: #8d7f70;
    }}
    code {{
      display: block;
      overflow-wrap: anywhere;
      padding: 8px 10px;
      border-radius: 10px;
      background: rgba(31, 28, 23, 0.05);
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.82rem;
    }}
    .pager {{
      display: flex;
      gap: 10px;
      align-items: center;
      justify-content: center;
      margin-top: 26px;
    }}
    button {{
      border: 0;
      border-radius: 999px;
      padding: 12px 18px;
      font: inherit;
      color: #fffaf5;
      background: linear-gradient(135deg, #c05621, #8f3f1e);
      cursor: pointer;
      box-shadow: 0 10px 22px rgba(192, 86, 33, 0.2);
    }}
    button:disabled {{
      opacity: 0.35;
      cursor: default;
      box-shadow: none;
    }}
    .empty {{
      margin-top: 20px;
      padding: 24px;
      border: 1px dashed var(--border);
      border-radius: 20px;
      text-align: center;
      color: var(--muted);
      background: rgba(255,255,255,0.55);
    }}
    @media (max-width: 980px) {{
      .toolbar {{
        grid-template-columns: 1fr;
      }}
      .shell {{
        width: min(100vw - 18px, 1480px);
      }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <h1>Physics Sample Browser</h1>
      <p class="sub">
        批量浏览刚体物理样本的视频、summary frames、状态图和接触时间线。页面风格参考 caption viewer，
        优先适合快速筛查异常案例，因此卡片尺寸做得更紧凑，视频框不会铺得过大。
      </p>
      <p class="sub"><strong>Dataset Root:</strong> {html.escape(str(dataset_root))}</p>
      <div class="toolbar">
        <input id="search" type="search" placeholder="搜索 scene / motion / role / 路径">
        <select id="composition">
          <option value="">全部 composition</option>
        </select>
        <select id="countBucket">
          <option value="">全部 count bucket</option>
        </select>
        <select id="motion">
          <option value="">全部 motion</option>
        </select>
        <select id="pageSize">
          <option value="12">每页 12 条</option>
          <option value="18" selected>每页 18 条</option>
          <option value="24">每页 24 条</option>
          <option value="36">每页 36 条</option>
        </select>
      </div>
      <div class="stats">
        <div class="pill" id="countPill">加载中</div>
        <div class="pill" id="pagePill"></div>
      </div>
    </section>
    <section id="grid" class="grid"></section>
    <section id="empty" class="empty" hidden>没有匹配结果。</section>
    <nav class="pager">
      <button id="prev">上一页</button>
      <button id="next">下一页</button>
    </nav>
  </main>
  <script id="records" type="application/json">{records_json}</script>
  <script>
    const state = {{
      items: [],
      filtered: [],
      page: 1,
      pageSize: 18,
    }};

    const grid = document.getElementById("grid");
    const empty = document.getElementById("empty");
    const search = document.getElementById("search");
    const composition = document.getElementById("composition");
    const countBucket = document.getElementById("countBucket");
    const motion = document.getElementById("motion");
    const pageSize = document.getElementById("pageSize");
    const countPill = document.getElementById("countPill");
    const pagePill = document.getElementById("pagePill");
    const prev = document.getElementById("prev");
    const next = document.getElementById("next");

    function escapeHtml(text) {{
      return String(text ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }}

    function uniqSorted(values) {{
      return [...new Set(values.filter(Boolean))].sort();
    }}

    function render() {{
      const totalPages = Math.max(1, Math.ceil(state.filtered.length / state.pageSize));
      state.page = Math.min(state.page, totalPages);
      const start = (state.page - 1) * state.pageSize;
      const visible = state.filtered.slice(start, start + state.pageSize);

      countPill.textContent = `共 ${{state.items.length}} 条，当前匹配 ${{state.filtered.length}} 条`;
      pagePill.textContent = `第 ${{state.page}} / ${{totalPages}} 页`;
      prev.disabled = state.page <= 1;
      next.disabled = state.page >= totalPages;

      if (!visible.length) {{
        grid.innerHTML = "";
        empty.hidden = false;
        return;
      }}

      empty.hidden = true;
      grid.innerHTML = visible.map((item) => {{
        const motionText = (item.motion_types || []).join(", ");
        const groupText = (item.motion_groups || []).join(", ");
        const roleText = (item.roles || []).join(", ");
        const chips = [
          item.composition,
          item.count_bucket,
        ].filter(Boolean).map((value) => `<span class="chip">${{escapeHtml(value)}}</span>`).join("");
        return `
          <article class="card">
            <div class="card-head">
              <div>
                <h3>${{escapeHtml(item.scene_id)}}</h3>
              </div>
              <div class="chip-row">${{chips}}</div>
            </div>
            <video controls preload="none" playsinline src="${{encodeURI(item.rgb_video)}}"></video>
            <div class="fig-grid">
              <a href="${{encodeURI(item.summary_frames)}}" target="_blank" rel="noreferrer"><img src="${{encodeURI(item.summary_frames)}}" alt="summary frames"></a>
              <a href="${{encodeURI(item.summary_state)}}" target="_blank" rel="noreferrer"><img src="${{encodeURI(item.summary_state)}}" alt="summary state"></a>
              <a href="${{encodeURI(item.contact_timeline)}}" target="_blank" rel="noreferrer"><img src="${{encodeURI(item.contact_timeline)}}" alt="contact timeline"></a>
              <a class="scene-link" href="${{encodeURI(item.scene_init_html)}}" target="_blank" rel="noreferrer">
                <span>打开初始化 3D Mesh 场景</span>
                <span style="font-size:0.78rem; font-weight:500; margin-top:6px;">HTML viewer / GLB</span>
              </a>
            </div>
            <div class="meta">
              <div class="meta-line">
                <div class="meta-label">Path</div>
                <code>${{escapeHtml(item.rel_dir)}}</code>
              </div>
              <div class="meta-line">
                <div class="meta-label">Motion Types</div>
                <code>${{escapeHtml(motionText)}}</code>
              </div>
              <div class="meta-line">
                <div class="meta-label">Motion Groups</div>
                <code>${{escapeHtml(groupText)}}</code>
              </div>
              <div class="meta-line">
                <div class="meta-label">Roles</div>
                <code>${{escapeHtml(roleText)}}</code>
              </div>
            </div>
          </article>
        `;
      }}).join("");
    }}

    function applyFilters() {{
      const q = search.value.trim().toLowerCase();
      const compositionValue = composition.value;
      const countValue = countBucket.value;
      const motionValue = motion.value;
      state.filtered = state.items.filter((item) => {{
        const haystack = [
          item.scene_id,
          item.rel_dir,
          item.composition,
          item.count_bucket,
          ...(item.motion_types || []),
          ...(item.motion_groups || []),
          ...(item.roles || []),
        ].join(" ").toLowerCase();
        const textOk = !q || haystack.includes(q);
        const compositionOk = !compositionValue || item.composition === compositionValue;
        const countOk = !countValue || item.count_bucket === countValue;
        const motionOk = !motionValue || (item.motion_types || []).includes(motionValue);
        return textOk && compositionOk && countOk && motionOk;
      }});
      state.page = 1;
      render();
    }}

    function fillSelect(selectEl, values) {{
      values.forEach((value) => {{
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        selectEl.appendChild(option);
      }});
    }}

    function init() {{
      const raw = document.getElementById("records").textContent || "[]";
      state.items = JSON.parse(raw);
      state.filtered = [...state.items];
      fillSelect(composition, uniqSorted(state.items.map((item) => item.composition)));
      fillSelect(countBucket, uniqSorted(state.items.map((item) => item.count_bucket)));
      fillSelect(
        motion,
        uniqSorted(state.items.flatMap((item) => item.motion_types || [])),
      );

      search.addEventListener("input", applyFilters);
      composition.addEventListener("change", applyFilters);
      countBucket.addEventListener("change", applyFilters);
      motion.addEventListener("change", applyFilters);
      pageSize.addEventListener("change", () => {{
        state.pageSize = Number(pageSize.value);
        state.page = 1;
        render();
      }});
      prev.addEventListener("click", () => {{
        state.page -= 1;
        render();
      }});
      next.addEventListener("click", () => {{
        state.page += 1;
        render();
      }});
      render();
    }}

    try {{
      init();
    }} catch (err) {{
      console.error(err);
      countPill.textContent = "加载失败";
      pagePill.textContent = "";
      empty.hidden = false;
      empty.textContent = `页面初始化失败: ${{err}}`;
    }}
  </script>
</body>
</html>
"""
    out_path = dataset_root / "physics_browser.html"
    out_path.write_text(page, encoding="utf-8")
    return out_path


class PhysicsBrowserHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory: str, index_name: str, **kwargs):
        self.index_name = str(index_name)
        super().__init__(*args, directory=directory, **kwargs)

    def do_GET(self) -> None:
        request_path = urlsplit(self.path).path
        if request_path in {"", "/"}:
            self.send_response(302)
            self.send_header("Location", f"/{self.index_name}")
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.end_headers()
            return
        super().do_GET()

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    script_path = Path(__file__).resolve().parent / "inspect_physics_sample.py"
    scene_script_path = Path(__file__).resolve().parent / "export_rigid_init_scene_html.py"

    sample_dirs = sorted(
        {
            path.parent
            for meta_name in META_FILENAMES
            for path in dataset_root.rglob(meta_name)
            if is_valid_sample_dir(path.parent)
        }
    )
    if not sample_dirs:
        raise FileNotFoundError(f"No meta.json/metadata.json files found under {dataset_root}")

    total = len(sample_dirs)
    cards: List[SampleCard] = []
    for idx, sample_dir in enumerate(sample_dirs, start=1):
        should_generate = not args.serve_only
        if args.skip_existing and has_visualization_bundle(sample_dir):
            should_generate = False

        if should_generate:
            print(f"[{idx}/{total}] inspect {sample_dir}")
            subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--sample_dir",
                    str(sample_dir),
                    "--num_preview_frames",
                    str(args.num_preview_frames),
                ],
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(scene_script_path),
                    "--sample_dir",
                    str(sample_dir),
                ],
                check=True,
            )
        else:
            print(f"[{idx}/{total}] reuse {sample_dir}")

        meta_path = find_meta_path(sample_dir)
        if meta_path is None:
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        outputs = meta.get("outputs", {})
        rel_dir = sample_dir.resolve().relative_to(dataset_root).as_posix()
        cards.append(
            SampleCard(
                composition=str(meta.get("scene_composition", "")),
                count_bucket=str(meta.get("object_count_bucket", "")),
                scene_id=str(meta.get("scene_id", sample_dir.name)),
                rel_dir=rel_dir,
                rgb_video=str((sample_dir / outputs.get("rgb_video", "videos/rgb.mp4")).resolve().relative_to(dataset_root).as_posix()),
                summary_frames=str((sample_dir / "visualizations" / "summary_frames.png").resolve().relative_to(dataset_root).as_posix()),
                summary_state=str((sample_dir / "visualizations" / "summary_state.png").resolve().relative_to(dataset_root).as_posix()),
                contact_timeline=str((sample_dir / "visualizations" / "contact_timeline.png").resolve().relative_to(dataset_root).as_posix()),
                scene_init_html=str((sample_dir / "visualizations" / "scene_init_interactive.html").resolve().relative_to(dataset_root).as_posix()),
                scene_init_glb=str((sample_dir / "visualizations" / "scene_init.glb").resolve().relative_to(dataset_root).as_posix()),
                motion_types=[str(obj.get("motion_type", "")) for obj in meta.get("objects", [])],
                motion_groups=[str(obj.get("motion_group", "")) for obj in meta.get("objects", [])],
                roles=[str(obj.get("role", "")) for obj in meta.get("objects", [])],
            )
        )

    index_path = build_index(cards, dataset_root)
    print(f"[DONE] generated visualizations for {total} samples under {dataset_root}")
    print(f"[DONE] index page: {index_path}")

    if not args.serve:
        return

    handler = partial(
        PhysicsBrowserHandler,
        directory=str(dataset_root),
        index_name=index_path.name,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"[INFO] browse: http://127.0.0.1:{args.port}/{index_path.name}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] stopped server")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
