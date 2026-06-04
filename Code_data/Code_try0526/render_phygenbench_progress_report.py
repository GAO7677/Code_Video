#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path


OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Output_try0526/phygenbench/output")
REPORT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0526dp/reports/abc_report")
REPORT_DIR = REPORT_ROOT / "phygenbench_progress"
REPORT_PATH = REPORT_DIR / "index.html"
MANIFEST_PATH = REPORT_DIR / "progress.json"
OUTPUT_LINK = REPORT_ROOT / "phygenbench_output"

BENCHMARK_NAME = "phygenbench"
FIRST_FRAME_METHOD = "FLUX_1_Kontext"
VIDEO_METHODS = ["wan22-5B-TI2V", "VACE_1p3B_TI2V", "VACE_1p3B_ctx08"]
ALL_METHODS = [FIRST_FRAME_METHOD] + VIDEO_METHODS
METHOD_LABELS = {
    "FLUX_1_Kontext": "FLUX.1-Kontext First Frame",
    "wan22-5B-TI2V": "Wan2.2-5B TI2V",
    "VACE_1p3B_TI2V": "VACE 1.3B TI2V",
    "VACE_1p3B_ctx08": "VACE 1.3B ctx=8",
}
MAX_FIRSTFRAME_CARDS = 24


def ensure_symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        try:
            if dst.resolve() == src.resolve():
                return
        except Exception:
            pass
        if dst.is_dir() and not dst.is_symlink():
            raise RuntimeError(f"Refusing to replace directory: {dst}")
        dst.unlink()
    os.symlink(src, dst)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def href_from_report(target: str | Path) -> str:
    return html.escape(
        os.path.relpath(Path(target).resolve(), REPORT_DIR.resolve()).replace("\\", "/")
    )


def href_output_path(target: str | Path) -> str:
    target_path = Path(target).resolve()
    rel = target_path.relative_to(OUTPUT_ROOT.resolve())
    return html.escape((Path("..") / OUTPUT_LINK.name / rel).as_posix())


def discover_firstframe_cases() -> list[dict]:
    root = OUTPUT_ROOT / FIRST_FRAME_METHOD / BENCHMARK_NAME
    cases: list[dict] = []
    for json_path in sorted(root.glob("*.json")):
        if not json_path.stem.isdigit():
            continue
        payload = load_json(json_path)
        cases.append(payload)
    return cases


def discover_full_cases(firstframe_cases: list[dict]) -> list[dict]:
    full_cases: list[dict] = []
    for payload in firstframe_cases:
        sample_id = str(payload["sample_id"])
        methods: dict[str, dict] = {FIRST_FRAME_METHOD: payload}
        complete = True
        for method in VIDEO_METHODS:
            method_json = OUTPUT_ROOT / method / BENCHMARK_NAME / f"{sample_id}.json"
            if not method_json.is_file():
                complete = False
                break
            methods[method] = load_json(method_json)
        if complete:
            full_cases.append(
                {
                    "sample_id": sample_id,
                    "prompt_index": payload["prompt_index"],
                    "clip_name": payload["clip_name"],
                    "prompt": payload["prompt"],
                    "caption": payload["caption"],
                    "physical_laws": payload.get("physical_laws", ""),
                    "sub_category": payload.get("sub_category", ""),
                    "main_category": payload.get("main_category", ""),
                    "methods": methods,
                }
            )
    return full_cases


def render_counts(firstframe_cases: list[dict], full_cases: list[dict]) -> str:
    counts = {
        method: len(list((OUTPUT_ROOT / method / BENCHMARK_NAME).glob("*.json")))
        for method in ALL_METHODS
    }
    cards = []
    cards.append(_stat_card("FLUX first frames", str(counts[FIRST_FRAME_METHOD]), "PNG + ctx08 seed video"))
    cards.append(_stat_card("Full 4-method cases", str(len(full_cases)), "all video methods ready"))
    cards.append(_stat_card("Wan videos", str(counts["wan22-5B-TI2V"]), "TI2V outputs"))
    cards.append(_stat_card("VACE TI2V", str(counts["VACE_1p3B_TI2V"]), "first-frame conditioned"))
    cards.append(_stat_card("VACE ctx=8", str(counts["VACE_1p3B_ctx08"]), "repeat-frame context video"))
    cards.append(_stat_card("Total prompts", "160", "PhyGenBench official prompt count"))
    return f"""
    <section class="stats">
      {''.join(cards)}
      <div class="progress-note">
        当前端到端完整样本较少，主要因为两张卡还在批量生成首帧。页面会先展示已经完整可看的样本，
        以及最新一批已经产出的 FLUX 首帧，便于尽早做人工检查。
      </div>
    </section>
    """


def _stat_card(title: str, value: str, note: str) -> str:
    return f"""
      <div class="stat-card">
        <div class="stat-title">{html.escape(title)}</div>
        <div class="stat-value">{html.escape(value)}</div>
        <div class="stat-note">{html.escape(note)}</div>
      </div>
    """


def render_full_case(case: dict) -> str:
    method_cards = []
    for method in ALL_METHODS:
        payload = case["methods"][method]
        if method == FIRST_FRAME_METHOD:
            media = f'<img src="{href_output_path(payload["first_frame"])}" alt="first frame" />'
            links = (
                f'<a href="{href_output_path(payload["first_frame"])}">Open image</a>'
                f'<a href="{href_output_path(payload["context_video"])}">Open ctx08 video</a>'
            )
        else:
            media = f'<video controls preload="metadata" src="{href_output_path(payload["video_path"])}"></video>'
            links = f'<a href="{href_output_path(payload["video_path"])}">Open video</a>'
        method_cards.append(
            f"""
            <article class="method-card">
              <div class="method-head">
                <div class="method-name">{html.escape(METHOD_LABELS[method])}</div>
                <div class="method-meta">{html.escape(str(payload.get("conditioning_mode", "-")))}</div>
              </div>
              {media}
              <div class="method-links">{links}</div>
            </article>
            """
        )
    return f"""
    <section class="case-card">
      <div class="case-copy">
        <div class="eyebrow">Full 4-method case</div>
        <h2>{html.escape(case['sample_id'])} · {html.escape(case['clip_name'])}</h2>
        <p><strong>Prompt</strong>: {html.escape(case['prompt'])}</p>
        <p><strong>Category</strong>: {html.escape(case['main_category'])} / {html.escape(case['sub_category'])} / {html.escape(case['physical_laws'])}</p>
      </div>
      <div class="method-grid">
        {''.join(method_cards)}
      </div>
    </section>
    """


def render_firstframe_card(payload: dict) -> str:
    return f"""
    <article class="firstframe-card">
      <img src="{href_output_path(payload['first_frame'])}" alt="first frame" />
      <div class="firstframe-copy">
        <div class="firstframe-id">{html.escape(str(payload['sample_id']))} · {html.escape(payload['clip_name'])}</div>
        <div class="firstframe-cat">{html.escape(payload.get('main_category', ''))} / {html.escape(payload.get('sub_category', ''))}</div>
        <p>{html.escape(payload['prompt'])}</p>
        <div class="firstframe-links">
          <a href="{href_output_path(payload['first_frame'])}">Open image</a>
          <a href="{href_output_path(payload['context_video'])}">Open ctx08 video</a>
        </div>
      </div>
    </article>
    """


def build_html(firstframe_cases: list[dict], full_cases: list[dict]) -> str:
    latest_firstframes = sorted(
        firstframe_cases,
        key=lambda item: int(item["sample_id"]),
    )[:MAX_FIRSTFRAME_CARDS]
    full_html = "".join(render_full_case(case) for case in full_cases)
    firstframe_html = "".join(render_firstframe_card(item) for item in latest_firstframes)
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta http-equiv="refresh" content="60" />
  <title>PhyGenBench Progress</title>
  <style>
    :root {{
      --bg: #f5efe6;
      --panel: #fffaf3;
      --panel-2: #fffdf9;
      --ink: #1f1a17;
      --muted: #6f6258;
      --line: #dccdc1;
      --accent: #c55c3b;
      --accent-2: #356d8c;
      --shadow: 0 12px 32px rgba(72, 51, 35, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Noto Sans SC", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(197, 92, 59, 0.10), transparent 28rem),
        radial-gradient(circle at top right, rgba(53, 109, 140, 0.08), transparent 30rem),
        var(--bg);
      color: var(--ink);
    }}
    main {{
      max-width: 1480px;
      margin: 0 auto;
      padding: 28px 24px 64px;
    }}
    .topbar {{
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: end;
      margin-bottom: 24px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 34px;
      line-height: 1.1;
    }}
    .sub {{
      color: var(--muted);
      max-width: 920px;
      line-height: 1.6;
    }}
    .top-links a {{
      color: var(--accent-2);
      text-decoration: none;
      font-weight: 600;
      margin-left: 16px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 28px;
    }}
    .stat-card, .progress-note, .case-card, .section-block {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: var(--shadow);
    }}
    .stat-card {{
      padding: 16px 18px;
    }}
    .stat-title {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .stat-value {{
      font-size: 28px;
      font-weight: 700;
      margin-bottom: 6px;
    }}
    .stat-note {{
      font-size: 12px;
      color: var(--muted);
      line-height: 1.4;
    }}
    .progress-note {{
      grid-column: 1 / -1;
      padding: 16px 18px;
      color: var(--muted);
      line-height: 1.6;
    }}
    .section-head {{
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 16px;
      margin: 30px 0 14px;
    }}
    .section-head h2 {{
      margin: 0 0 6px;
      font-size: 24px;
    }}
    .section-head .meta {{
      color: var(--muted);
      font-size: 14px;
    }}
    .case-card {{
      padding: 20px;
      margin-bottom: 20px;
    }}
    .eyebrow {{
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.08em;
      color: var(--accent);
      text-transform: uppercase;
    }}
    .case-copy h2 {{
      margin: 8px 0 10px;
      font-size: 24px;
    }}
    .case-copy p {{
      margin: 8px 0;
      line-height: 1.6;
    }}
    .method-grid {{
      margin-top: 18px;
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
    }}
    .method-card {{
      background: var(--panel-2);
      border: 1px solid var(--line);
      border-radius: 16px;
      overflow: hidden;
    }}
    .method-head {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      padding: 12px 14px 10px;
      border-bottom: 1px solid var(--line);
    }}
    .method-name {{
      font-weight: 700;
      font-size: 14px;
    }}
    .method-meta {{
      color: var(--muted);
      font-size: 12px;
      text-align: right;
    }}
    .method-card video, .method-card img {{
      width: 100%;
      aspect-ratio: 7 / 4;
      display: block;
      object-fit: cover;
      background: #ddd2c6;
    }}
    .method-links, .firstframe-links {{
      display: flex;
      gap: 14px;
      padding: 12px 14px 14px;
      flex-wrap: wrap;
    }}
    a {{
      color: var(--accent-2);
      text-decoration: none;
      font-weight: 600;
    }}
    .firstframe-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 16px;
    }}
    .firstframe-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      overflow: hidden;
      box-shadow: var(--shadow);
    }}
    .firstframe-card img {{
      width: 100%;
      aspect-ratio: 7 / 4;
      object-fit: cover;
      display: block;
      background: #ddd2c6;
    }}
    .firstframe-copy {{
      padding: 14px 16px 16px;
    }}
    .firstframe-id {{
      font-weight: 700;
      margin-bottom: 6px;
    }}
    .firstframe-cat {{
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 8px;
    }}
    .firstframe-copy p {{
      margin: 0;
      line-height: 1.55;
      min-height: 72px;
    }}
    @media (max-width: 1280px) {{
      .stats {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
      .method-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .firstframe-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 760px) {{
      main {{ padding: 18px 14px 40px; }}
      .topbar {{ display: block; }}
      .top-links {{ margin-top: 12px; }}
      .stats {{ grid-template-columns: 1fr 1fr; }}
      .method-grid, .firstframe-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <div class="topbar">
      <div>
        <h1>PhyGenBench Progress</h1>
        <div class="sub">
          当前页展示 PhyGenBench 的阶段性生成结果。上半部分是已经完整生成
          <code>FLUX first frame + Wan + VACE TI2V + VACE ctx08</code> 的样本，
          下半部分是已经产出的 FLUX 首帧样本，便于在全量任务完成前先做人工检查。
        </div>
      </div>
      <div class="top-links">
        <a href="../index.html">Back to ABCD Report</a>
        <a href="../phygenbench_output/">Open Raw Output Dir</a>
      </div>
    </div>
    <div class="sub">Last updated: {html.escape(updated)}</div>
    {render_counts(firstframe_cases, full_cases)}
    <div class="section-head">
      <div>
        <h2>完整四方法样本</h2>
        <div class="meta">当前已完整生成 {len(full_cases)} / 160 个样本。</div>
      </div>
    </div>
    {full_html or '<div class="section-block" style="padding:18px 20px;color:#6f6258;">暂时还没有完整四方法样本。</div>'}
    <div class="section-head">
      <div>
        <h2>已完成 FLUX 首帧样本</h2>
        <div class="meta">当前已完成 {len(firstframe_cases)} / 160 个首帧样本。这里先展示前 {min(len(latest_firstframes), MAX_FIRSTFRAME_CARDS)} 个。</div>
      </div>
    </div>
    <section class="firstframe-grid">
      {firstframe_html}
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ensure_symlink(OUTPUT_ROOT, OUTPUT_LINK)
    firstframe_cases = discover_firstframe_cases()
    full_cases = discover_full_cases(firstframe_cases)
    REPORT_PATH.write_text(build_html(firstframe_cases, full_cases), encoding="utf-8")
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "firstframe_count": len(firstframe_cases),
                "full_case_count": len(full_cases),
                "firstframe_sample_ids": [item["sample_id"] for item in firstframe_cases],
                "full_case_sample_ids": [item["sample_id"] for item in full_cases],
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
