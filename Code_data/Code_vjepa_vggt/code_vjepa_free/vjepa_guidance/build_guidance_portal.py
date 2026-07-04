#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path


CSS = """
:root {
  --bg: #f4efe8;
  --paper: #fffaf3;
  --card: rgba(255, 252, 246, 0.94);
  --ink: #1a1512;
  --muted: #6d655c;
  --line: #ddd2c4;
  --accent: #0f6d64;
  --accent-2: #a4452b;
  --good: #1f7a61;
  --bad: #a53e31;
  --shadow: 0 18px 40px rgba(42, 28, 17, 0.10);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  color: var(--ink);
  font-family: Georgia, "Times New Roman", serif;
  background:
    radial-gradient(circle at top right, rgba(15,109,100,0.12), transparent 24%),
    radial-gradient(circle at left top, rgba(164,69,43,0.08), transparent 18%),
    linear-gradient(180deg, #f8f4ec 0%, var(--bg) 100%);
}
.wrap {
  width: min(1560px, calc(100vw - 28px));
  margin: 0 auto;
  padding: 28px 0 64px;
}
h1 {
  margin: 0 0 10px;
  font-size: clamp(34px, 4.4vw, 64px);
  line-height: 0.95;
  letter-spacing: -0.045em;
}
.sub {
  color: var(--muted);
  font-size: 18px;
  line-height: 1.5;
  margin-bottom: 24px;
  max-width: 980px;
}
.meta, .baseline-meta, .phase-meta, .stats, .analysis {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}
.pill {
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 8px 14px;
  background: rgba(255,255,255,0.68);
  font-size: 13px;
}
.hero {
  background: linear-gradient(180deg, rgba(255,255,255,0.74), rgba(255,250,243,0.94));
  border: 1px solid var(--line);
  border-radius: 28px;
  box-shadow: var(--shadow);
  padding: 22px 22px 20px;
  margin-bottom: 24px;
}
.phase {
  background: linear-gradient(180deg, rgba(255,255,255,0.74), rgba(255,250,243,0.94));
  border: 1px solid var(--line);
  border-radius: 28px;
  box-shadow: var(--shadow);
  padding: 20px;
  margin-bottom: 24px;
}
.phase-head {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: end;
  margin-bottom: 10px;
}
.phase-title {
  margin: 0;
  font-size: clamp(26px, 3vw, 40px);
  line-height: 1;
}
.phase-note {
  color: var(--muted);
  font-size: 15px;
  max-width: 900px;
  line-height: 1.45;
  margin: 8px 0 0;
}
.phase-grid {
  display: grid;
  grid-template-columns: minmax(300px, 420px) 1fr;
  gap: 18px;
  margin-top: 18px;
}
.panel {
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 22px;
  padding: 16px;
}
.panel h3 {
  margin: 0 0 12px;
  font-size: 19px;
}
.analysis .pill {
  background: rgba(15,109,100,0.08);
}
.rank-list {
  display: grid;
  gap: 10px;
}
.rank-card {
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 12px;
  background: rgba(255,255,255,0.7);
}
.rank-card.top {
  border-color: rgba(31,122,97,0.35);
  background: rgba(31,122,97,0.08);
}
.rank-top {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  align-items: baseline;
}
.rank-name {
  margin: 0;
  font-size: 16px;
  font-family: "SFMono-Regular", Consolas, monospace;
}
.rank-score {
  font-size: 24px;
  font-weight: 700;
  letter-spacing: -0.03em;
}
.rank-meta {
  margin-top: 8px;
  color: var(--muted);
  font-size: 13px;
  line-height: 1.45;
}
.curve {
  width: 100%;
  display: block;
  border-radius: 16px;
  border: 1px solid var(--line);
  background: #fff;
}
.video-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: 16px;
}
.video-card {
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 22px;
  overflow: hidden;
}
.video-card.best {
  border-color: rgba(31,122,97,0.35);
  box-shadow: inset 0 0 0 1px rgba(31,122,97,0.08);
}
.video-card.baseline {
  border-color: rgba(165,62,49,0.22);
}
.video-card video {
  display: block;
  width: 100%;
  aspect-ratio: 16 / 9;
  background: #111;
}
.video-body {
  padding: 14px;
}
.video-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 12px;
  margin-bottom: 8px;
}
.label {
  margin: 0;
  font-size: 17px;
  line-height: 1.15;
  font-family: "SFMono-Regular", Consolas, monospace;
  word-break: break-word;
}
.delta.good { color: var(--good); }
.delta.bad { color: var(--bad); }
.delta {
  font-size: 24px;
  font-weight: 700;
  letter-spacing: -0.03em;
}
.mini-stats {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.stat {
  border: 1px solid var(--line);
  border-radius: 999px;
  padding: 6px 10px;
  background: rgba(255,255,255,0.7);
  font-size: 12px;
}
.path {
  margin-top: 10px;
  color: var(--muted);
  font-size: 12px;
  word-break: break-all;
  font-family: "SFMono-Regular", Consolas, monospace;
}
.footer-note {
  color: var(--muted);
  font-size: 14px;
  line-height: 1.5;
  margin-top: 18px;
}
@media (max-width: 980px) {
  .phase-grid {
    grid-template-columns: 1fr;
  }
}
@media (max-width: 720px) {
  .wrap {
    width: min(100vw - 16px, 1560px);
    padding-top: 16px;
  }
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a static HTML portal for V-JEPA guidance sweep results.")
    parser.add_argument(
        "--probe-root",
        type=Path,
        default=Path("/data/gaoya/agent-data/outputs/probe_sweep"),
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("/data/gaoya/agent-data/outputs/probe_sweep/portal/index.html"),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def rel_from_html(output_html: Path, target: Path) -> str:
    return os.path.relpath(target, start=output_html.parent).replace(os.sep, "/")


def fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def fmt_delta(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        if value > 0:
            return f"+{value}"
        return str(value)
    if value > 0:
        return f"+{value:.{digits}f}"
    return f"{value:.{digits}f}"


def phase_note(phase_name: str) -> str:
    notes = {
        "phase5": "强度阶梯。核心问题是 guidance 太弱还是方向不对；这里主要看 step size 增大后视频是否真正偏离 baseline。",
        "phase6": "拐点附近微调。固定强 guidance 区间，比较 timing、inner_k 和 backtracking 是否能在不削弱效果的前提下更稳。",
        "phase7": "target-shape sweep。固定 ladder_s20 调度，只改 anchored future horizon，测试 energy window shape 是否影响最终物理分数。",
    }
    return notes.get(phase_name, "")


def build_phase_analysis(phase_name: str, rows: list[dict], summary: dict) -> list[str]:
    scored = [row for row in rows if row["label"] != "baseline"]
    if not scored:
        return ["没有可分析的 guided 样本。"]
    best_surprise = min(scored, key=lambda row: row.get("surprise", float("inf")))
    best_delta = min(scored, key=lambda row: row.get("delta_surprise_vs_base", float("inf")))
    strongest_write = max(
        [row for row in scored if row.get("mean_delta_post") is not None],
        key=lambda row: row["mean_delta_post"],
        default=None,
    )
    bullets: list[str] = []
    if phase_name == "phase5":
        bullets.append(
            f"按最终 wmreward 排序，`{best_surprise['label']}` 最好，Δsurprise={fmt_delta(best_surprise.get('delta_surprise_vs_base'))}。"
        )
        bullets.append("弱步长 `s01/s02/s05` 基本没有收益，说明问题主要不是方向错，而是 guidance 强度不够。")
        if strongest_write is not None and strongest_write["label"] != best_surprise["label"]:
            bullets.append(
                f"写入最强的是 `{strongest_write['label']}` 的 `mean_delta_post={fmt(strongest_write['mean_delta_post'], 6)}`，但最佳视频仍取决于最终 surprise。"
            )
        else:
            bullets.append("`ladder_s20` 同时把 surprise、similarity、videophy2 和 physics_iq 都推到当前单 case 最优区间。")
    elif phase_name == "phase6":
        bullets.append(
            f"按最终 wmreward 排序，`{best_surprise['label']}` 最好，Δsurprise={fmt_delta(best_surprise.get('delta_surprise_vs_base'))}。"
        )
        bullets.append("`knee_mid_s15_bt` 明显退回 baseline 邻域，说明 backtracking 在当前能量地形上过于保守。")
        bullets.append("更早时机和 `inner_k=2` 都能进入有效区，但还没有超过 phase5 的 `ladder_s20`。")
    elif phase_name == "phase7":
        bullets.append(
            f"按最终 wmreward 排序，`{best_surprise['label']}` 最好，Δsurprise={fmt_delta(best_surprise.get('delta_surprise_vs_base'))}。"
        )
        if strongest_write is not None:
            bullets.append(
                f"`{strongest_write['label']}` 的 `mean_delta_post={fmt(strongest_write['mean_delta_post'], 6)}` 最大，但最终 surprise 不如 `{best_surprise['label']}`。"
            )
        bullets.append("这说明 window shape 已经在影响优化方向本身，而不只是影响 guidance 写入强度。")
    else:
        bullets.append(
            f"最佳 guided 样本是 `{best_surprise['label']}`，Δsurprise={fmt_delta(best_delta.get('delta_surprise_vs_base'))}。"
        )
    return bullets


def build_phase_section(phase_name: str, phase_dir: Path, output_html: Path) -> str:
    score_json = phase_dir / f"{phase_name}_multimetric_scores.json"
    summary_json = phase_dir / f"{phase_name}_summary.json"
    scores = load_json(score_json)
    summary = load_json(summary_json)
    rows = sorted(scores["rows"], key=lambda row: (row["label"] == "baseline", row.get("surprise", float("inf"))))
    baseline = next(row for row in scores["rows"] if row["label"] == scores["baseline_label"])
    guided_rows = [row for row in rows if row["label"] != "baseline"]
    best_by_surprise = min(guided_rows, key=lambda row: row.get("surprise", float("inf")))
    best_summary = summary.get("best", {})
    analysis_items = build_phase_analysis(phase_name, rows, summary)

    ranked_cards = []
    for idx, row in enumerate(sorted(guided_rows, key=lambda row: row.get("surprise", float("inf"))), start=1):
        is_top = row["label"] == best_by_surprise["label"]
        ranked_cards.append(
            f"""
            <div class="rank-card{' top' if is_top else ''}">
              <div class="rank-top">
                <h4 class="rank-name">#{idx} {html.escape(row['label'])}</h4>
                <div class="rank-score">{fmt(row.get('surprise'), 4)}</div>
              </div>
              <div class="rank-meta">
                Δsurprise {fmt_delta(row.get('delta_surprise_vs_base'))}
                · similarity {fmt(row.get('similarity'), 4)}
                · physics_iq {fmt(row.get('physics_iq_score'), 2)}
                · videophy2 {fmt(row.get('videophy2_score'))}
                · mean_delta_post {fmt(row.get('mean_delta_post'), 6)}
              </div>
            </div>
            """
        )

    curve_path = phase_dir / f"{phase_name.capitalize()}_delta_curves.png"
    curve_html = ""
    if curve_path.is_file():
        curve_html = (
            f"<img class='curve' src='{html.escape(rel_from_html(output_html, curve_path))}' "
            f"alt='{html.escape(phase_name)} delta curves'>"
        )

    video_cards = []
    for row in rows:
        label = row["label"]
        classes = ["video-card"]
        if label == "baseline":
            classes.append("baseline")
        if label == best_by_surprise["label"]:
            classes.append("best")
        delta = row.get("delta_surprise_vs_base")
        delta_cls = "good" if isinstance(delta, (int, float)) and delta < 0 else "bad"
        video_path = Path(row["video"])
        video_cards.append(
            f"""
            <article class="{' '.join(classes)}">
              <video controls preload="metadata" src="{html.escape(rel_from_html(output_html, video_path))}"></video>
              <div class="video-body">
                <div class="video-head">
                  <h4 class="label">{html.escape(label)}</h4>
                  <div class="delta {delta_cls}">{fmt_delta(delta)}</div>
                </div>
                <div class="mini-stats">
                  <div class="stat">surprise {fmt(row.get('surprise'), 4)}</div>
                  <div class="stat">similarity {fmt(row.get('similarity'), 4)}</div>
                  <div class="stat">physics_iq {fmt(row.get('physics_iq_score'), 2)}</div>
                  <div class="stat">videophy2 {fmt(row.get('videophy2_score'))}</div>
                  <div class="stat">cosmos {fmt(row.get('cosmos_reason1_score'))}</div>
                  <div class="stat">mean_delta_post {fmt(row.get('mean_delta_post'), 6)}</div>
                </div>
                <div class="path">{html.escape(str(video_path))}</div>
              </div>
            </article>
            """
        )

    summary_best_label = best_summary.get("label")
    summary_best_note = (
        f"summary-best by recorded ranking: {summary_best_label}"
        if summary_best_label
        else "summary-best unavailable"
    )
    energy_note = ""
    if phase_name == "phase7":
        energy_note = "注意：phase7 的 summary 按 `abs_mean_delta_desc` 排序，所以 summary-best 代表写入最强，不一定是最终 wmreward 最优。"

    return f"""
    <section class="phase">
      <div class="phase-head">
        <div>
          <h2 class="phase-title">{html.escape(phase_name.upper())}</h2>
          <p class="phase-note">{html.escape(phase_note(phase_name))}</p>
        </div>
      </div>
      <div class="phase-meta">
        <div class="pill">baseline surprise {fmt(scores.get('baseline_surprise'), 4)}</div>
        <div class="pill">baseline similarity {fmt(scores.get('baseline_similarity'), 4)}</div>
        <div class="pill">baseline physics_iq {fmt(scores.get('baseline_physics_iq_score'), 2)}</div>
        <div class="pill">baseline videophy2 {fmt(scores.get('baseline_videophy2_score'))}</div>
        <div class="pill">best by surprise {html.escape(best_by_surprise['label'])}</div>
        <div class="pill">{html.escape(summary_best_note)}</div>
      </div>
      <div class="phase-grid">
        <div class="panel">
          <h3>Phase Reading</h3>
          <div class="analysis">
            {''.join(f"<div class='pill'>{html.escape(item)}</div>" for item in analysis_items)}
          </div>
          <div class="footer-note">{html.escape(energy_note)}</div>
          {"<div style='height:14px'></div>" + curve_html if curve_html else ""}
        </div>
        <div class="panel">
          <h3>Ranking By Final Surprise</h3>
          <div class="rank-list">
            {''.join(ranked_cards)}
          </div>
        </div>
      </div>
      <div style="height:18px"></div>
      <div class="video-grid">
        {''.join(video_cards)}
      </div>
    </section>
    """


def build_html(probe_root: Path, output_html: Path) -> None:
    phases = ["phase5", "phase6", "phase7"]
    sections = [build_phase_section(phase_name, probe_root / phase_name, output_html) for phase_name in phases]
    body = f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>V-JEPA Guidance Portal</title>
  <style>{CSS}</style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>V-JEPA Guidance Portal</h1>
      <div class="sub">
        汇总展示当前单 case sweep 的生成视频、物理分数和结论性分析。这里把 `phase5` 的强度阶梯、
        `phase6` 的 timing / inner-k 微调，以及 `phase7` 的 future-horizon target-shape sweep 放在同一页，便于直接看视频和分数是否一致。
      </div>
      <div class="meta">
        <div class="pill">probe root: {html.escape(str(probe_root))}</div>
        <div class="pill">focus metric: wmreward surprise ↓</div>
        <div class="pill">cross-check: physics_iq / videophy2_pc / cosmos_reason1</div>
        <div class="pill">current phase7 metric winner: target_w24</div>
      </div>
    </section>
    {''.join(sections)}
  </div>
</body>
</html>
"""
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(body, encoding="utf-8")


def main() -> None:
    args = parse_args()
    probe_root = args.probe_root.expanduser().resolve()
    output_html = args.output_html.expanduser().resolve()
    build_html(probe_root, output_html)
    print(output_html)


if __name__ == "__main__":
    main()
