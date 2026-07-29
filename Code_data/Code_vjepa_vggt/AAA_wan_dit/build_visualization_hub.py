#!/usr/bin/env python3
"""Build a curated entry point for the Wan DiT visualization pages."""

from __future__ import annotations

import html
import re
from datetime import datetime, timezone
from pathlib import Path


GALLERY_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/gallery"
)
OUTPUTS = (
    (GALLERY_ROOT / "index.html", ""),
    (GALLERY_ROOT / "visualizations" / "index.html", "../"),
)
ENTRIES = (
    (
        "训练诊断",
        "xssc-training-xt-v-x0/index.html",
        "xSSC训练单步 xt → v → x0",
        "真实训练样本加噪、DiT速度预测、x0反推与完整49帧VAE解码对照。",
    ),
    (
        "Head 分类与注意力",
        "fulltoken-head-classification.html",
        "全 token 时间矩阵与运动轨迹 Head 分类 Pilot",
        "S/T/P/C/G 分类、生成视频与代表 Head 的完整 Q@K 矩阵。",
    ),
    (
        "Head 分类与注意力",
        "head_roles_50seeds/index.html",
        "多 seed Head Softmax Q@K 对比",
        "跨 seed、跨模型的 Head 角色稳定性与注意力证据。",
    ),
    (
        "Head 分类与注意力",
        "s_score_extremes_seed851/index.html",
        "Common S score 极值 Q@K",
        "公共 S 类 Head 的 score 前后极值与注意力矩阵。",
    ),
    (
        "Head 分类与注意力",
        "head-role-depth-distribution/index.html",
        "Head 类别与子类别 Block 深度分布",
        "三模型公共稳定 S/T/P/C/G 及互斥特征子类别的逐 Block 分布。",
    ),
    (
        "消融视频",
        "head-role-dose-control-pilot/cases/index.html",
        "S/T/C 等数量匹配与 S-depth 分层 Pilot",
        "20个case独立页面；三模型Baseline、S/T/C匹配消融和Early/Middle/Late All-S分层消融。",
    ),
    (
        "消融视频",
        "test5-st-phased-seed851/cases/index.html",
        "test_5 · Seed 851 · S/T/ST 分阶段消融",
        "20个case下拉切换；视频、Head block分布及相对baseline的17项指标。",
    ),
    (
        "消融视频",
        "multiseed/index.html",
        "Common22 公共 Head 多 seed 消融",
        "公共稳定Head在多个seed与模型下的消融视频。",
    ),
    (
        "消融视频",
        "multiseed/seed851/index.html",
        "Seed 851 分阶段 Head 消融",
        "S/T/P/C/G及S-score子集的分阶段视频比较。",
    ),
    (
        "消融视频",
        "multiseed/stc-phased/index.html",
        "S/T/ST 联合与单独分阶段消融",
        "按模型、Head类别和去噪阶段组织的比较页面。",
    ),
    (
        "指标分析",
        "head-role-dose-control-pilot/metrics/s-t-head-count-control/index.html",
        "S/T 等 Head 数量控制分析",
        "Exact k=5 与 depth-matched k=8 的曲线、热力图、覆盖率和代表视频。",
    ),
    (
        "指标分析",
        "head-role-dose-control-pilot/metrics/index.html",
        "S/T/C 等数量与深度匹配动态指标",
        "按模型独立整合17项曲线及相对Baseline变化总表。",
    ),
    (
        "指标分析",
        "multiseed/benchmark-metrics/index.html",
        "503-case 分阶段消融指标",
        "第一批大规模case的全部指标曲线。",
    ),
    (
        "指标分析",
        "multiseed/seed851/benchmark-metrics/index.html",
        "Seed 851 · test_5 完整指标",
        "45种消融方法、17项指标和同case baseline配对变化表。",
    ),
    (
        "指标分析",
        "multiseed/seed851/benchmark-metrics/focused-impact-curves/index.html",
        "高 Impact 方案指标变化",
        "高Impact配置的配对变化、置信区间与完整表格。",
    ),
    (
        "指标分析",
        "multiseed/seed851/benchmark-metrics/metric-extreme-pairs/index.html",
        "单指标极端消融视频",
        "每项指标、每个模型下同source最好/最差消融比较。",
    ),
    (
        "指标分析",
        "multiseed/seed851/benchmark-metrics/metric-extreme-pairs/physics-iq-pmf.html",
        "Physics-IQ / PMF 单指标极端比较",
        "聚焦Physics-IQ和PMF四项分数的极端视频。",
    ),
    (
        "指标分析",
        "multiseed/seed851/benchmark-metrics/metric-extreme-pairs/physics-iq-pmf-disagreement/index.html",
        "Physics-IQ 与 PMF 评价歧义",
        "同模型同source下，两项指标给出反向排序的视频对。",
    ),
    (
        "运动与 Impact",
        "multiseed/motion-analysis/index.html",
        "运动影响与物理合理性",
        "光流、轨迹、GT合理性及17项Benchmark改善热力图。",
    ),
    (
        "运动与 Impact",
        "multiseed/impact-examples/index.html",
        "Impact 大小视频对照",
        "同baseline下Impact较大与较小的定性视频案例。",
    ),
    (
        "运动与 Impact",
        "multiseed/impact-stage-analysis/index.html",
        "S/T 分阶段运动轨迹 Impact",
        "按模型、Head类别和去噪阶段比较运动轨迹影响。",
    ),
)


def normalized_title(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")[:4096]
    match = re.search(r"<title>(.*?)</title>", text, flags=re.I | re.S)
    return html.unescape(re.sub(r"<[^>]+>", "", match.group(1))).strip() if match else ""


def validate_entries() -> None:
    missing = []
    for _, relative, expected_title, _ in ENTRIES:
        path = GALLERY_ROOT / relative
        if not path.is_file():
            missing.append(str(path))
            continue
        if not normalized_title(path):
            raise RuntimeError(f"Visualization has no HTML title: {path}")
    if missing:
        raise FileNotFoundError("Missing visualization pages:\n" + "\n".join(missing))


def build_html(link_prefix: str) -> str:
    categories = []
    for category in dict.fromkeys(entry[0] for entry in ENTRIES):
        rows = []
        for _, relative, title, description in (
            entry for entry in ENTRIES if entry[0] == category
        ):
            href = link_prefix + relative
            rows.append(
                "<a class='entry' "
                f"href='{html.escape(href)}' "
                f"data-search='{html.escape((title + ' ' + description + ' ' + relative).lower())}'>"
                f"<span class='entry-title'>{html.escape(title)}</span>"
                f"<span class='entry-description'>{html.escape(description)}</span>"
                f"<code>{html.escape('/' + relative)}</code></a>"
            )
        categories.append(
            f"<section><h2>{html.escape(category)}</h2>"
            f"<div class='entries'>{''.join(rows)}</div></section>"
        )
    updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Wan DiT 可视化总入口</title>
<style>
:root{{--bg:#f4f5f2;--ink:#202423;--muted:#66706b;--line:#cbd1cd;--accent:#176f62}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}}
header,main{{max-width:1280px;margin:auto;padding:18px 24px}}header{{border-bottom:1px solid var(--line)}}
h1,h2,p{{margin:0}}h1{{font-size:25px}}h2{{font-size:17px;margin:24px 0 7px}}.sub{{color:var(--muted);margin-top:4px}}
.search{{margin-top:14px;width:min(560px,100%);padding:9px 11px;border:1px solid #aeb7b1;background:#fff;font:inherit}}
.entries{{border-top:1px solid var(--line)}}.entry{{display:grid;grid-template-columns:minmax(230px,.8fr) minmax(320px,1.25fr) minmax(260px,.85fr);gap:18px;align-items:center;padding:11px 8px;border-bottom:1px solid var(--line);color:inherit;text-decoration:none}}
.entry:hover{{background:#fff}}.entry-title{{font-weight:750;color:var(--accent)}}.entry-description{{color:var(--muted)}}code{{font-size:11px;color:#59625e;overflow-wrap:anywhere}}
.empty{{display:none;color:var(--muted);margin-top:18px}}@media(max-width:780px){{.entry{{grid-template-columns:1fr;gap:3px}}}}
</style></head><body><header><h1>Wan DiT 可视化总入口</h1>
<p class="sub">{len(ENTRIES)}个入口 · 更新 {updated}</p>
<input class="search" id="search" type="search" placeholder="搜索模型、实验、指标或页面路径"></header>
<main id="content">{''.join(categories)}<p class="empty" id="empty">没有匹配的页面。</p></main>
<script>
const input=document.getElementById("search"),entries=[...document.querySelectorAll(".entry")],sections=[...document.querySelectorAll("section")],empty=document.getElementById("empty");
function filter(){{const query=input.value.trim().toLowerCase();entries.forEach(entry=>entry.hidden=query&&!entry.dataset.search.includes(query));sections.forEach(section=>section.hidden=![...section.querySelectorAll(".entry")].some(entry=>!entry.hidden));empty.style.display=entries.some(entry=>!entry.hidden)?"none":"block";}}
input.addEventListener("input",filter);
</script></body></html>"""


def main() -> None:
    validate_entries()
    for output, link_prefix in OUTPUTS:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(build_html(link_prefix), encoding="utf-8")
        print(f"[visualization-hub] entries={len(ENTRIES)} output={output}")


if __name__ == "__main__":
    main()
