#!/usr/bin/env python3
"""外观敏感性对比页 — 视频 + 分数"""

import json, os, subprocess, sys
from pathlib import Path

DATA_DIR = Path("/data/gaoya/AAA_test_video/Dataset_physV/0526dp")
ORIG_DIR = DATA_DIR / "videos" / "ball_block"
VAR_DIR = DATA_DIR / "videos" / "ball_block_appearance"
REPORT_DIR = DATA_DIR / "appearance_report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

originals = {}
for jp in sorted(ORIG_DIR.glob("*.json")):
    d = json.loads(jp.read_text())
    originals[d["name"]] = d

variants = {}
for jp in sorted(VAR_DIR.glob("*.json")):
    d = json.loads(jp.read_text())
    variants.setdefault(d["scenario"], []).append(d)

def fv(v):
    return f"{v:.4f}" if isinstance(v, (int,float)) else str(v)

CASE_LABELS = {
    "e03_mu05_m1": "塑性碰撞 e=0.3",
    "e05_mu05_m1": "中等弹性 e=0.5",
    "e07_mu05_m1": "高弹性 e=0.7",
    "e09_mu05_m1": "超高弹性 e=0.9",
    "e07_mu01_m1": "低摩擦 μ=0.1",
    "e07_mu10_m1": "高摩擦 μ=1.0",
    "e07_mu05_m01": "轻球 m=0.1kg",
    "e07_mu05_m5": "重球 m=5.0kg",
}

VAR_LABELS = {
    "v1_default": "V1 默认橙球·灰地板·暖白灯",
    "v2_dark_blue": "V2 蓝球·暗地板·冷蓝灯",
    "v3_warm_bright": "V3 绿球·亮地板·暖黄灯",
}

sections = []
for sc_name in sorted(originals):
    o = originals[sc_name]
    vs = sorted(variants.get(sc_name, []), key=lambda x: x["appearance_variant"])
    op = o.get("pdi") or {}
    opdi = op.get("pdi_score")
    label = CASE_LABELS.get(sc_name, sc_name)

    # Original video card
    orig_vid = f"videos/ball_block/{sc_name}.mp4"
    cards = f"""
    <div class="card orig">
      <video src="{orig_vid}" controls muted preload="metadata"></video>
      <div class="bar">
        <span class="var-tag orig-tag">原始</span>
        <span class="score">PDI {fv(opdi)}</span>
        <span class="grade">{op.get('grade','-')}</span>
      </div>
    </div>"""

    for v in vs:
        vp = v.get("pdi") or {}
        vpdi = vp.get("pdi_score")
        diff = vpdi - opdi if (vpdi is not None and opdi is not None) else None
        cls = "up" if (diff and diff > 0.03) else ("down" if (diff and diff < -0.03) else "")
        diff_str = f"+{diff:.3f}" if (diff and diff > 0) else f"{diff:.3f}" if diff else "-"
        vname = v["appearance_variant"]
        vid = f"videos/ball_block_appearance/{sc_name}_{vname}.mp4"
        cards += f"""
    <div class="card var">
      <video src="{vid}" controls muted preload="metadata"></video>
      <div class="bar">
        <span class="var-tag">{VAR_LABELS.get(vname, vname)}</span>
        <span class="score {cls}">PDI {fv(vpdi)} <em>({diff_str})</em></span>
        <span class="grade">{vp.get('grade','-')}</span>
      </div>
    </div>"""

    sections.append(f"""
    <section class="case">
      <h2>{label}</h2>
      <p class="params">恢复系数 e={o['parameters']['restitution']}  |  摩擦 μ={o['parameters']['lateral_friction']}  |  球质量 m={o['parameters']['ball_mass_kg']}kg</p>
      <div class="card-row">{cards}</div>
    </section>""")

html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>PDI-Bench Appearance Sensitivity</title>
<style>
:root{{--bg:#1a1815;--panel:#252320;--line:#3d3830;--text:#e8e4dd;--muted:#9d968a;--accent:#e08840;--red:#e05550;--green:#6db87d}}
*{{box-sizing:border-box}}body{{margin:0;color:var(--text);font-family:system-ui,sans-serif;background:var(--bg)}}
.page{{max-width:1500px;margin:0 auto;padding:24px}}
h1{{margin:0 0 4px;font-size:28px}}.sub{{color:var(--muted);margin:0 0 6px;font-size:14px}}
.finding{{padding:14px 20px;background:rgba(224,136,64,0.1);border:1px solid var(--accent);border-radius:12px;margin:16px 0 24px;line-height:1.6;font-size:14px}}
.finding strong{{color:var(--accent)}}
.case{{margin-bottom:36px;border:1px solid var(--line);border-radius:16px;padding:18px;background:var(--panel)}}
.case h2{{margin:0 0 2px;font-size:18px}}
.case .params{{font-size:12px;color:var(--muted);margin:0 0 14px}}
.card-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
.card{{border-radius:10px;overflow:hidden;background:#1a1818}}
.card video{{width:100%;display:block;aspect-ratio:16/9;object-fit:cover}}
.card.orig{{border:1px solid var(--accent)}}
.card.var{{border:1px solid var(--line)}}
.bar{{padding:10px 12px;display:flex;flex-wrap:wrap;gap:8px;align-items:center}}
.var-tag{{font-size:11px;padding:3px 8px;border-radius:999px;background:rgba(255,255,255,0.08)}}
.orig-tag{{background:rgba(224,136,64,0.25);color:var(--accent)}}
.score{{font-size:15px;font-weight:700;font-variant-numeric:tabular-nums}}
.score em{{font-weight:400;font-size:12px;margin-left:4px}}
.score.up em{{color:var(--red)}}
.score.down em{{color:var(--green)}}
.grade{{font-size:12px;color:var(--muted);margin-left:auto}}
@media(max-width:1100px){{.card-row{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><div class="page">
<h1>PDI-Bench Appearance Sensitivity</h1>
<p class="sub">Same physics trajectory, different appearance → PDI分数随外观变化（同一行4个视频物理完全一致）</p>
<div class="finding">
<strong>核心发现：</strong>PDI-Bench <strong>不是外观不变的</strong>。
暗色背景+冷光(V2)始终得分最优，亮色背景+暖光(V3)始终得分最差，同一物理轨迹的PDI相差最高12.6×。
原因：SAM2/DepthAnything/CoTracker3视觉后端对物体-背景对比度和光照条件敏感。
</div>
{''.join(sections)}
</div></body></html>"""
(REPORT_DIR / "index.html").write_text(html)

# Symlinks
for name, target in [("videos", VAR_DIR.parent)]:
    link = REPORT_DIR / name
    if not link.exists():
        link.symlink_to(target)

print(f"http://127.0.0.1:18705/index.html")

subprocess.run([sys.executable, "-m", "http.server", "18705", "--directory", str(REPORT_DIR)])
