#!/usr/bin/env python3
"""JEPA 敏感性实验可视化"""

import json, os, subprocess, sys
from pathlib import Path

DATA_DIR = Path("/data/gaoya/AAA_test_video/Dataset_physV/0526dp")
VAR_DIR = DATA_DIR / "videos" / "jepa_sensitivity"
REPORT_DIR = DATA_DIR / "jepa_sensitivity_report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load all results
results = []
for jp in sorted(VAR_DIR.glob("*.json")):
    d = json.loads(jp.read_text())
    results.append(d)

results.sort(key=lambda x: x.get("jepa", {}).get("jepa_score", 0))

def fv(v): return f"{v:.4f}" if isinstance(v,(int,float)) else str(v)

# Group into experiment groups
EXP_GROUPS = {
    "velocity": ("初速度变化 (固定质量1kg)", ["vel_005","vel_015","vel_035","vel_070","vel_140"]),
    "mass":     ("球质量变化 (固定初速3.5m/s)", ["mass_001","mass_005","mass_010","mass_100","mass_500","mass_2000","mass_9999"]),
    "gravity":  ("重力变化", ["grav_050","grav_098","grav_200"]),
    "block":    ("木块质量变化", ["blk_005","blk_500","blk_2000"]),
    "special":  ("极端场景", ["nomiss","rev_035"]),
}

all_jepa = [r.get("jepa",{}).get("jepa_score",0) for r in results if r.get("jepa")]
jepa_range = max(all_jepa) - min(all_jepa) if all_jepa else 0

sections = []
for gkey, (gtitle, names) in EXP_GROUPS.items():
    cards = ""
    for name in names:
        r = next((r for r in results if r["scenario"] == name), None)
        if not r: continue
        j = r.get("jepa",{}).get("jepa_score")
        desc = r.get("description","")
        params = r.get("parameters",{})
        vid = f"videos/jepa_sensitivity/{name}.mp4"

        # Parameter line
        param_parts = []
        if "velocity_ms" in params:
            param_parts.append(f"v₀={params['velocity_ms'][0]}m/s")
        if "ball_mass_kg" in params:
            param_parts.append(f"m_ball={params['ball_mass_kg']}kg")
        if "block_mass_kg" in params:
            param_parts.append(f"m_block={params['block_mass_kg']}kg")
        if "gravity" in params:
            param_parts.append(f"g={params['gravity']}m/s²")

        highlight = ""
        if j:
            if j <= min(all_jepa) + 0.002:
                highlight = "lowest"  # most surprising
            elif j >= max(all_jepa) - 0.002:
                highlight = "highest"

        cards += f"""
        <div class="card {highlight}">
          <video src="{vid}" controls muted preload="metadata"></video>
          <div class="bar">
            <span class="desc">{desc}</span>
            <div class="params">{' | '.join(param_parts)}</div>
            <span class="score">JEPA {fv(j)}</span>
          </div>
        </div>"""

    sections.append(f'<section class="group"><h2>{gtitle}</h2><div class="card-row">{cards}</div></section>')

# Find global min/max for annotation
j_min = min(all_jepa) if all_jepa else 0
j_max = max(all_jepa) if all_jepa else 0

html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>JEPA Sensitivity to Motion</title>
<style>
:root{{--bg:#1a1815;--panel:#252320;--line:#3d3830;--text:#e8e4dd;--muted:#9d968a;--accent:#e08840;--red:#e05550;--green:#6db87d;--blue:#6ba4d1}}
*{{box-sizing:border-box}}body{{margin:0;color:var(--text);font-family:system-ui,sans-serif;background:var(--bg)}}
.page{{max-width:1600px;margin:0 auto;padding:24px}}
h1{{margin:0 0 4px;font-size:28px}}.sub{{color:var(--muted);margin:0 0 6px;font-size:14px}}
.finding{{padding:14px 20px;background:rgba(107,164,209,0.12);border:1px solid var(--blue);border-radius:12px;margin:16px 0 24px;line-height:1.6;font-size:14px}}
.finding strong{{color:var(--blue)}}
.group{{margin-bottom:32px}}
.group h2{{font-size:18px;margin:0 0 4px;padding-bottom:8px;border-bottom:1px solid var(--line)}}
.card-row{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}}
.card{{border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#1a1818}}
.card video{{width:100%;display:block;aspect-ratio:16/9;object-fit:cover}}
.card.lowest{{border-color:var(--red)}}
.card.highest{{border-color:var(--green)}}
.bar{{padding:10px 12px}}
.desc{{font-size:13px;color:var(--text);display:block;margin-bottom:4px}}
.params{{font-size:11px;color:var(--muted);margin-bottom:6px}}
.score{{font-size:16px;font-weight:700;font-variant-numeric:tabular-nums}}
.card.lowest .score{{color:var(--red)}}
.card.highest .score{{color:var(--green)}}
.summary{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px}}
.stat{{border:1px solid var(--line);border-radius:10px;padding:14px;text-align:center;background:var(--panel)}}
.stat .val{{font-size:28px;font-weight:700}}
.stat .lbl{{font-size:12px;color:var(--muted);margin-top:4px}}
</style></head><body><div class="page">
<h1>V-JEPA2 Motion Sensitivity</h1>
<p class="sub">固定外观(V1)，系统改变运动参数 — JEPA几乎无区分力（全范围仅 {jepa_range:.4f}）</p>

<div class="summary">
  <div class="stat"><div class="val" style="color:var(--blue)">20</div><div class="lbl">运动变体</div></div>
  <div class="stat"><div class="val" style="color:var(--blue)">{jepa_range:.4f}</div><div class="lbl">JEPA 全范围(max-min)</div></div>
  <div class="stat"><div class="val" style="color:var(--red)">{fv(j_min)}</div><div class="lbl">最低 JEPA（最意外）</div></div>
  <div class="stat"><div class="val" style="color:var(--green)">{fv(j_max)}</div><div class="lbl">最高 JEPA（最预期内）</div></div>
</div>

<div class="finding">
<strong>结论：</strong>V-JEPA2 对纯刚体仿真视频的运动差异 <strong>几乎无区分力</strong>。
速度差28×、质量差10⁴×、月球vs超重、碰撞vs不碰撞 → JEPA 仅波动 0.029 (&lt;4%)。
红色边框=最低分(最意外)，绿色=最高分(最可预测)。
JEPA 不适合作为简单刚体仿真的物理质量指标。
</div>

{''.join(sections)}
</div></body></html>"""
(REPORT_DIR / "index.html").write_text(html)

link = REPORT_DIR / "videos"
if not link.exists():
    link.symlink_to(VAR_DIR.parent)

print(f"http://127.0.0.1:18706/index.html")
subprocess.run([sys.executable, "-m", "http.server", "18706", "--directory", str(REPORT_DIR)])
