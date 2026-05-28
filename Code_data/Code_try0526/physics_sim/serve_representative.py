#!/usr/bin/env python3
"""最具代表性 case 的指标对比"""

import json, sys, subprocess
from pathlib import Path

DATA_DIR = Path("/data/gaoya/AAA_test_video/Dataset_physV/0526dp/wmreward_report")

CASES = [
    {
        "group": "PDI-Bench: Real vs Generated",
        "cards": [
            {"label": "GT · Biological Motion · blackswan",
             "desc": "Most SURPRISING<br>Real video with complex motion",
             "video": "pdi_videos/GT/Biological_Motion/blackswan.mp4",
             "pdi_label": "GT mean PDI", "pdi": 0.144,
             "wmr": 0.380, "highlight": "surprise"},
            {"label": "Wan · partial occlusion · rhino",
             "desc": "Least SURPRISING<br>Generated video with simple motion",
             "video": "pdi_videos/wan22-5B-TI2V/partial_occlusion/rhino.mp4",
             "pdi_label": "Wan mean PDI", "pdi": 0.878,
             "wmr": 0.466, "highlight": "predictable"},
            {"label": "GT · Longitudinal · ball",
             "desc": "GT reference<br>Real video — best geometry",
             "video": "pdi_videos/GT/Longitudinal_Convergence/ball.mp4",
             "pdi_label": "GT mean PDI", "pdi": 0.144,
             "wmr": 0.446, "highlight": "pdi_best"},
            {"label": "Wan · Longitudinal · ball",
             "desc": "Generated<br>Worst geometry overall",
             "video": "pdi_videos/wan22-5B-TI2V/Longitudinal_Convergence/ball.mp4",
             "pdi_label": "Wan mean PDI", "pdi": 0.878,
             "wmr": 0.448, "highlight": "pdi_worst"},
        ]
    },
    {
        "group": "Simulation: Ball-Block",
        "cards": [
            {"label": "e=0.7 baseline (PDI best)",
             "desc": "PDI=0.022 Grade A<br>Near-perfect geometric consistency",
             "video": "sim_videos/ball_block/e07_mu05_m1.mp4",
             "pdi_label": "PDI", "pdi": 0.022,
             "wmr": 0.457, "highlight": "pdi_best"},
            {"label": "m=0.1kg light ball (PDI worst)",
             "desc": "PDI=2.064 Grade F<br>Light ball bounces erratically",
             "video": "sim_videos/ball_block/e07_mu05_m01.mp4",
             "pdi_label": "PDI", "pdi": 2.064,
             "wmr": 0.467, "highlight": "pdi_worst"},
            {"label": "m=5.0kg heavy ball (JEPA best)",
             "desc": "WMR=0.469<br>Most predictable motion",
             "video": "sim_videos/ball_block/e07_mu05_m5.mp4",
             "pdi_label": "PDI", "pdi": 0.139,
             "wmr": 0.469, "highlight": "predictable"},
            {"label": "μ=1.0 high-fric (JEPA worst)",
             "desc": "WMR=0.450<br>Most surprising motion with grip",
             "video": "sim_videos/ball_block/e07_mu10_m1.mp4",
             "pdi_label": "PDI", "pdi": 0.138,
             "wmr": 0.450, "highlight": "surprise"},
        ]
    },
]

def fv(v): return f"{v:.4f}" if isinstance(v,(int,float)) else str(v)

DATA_DIR.mkdir(parents=True, exist_ok=True)
sections = ""
for g in CASES:
    cards = ""
    for c in g["cards"]:
        cards += f"""
        <div class="card {c['highlight']}">
          <video src="{c['video']}" controls muted preload="metadata"></video>
          <div class="info">
            <div class="label">{c['label']}</div>
            <div class="desc">{c['desc']}</div>
            <div class="scores">
              <div class="s"><span class="sl">{c['pdi_label']}</span><span class="sv pdi">{fv(c['pdi'])}</span></div>
              <div class="s"><span class="sl">WMR JEPA</span><span class="sv jepa">{fv(c['wmr'])}</span></div>
            </div>
          </div>
        </div>"""
    sections += f'<section class="group"><h2>{g["group"]}</h2><div class="grid">{cards}</div></section>'

html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Representative Cases — PDI vs JEPA</title>
<style>
:root{{--bg:#1a1815;--panel:#252320;--line:#3d3830;--text:#e8e4dd;--muted:#9d968a;--accent:#e08840;--red:#e05550;--green:#6db87d;--blue:#6ba4d1}}
*{{box-sizing:border-box}}body{{margin:0;color:var(--text);font-family:system-ui,sans-serif;background:var(--bg)}}
.page{{max-width:1200px;margin:0 auto;padding:24px}}h1{{margin:0 0 4px;font-size:24px}}.sub{{color:var(--muted);margin:0 0 20px;font-size:14px}}
.finding{{padding:14px 18px;background:rgba(224,136,64,0.1);border:1px solid var(--accent);border-radius:10px;margin:0 0 24px;font-size:13px;line-height:1.7}}
.finding strong{{color:var(--accent)}}
.group h2{{font-size:15px;margin:24px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
.card{{border-radius:10px;overflow:hidden;background:#1a1818;border:1px solid var(--line)}}
.card video{{width:100%;display:block;aspect-ratio:16/9;object-fit:cover}}
.card.surprise{{border-color:var(--red)}}
.card.predictable{{border-color:var(--green)}}
.card.pdi_best{{border-color:var(--blue)}}
.card.pdi_worst{{border-color:var(--red)}}
.info{{padding:10px 12px}}.label{{font-size:12px;font-weight:700;margin-bottom:4px}}
.desc{{font-size:11px;color:var(--muted);margin-bottom:8px;line-height:1.4}}
.scores{{display:flex;gap:12px}}.s{{flex:1}}.sl{{font-size:10px;color:var(--muted);display:block}}.sv{{font-size:20px;font-weight:700;font-variant-numeric:tabular-nums}}
.pdi{{color:var(--blue)}}.jepa{{color:var(--accent)}}
.surprise .jepa{{color:var(--red)}}.predictable .jepa{{color:var(--green)}}
.insight{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:20px}}
.insight .box{{padding:12px 16px;border-radius:10px;background:var(--panel);border:1px solid var(--line);font-size:13px;line-height:1.6}}
.insight .box strong{{display:block;margin-bottom:4px}}
@media(max-width:900px){{.grid{{grid-template-columns:repeat(2,1fr)}}.insight{{grid-template-columns:1fr}}}}
</style></head><body><div class="page">
<h1>PDI vs JEPA — Representative Cases</h1>
<p class="sub">Geometric correctness (PDI) vs predictive plausibility (WMReward JEPA) — complementary signals</p>

<div class="insight">
  <div class="box"><strong style="color:var(--red)">PDI-Bench: Real videos</strong>GT has PDI=0.144 (best geometry) but WMR=0.380-0.446 (most surprising). Real-world motion complexity confuses V-JEPA2.</div>
  <div class="box"><strong style="color:var(--green)">PDI-Bench: Generated videos</strong>Wan has PDI=0.878 (worst geometry) but WMR=0.412-0.466 (most predictable). Simple, repetitive motion is easy for V-JEPA2.</div>
  <div class="box"><strong style="color:var(--blue)">Simulation: Geometric extremes</strong>PDI correctly identifies light ball (0.1kg, PDI=2.06 F) as geometrically broken vs baseline (0.022 A). PDI distinguishes physics quality.</div>
  <div class="box"><strong style="color:var(--accent)">Simulation: JEPA blind spots</strong>High-fric (WMR=0.450) vs heavy-ball (WMR=0.469). 0.019 range across all simulations — V-JEPA2 barely discriminates rigid-body motion.</div>
</div>

{sections}
</div></body></html>"""
(DATA_DIR / "representative.html").write_text(html)

# Ensure videos symlink
link = DATA_DIR / "videos"
if not link.exists():
    link.symlink_to(Path("/data/gaoya/AAA_test_video/Dataset_physV/0526dp") / "videos")

subprocess.run([sys.executable, "-m", "http.server", "18706", "--directory", str(DATA_DIR)])
