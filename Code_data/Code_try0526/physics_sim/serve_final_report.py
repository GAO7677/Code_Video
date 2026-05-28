#!/usr/bin/env python3
"""PDI + WMReward JEPA 最终对比报告"""

import csv, html, json, sys, subprocess
from pathlib import Path

DATA = Path("/data/gaoya/AAA_test_video/Dataset_physV/0526dp/wmreward_report")
DATA.mkdir(parents=True, exist_ok=True)

# ── Table from CSV ──
with open("/data/gaoya/AAA_test_video/Output_try0526/PDI-Bench/result/metrics.csv") as f:
    csv_rows = list(csv.DictReader(f))

def _th(k):
    label = k.replace("_", " ")
    if k in ("wmreward_jepa",):
        return f"<th>{label} ↑</th>"
    if k in ("method", "num_videos"):
        return f"<th>{label}</th>"
    return f"<th>{label} ↓</th>"

thead = "<tr>" + "".join(_th(k) for k in csv_rows[0]) + "</tr>"
tbody = ""
for r in csv_rows:
    tbody += "<tr>" + "".join(f"<td class='n'>{v}</td>" for v in r.values()) + "</tr>"

# ── Representative cases ──
CASES = [
    {
        "group": "PDI-Bench: Real vs Generated",
        "cards": [
            {"label": "GT · blackswan",     "sub": "Most surprising JEPA (0.380)", "video": "pdi_videos/GT/Biological_Motion/blackswan.mp4", "pdi": "0.144*", "wmr": "0.380", "hl": "red"},
            {"label": "Wan · rhino",         "sub": "Least surprising JEPA (0.466)", "video": "pdi_videos/wan22-5B-TI2V/partial_occlusion/rhino.mp4", "pdi": "0.878*", "wmr": "0.466", "hl": "green"},
            {"label": "GT · ball",           "sub": "Best geometry (PDI=0.144)", "video": "pdi_videos/GT/Longitudinal_Convergence/ball.mp4", "pdi": "0.144*", "wmr": "0.446", "hl": "blue"},
            {"label": "Wan · ball",          "sub": "Worst geometry (PDI=0.878)", "video": "pdi_videos/wan22-5B-TI2V/Longitudinal_Convergence/ball.mp4", "pdi": "0.878*", "wmr": "0.448", "hl": "red"},
        ]
    },
    {
        "group": "Simulation: Ball-Block",
        "cards": [
            {"label": "e=0.7 baseline",      "sub": "PDI=0.022 (Grade A)", "video": "sim_videos/ball_block/e07_mu05_m1.mp4", "pdi": "0.022", "wmr": "0.457", "hl": "blue"},
            {"label": "m=0.1kg light-ball",  "sub": "PDI=2.064 (Grade F)", "video": "sim_videos/ball_block/e07_mu05_m01.mp4", "pdi": "2.064", "wmr": "0.467", "hl": "red"},
            {"label": "m=5.0kg heavy-ball",  "sub": "Best JEPA (0.469)", "video": "sim_videos/ball_block/e07_mu05_m5.mp4", "pdi": "0.139", "wmr": "0.469", "hl": "green"},
            {"label": "u=1.0 high-fric",     "sub": "Worst JEPA (0.450)", "video": "sim_videos/ball_block/e07_mu10_m1.mp4", "pdi": "0.138", "wmr": "0.450", "hl": "red"},
        ]
    },
    {
        "group": "Sanity Check: Frame Shuffle",
        "cards": [
            {"label": "GT ball · Original",  "sub": "PDI=0.175 WMR=0.454", "video": "shuffle_videos/gt_ball_original.mp4", "pdi": "0.175", "wmr": "0.454", "hl": "green"},
            {"label": "GT ball · SHUFFLED",  "sub": "FRAMES RANDOMIZED — PDI=0.102 WMR=0.439", "video": "shuffle_videos/gt_ball_shuffled.mp4", "pdi": "0.102", "wmr": "0.439", "hl": "red"},
        ]
    },
]

sections = ""
for g in CASES:
    cards = ""
    for c in g["cards"]:
        cards += f"""<div class="card {c['hl']}">
  <video src="{c['video']}" controls muted preload="metadata"></video>
  <div class="info"><div class="label">{c['label']}</div><div class="sub">{c['sub']}</div>
  <div class="scores"><span>PDI {c['pdi']}</span><span>WMR {c['wmr']}</span></div></div>
</div>"""
    sections += f'<section><h2>{g["group"]}</h2><div class="grid">{cards}</div></section>'

html_tmpl = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>PDI vs WMReward JEPA</title>
<style>
:root{{--bg:#1a1815;--panel:#252320;--line:#3d3830;--text:#e8e4dd;--muted:#9d968a;--accent:#e08840;--red:#e05550;--green:#6db87d;--blue:#6ba4d1}}
*{{box-sizing:border-box}}body{{margin:0;color:var(--text);font-family:system-ui,sans-serif;background:var(--bg)}}
.page{{max-width:1200px;margin:0 auto;padding:24px}}h1{{margin:0 0 4px;font-size:24px}}.sub{{color:var(--muted);margin:0 0 6px;font-size:13px}}
.finding{{padding:12px 16px;background:rgba(224,136,64,0.1);border:1px solid var(--accent);border-radius:8px;margin:12px 0 20px;font-size:13px;line-height:1.6}}
table{{width:100%;border-collapse:collapse;background:var(--panel);border-radius:8px;overflow:hidden;margin-bottom:24px}}
th,td{{padding:8px 12px;border-bottom:1px solid var(--line);font-size:13px;white-space:nowrap}}
th{{background:rgba(255,255,255,0.05);font-size:11px;text-transform:uppercase}}
.n{{text-align:right;font-variant-numeric:tabular-nums}}td:first-child{{text-align:left;font-weight:600}}
h2{{font-size:15px;margin:24px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}}
.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
.card{{border-radius:10px;overflow:hidden;background:#1a1818;border:1px solid var(--line)}}
.card video{{width:100%;display:block;aspect-ratio:16/9;object-fit:cover}}
.card.red{{border-color:var(--red)}}.card.green{{border-color:var(--green)}}.card.blue{{border-color:var(--blue)}}
.info{{padding:10px 12px}}.label{{font-size:12px;font-weight:700}}.sub{{font-size:11px;color:var(--muted);margin:2px 0 6px}}
.scores{{display:flex;gap:12px;font-size:16px;font-weight:700;font-variant-numeric:tabular-nums}}
.scores span:first-child{{color:var(--blue)}}.scores span:last-child{{color:var(--accent)}}
.card.red .scores span:last-child{{color:var(--red)}}.card.green .scores span:last-child{{color:var(--green)}}
@media(max-width:900px){{.grid{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><div class="page">
<h1>PDI-Bench vs WMReward JEPA</h1>
<p class="sub">PDI (geometric consistency, lower=better) &nbsp;|&nbsp; WMReward JEPA (predictive similarity, higher=better) &nbsp;|&nbsp; * = method average</p>

<div class="finding">
<strong>PDI and JEPA ranks are inverted:</strong> GT has the best PDI (0.144) but the worst JEPA (0.412). Wan has the worst PDI (0.878) but the best JEPA (0.435).
Real videos are geometrically correct but visually unpredictable; generated videos are geometrically flawed but visually simple.
The two metrics capture <strong>complementary</strong> dimensions of video quality.<br><br>
<strong>Sanity check:</strong> Randomly shuffling all frames in a GT video → PDI paradoxically <strong>improves</strong> (0.175→0.102, SAM2 tracking breaks on shuffled frames),
WMReward JEPA drops only <strong>0.015</strong> (0.454→0.439). Complete temporal chaos barely registers for V-JEPA2.
</div>

<table>{thead}{tbody}</table>
{sections}
</div></body></html>"""

(DATA / "index.html").write_text(html_tmpl.format(thead=thead, tbody=tbody, sections=sections))

# Symlinks
for name, target in [
    ("sim_videos", Path("/data/gaoya/AAA_test_video/Dataset_physV/0526dp/videos")),
    ("pdi_videos", Path("/data/gaoya/AAA_test_video/Output_try0526/PDI-Bench/output")),
    ("shuffle_videos", Path("/data/gaoya/AAA_test_video/Dataset_physV/0526dp/videos/shuffle_test")),
]:
    link = DATA / name
    if not link.exists(): link.symlink_to(target)

subprocess.run([sys.executable, "-m", "http.server", "18707", "--directory", str(DATA)])
