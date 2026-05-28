#!/usr/bin/env python3
"""WMReward vs Old JEPA — 视频 + 分数并排对比"""

import json, sys, subprocess
from pathlib import Path

DATA_DIR = Path("/data/gaoya/AAA_test_video/Dataset_physV/0526dp")
DIRS = [
    ("ball_block", "Baseline (8 scenarios)"),
    ("jepa_sensitivity", "Motion Sensitivity (20 scenarios)"),
]
REPORT_DIR = DATA_DIR / "wmreward_report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def fv(v): return f"{v:.4f}" if isinstance(v,(int,float)) else str(v)

sections = ""
for dname, dlabel in DIRS:
    vdir = DATA_DIR / "videos" / dname
    if not vdir.exists(): continue
    cards = ""
    for jp in sorted(vdir.glob("*.json")):
        d = json.loads(jp.read_text())
        name = d.get("scenario", jp.stem)
        desc = d.get("description", "")
        old_j = d.get("jepa", {}).get("jepa_score")
        wmr = d.get("wmreward_jepa", {})
        wmr_sim = wmr.get("similarity")
        wmr_surp = wmr.get("surprise")
        delta = wmr_sim - old_j if (wmr_sim and old_j) else None
        vid = f"videos/{dname}/{jp.stem}.mp4"

        # Score bar visualization: normalize 0.44-0.50 for WMR, 0.72-0.76 for old
        def bar(val, lo, hi, color):
            if val is None: return ""
            pct = max(0, min(100, (val-lo)/(hi-lo)*100))
            return f'<div class="bar-bg"><div class="bar-fill" style="width:{pct:.0f}%;background:{color}"></div></div>'

        cards += f"""
        <div class="card">
          <video src="{vid}" controls muted preload="metadata"></video>
          <div class="info">
            <div class="name">{name}</div>
            <div class="desc">{desc[:80]}</div>
            <div class="scores">
              <div class="score-block old">
                <div class="score-label">Old JEPA</div>
                <div class="score-val">{fv(old_j)}</div>
                {bar(old_j, 0.725, 0.755, 'var(--muted)')}
              </div>
              <div class="score-block wmr">
                <div class="score-label">WMReward</div>
                <div class="score-val">{fv(wmr_sim)}</div>
                {bar(wmr_sim, 0.445, 0.500, 'var(--accent)')}
              </div>
              <div class="score-block delta">
                <div class="score-label">Δ</div>
                <div class="score-val" style="color:{'var(--red)' if delta and delta<0 else 'var(--green)'}">{fv(delta) if delta else '-'}</div>
              </div>
            </div>
          </div>
        </div>"""

    sections += f"""
    <section class="group">
      <h2>{dlabel}</h2>
      <div class="card-grid">{cards}</div>
    </section>"""

html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>WMReward vs Old JEPA</title>
<style>
:root{{--bg:#1a1815;--panel:#252320;--line:#3d3830;--text:#e8e4dd;--muted:#9d968a;--accent:#e08840;--red:#e05550;--green:#6db87d}}
*{{box-sizing:border-box}}body{{margin:0;color:var(--text);font-family:system-ui,sans-serif;background:var(--bg)}}
.page{{max-width:1600px;margin:0 auto;padding:24px}}
h1{{margin:0 0 4px;font-size:26px}}.sub{{color:var(--muted);margin:0 0 18px;font-size:14px}}
.finding{{padding:12px 18px;background:rgba(224,136,64,0.1);border:1px solid var(--accent);border-radius:10px;margin:0 0 24px;font-size:13px;line-height:1.6}}
.finding strong{{color:var(--accent)}}
.group h2{{font-size:16px;margin:24px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}}
.card-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}}
.card{{border:1px solid var(--line);border-radius:10px;overflow:hidden;background:#1a1818;transition:border-color .2s}}
.card:hover{{border-color:var(--accent)}}
.card video{{width:100%;display:block;aspect-ratio:16/9;object-fit:cover}}
.info{{padding:10px 12px}}
.name{{font-size:13px;font-weight:700;margin-bottom:2px}}
.desc{{font-size:11px;color:var(--muted);margin-bottom:8px;line-height:1.3}}
.scores{{display:flex;gap:8px}}
.score-block{{flex:1;text-align:center}}
.score-label{{font-size:10px;color:var(--muted);text-transform:uppercase;margin-bottom:2px}}
.score-val{{font-size:17px;font-weight:700;font-variant-numeric:tabular-nums}}
.score-block.old .score-val{{color:var(--muted)}}
.score-block.wmr .score-val{{color:var(--accent)}}
.bar-bg{{height:3px;background:rgba(255,255,255,.08);border-radius:2px;margin-top:3px}}
.bar-fill{{height:100%;border-radius:2px}}
.summary{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:20px}}
.stat{{border:1px solid var(--line);border-radius:10px;padding:12px;text-align:center;background:var(--panel)}}
.stat .val{{font-size:26px;font-weight:700}}.stat .lbl{{font-size:11px;color:var(--muted);margin-top:2px}}
@media(max-width:800px){{.card-grid{{grid-template-columns:repeat(2,1fr)}}.summary{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><div class="page">
<h1>WMReward vs Old Custom JEPA</h1>
<p class="sub">Same video, two scoring methods — Old: custom 3-way composite (cos+Gram+delta) &nbsp;|&nbsp; WMR: Meta official sliding-window cosine distance</p>

<div class="summary">
  <div class="stat"><div class="val" style="color:var(--accent)">28</div><div class="lbl">videos</div></div>
  <div class="stat"><div class="val" style="color:var(--red)">0.28</div><div class="lbl">avg |Δ| (WMR − Old)</div></div>
  <div class="stat"><div class="val" style="color:var(--green)">0.045</div><div class="lbl">WMReward range</div></div>
  <div class="stat"><div class="val" style="color:var(--muted)">0.029</div><div class="lbl">Old JEPA range</div></div>
</div>

<div class="finding">
<strong>V-JEPA2 对合成刚体仿真视频区分力很弱</strong>，无论用旧自定义方法还是 WMReward 官方方法。
WMReward 区分力稍强（range 0.045 vs 0.029），但绝对值低（~0.47 = 高 surprise）。
Score bar 范围：Old 0.725-0.755 | WMR 0.445-0.500
</div>
{sections}
</div></body></html>"""
(REPORT_DIR / "index.html").write_text(html)

link = REPORT_DIR / "videos"
if not link.exists():
    link.symlink_to(DATA_DIR / "videos")

subprocess.run([sys.executable, "-m", "http.server", "18706", "--directory", str(REPORT_DIR)])
